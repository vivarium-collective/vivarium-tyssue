"""A subcellular SBML cell-cycle model in every cell of a growing tissue (SIMULATION).

Demonstrates two things:

  * **SBML integration with tyssue.** The Gérard & Goldbeter 2011 skeleton model of
    the mammalian Cdk network (BioModels BIOMD0000000941) is read straight from its
    SBML file and compiled once into a vectorised numpy right-hand side
    (``vivarium_tyssue.sbml.VectorizedSBML``).
  * **One copy of that model per cell.** The ``SBMLCellCycle`` process integrates a
    6-variable state for every living face and reads each cell's phase from it.
    A cell grows in G1 and G2, holds its size in S and M, and divides on entering
    M; daughters inherit their mother's state.

One scenario, **growth**: the flat square sheet (``test_square.hf5``, 206 cells)
with a free border (``model_factory``), growth factor GF = 1 in every cell, grown
from 206 to ``TARGET_CELLS`` cells.

A freely growing vertex sheet eventually folds: faces pass through each other,
since the model has no anti-self-intersection term. How large it gets first is set
by how fast the mechanics relax compared with growth (area elasticity / viscosity;
see ``SWEEPS.md``). The low viscosity below keeps it clean to ~1500 cells, so the
run stops at ``TARGET_CELLS``. Every frame is checked with tyssue's exact
``find_intersections``; the run also stops at a first fold.

The run is advanced one ``FRAME_DT`` at a time and the mesh is snapshotted after
each: at dt = 0.002 the solver's per-step History would hold ~30,000 frames.

This script only runs the simulation and archives its data:
``outputs/growth/history.hf5`` (vert/edge/face snapshots, including every cell's
``cc_*`` model state) and ``outputs/growth/summary.csv``. Model description,
single-cell behaviour, figures and GIFs are in ``sbml_cell_cycle_analysis.ipynb``.

Run from the repo's ``vivarium-tyssue`` conda env:

    cd Experiments/sbml_cell_cycle
    python sbml_cell_cycle.py
"""
from __future__ import annotations

import os

# Pin BLAS to one thread before numpy loads: multithreaded reductions make
# same-seed runs diverge through T1 flips.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import copy  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
FLAT_DATASET = "test_square.hf5"
MODEL_SBML = REPO / "workspace" / "datasets" / "BIOMD0000000941.xml"
SEED = 20260930
SCENARIO = "growth"

# The SBML model and its coupling.
# Cell-cycle phases from each cell's own cyclins, first match wins, else G1.
# Timed from mitotic exit along the GF = 1 limit cycle (period 23.6): S 4.4-13.1,
# G2 13.1-17.6, M 17.6-22.2 (cyclin A being destroyed), G1 the rest.
PHASES = [
    "S: cyclin_E_Cdk2 > 1",
    "M: cyclin_B_Cdk1 > 1 & cyclin_E_Cdk2 <= 1 & cyclin_A_Cdk2 < 2",
    "G2: cyclin_B_Cdk1 > 1 & cyclin_E_Cdk2 <= 1",
]
PHASE_NAMES = ["G1", "S", "M", "G2"]     # SBMLCellCycle.phase_names for PHASES
GROWTH_PHASES, DIVISION_PHASE = ["G1", "G2"], "M"
GROWTH_FRACTION = 0.44              # share of the cycle spent in G1 + G2
GROWTH_FACTOR = 1.0                 # GF in every cell (full cycles need GF > ~0.072)
MODEL_PERIOD = 23.6                 # model time units at GF = 1
CYCLE_TIME = 20.0                   # desired cycle length in sim time units
TIME_SCALE = MODEL_PERIOD / CYCLE_TIME
# Target area doubles over the G1 + G2 time of one cycle; each half restarts at 1.
GROWTH_RATE = np.log(2.0) / (GROWTH_FRACTION * CYCLE_TIME)
BIRTH_AREA = 1.0
# Per-cell lognormal noise on the synthesis/degradation rates the period is most
# sensitive to (±20% moves the period ~±8%), so sisters drift apart.
PARAMETER_NOISE = {"vsa": 0.1, "vse": 0.1, "Vde": 0.1}

# Mechanics (measured with tyssue's find_intersections on the free sheet; see
# SWEEPS.md):
#   * fast relaxation is what keeps a growing sheet clean. The cells-before-first-
#     fold scales with area_elasticity / viscosity: ~400 cells at viscosity 0.2,
#     ~930 at 0.05, ~1500-1600 at 0.02. dt shrinks with viscosity to keep Euler
#     stable; dt alone changes nothing.
#   * perimeter tension: prefered_perimeter 3.5 (below the hexagon's 3.72) with
#     perimeter_elasticity 0.5, as in the crypt. Raising P0 toward 3.72 folds sooner.
#   * growth spread over G1 + G2 (~9 time units per cycle), not in a burst.
DT = 0.002
PREF_PERIMETER, PERIMETER_ELASTICITY, VISCOSITY = 3.5, 0.5, 0.02
TARGET_CELLS = 1200    # stop here: clean on every seed tried at this viscosity
FRAME_DT = 0.25        # snapshot (and fold check) interval
TF = 80.0              # safety cap; the target is reached near t = 60

# A cell counts as cycling if it divided within this long (sim time).
CYCLING_WINDOW = 1.5 * CYCLE_TIME


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def ensure_dataset() -> Path:
    """The flat square (206 cells), copied into ``data/``. Its border is free:
    with ``model_factory`` (not ``model_factory_bound``, which zeroes the force on
    every ``boundary`` vertex) the sheet spreads as it grows."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_DIR / FLAT_DATASET
    if not dst.exists():
        shutil.copy(REPO / "workspace" / "datasets" / FLAT_DATASET, dst)
    return dst


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
def state_columns():
    from vivarium_tyssue.sbml import VectorizedSBML
    ids = VectorizedSBML(MODEL_SBML).state_ids
    return [f"cc_{s}" for s in ids] + ["cc_phase", "cc_last_division"]


def solver_config(dataset_path: Path) -> dict:
    face_params = {
        "area_elasticity": 1.0,
        "prefered_area": 1.0,
        "perimeter_elasticity": PERIMETER_ELASTICITY,
        "prefered_perimeter": PREF_PERIMETER,
        "is_alive": 1.0,
        "cell_type": "normal",
        "mother_uid": -1,          # set on each daughter by split_cells
    }
    # Declared so the History records every cell's model state (and daughters
    # inherit it through the copied face row).
    face_params.update({col: 0.0 for col in state_columns()})
    return {
        "name": "Growing Flat Square",
        "eptm": str(dataset_path),
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": face_params,
            "edge_df": {"line_tension": 0.0, "is_active": 1.0},
            "vert_df": {"viscosity": VISCOSITY, "is_alive": 1.0},
        },
        "geom": "SheetGeometry",
        "effectors": ["LineTension", "FaceAreaElasticity", "PerimeterElasticity"],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory",       # free border: the tissue spreads
        "settings": {"threshold_length": 0.03},
        "auto_reconnect": {"enabled": True},
        "bounds": None,
        "output_columns": {},
        "history_columns": {},
        "maps": {},
        "backend": "python",
        "substeps": 1,
        "max_displacement": 0.0,
        "record_history": False,   # run() snapshots every FRAME_DT instead
    }


def cell_cycle_config() -> dict:
    return {
        "model_source": str(MODEL_SBML),
        "time_scale": TIME_SCALE,
        "parameters": {"GF": GROWTH_FACTOR},
        "phases": PHASES,
        "growth_phases": GROWTH_PHASES,
        "division_phase": DIVISION_PHASE,
        "growth_rate": GROWTH_RATE,
        "birth_area": BIRTH_AREA,
        "parameter_noise": PARAMETER_NOISE,
        "geom": "SheetGeometry",
        "initial_state": "limit_cycle",
    }


def build_spec(dataset_path: Path) -> dict:
    return {
        "Tyssue": {
            "_type": "process",
            "address": "local:EulerSolver",
            "config": solver_config(dataset_path),
            "inputs": {"behaviors": ["Behaviors"], "global_time": ["global_time"]},
            "outputs": {
                "datasets": ["Tissue State"],
                "network_changed": ["Network Changed"],
                "behaviors_update": ["Behaviors"],
            },
            "interval": DT,
        },
        "Cell Cycle": {
            "_type": "process",
            "address": "local:SBMLCellCycle",
            "config": cell_cycle_config(),
            "inputs": {"global_time": ["global_time"], "datasets": ["Tissue State"]},
            "outputs": {"behaviors": ["Behaviors"]},
            "interval": DT,
        },
        "Network Changed": False,
        "Behaviors": {},
    }


# ---------------------------------------------------------------------------
# Run / archive
# ---------------------------------------------------------------------------
def seed_all(seed: int = SEED):
    """Seed numpy's global stream (initial phases, parameter noise) and Python's
    ``random`` (tyssue's EventManager shuffles its queue with it)."""
    np.random.seed(seed)
    random.seed(seed)


# Columns a frame needs: the drawing rebuilds a Sheet from vert positions, the
# edge topology and every face column (the cc_* state, cell_type, unique_id).
SNAP_COLUMNS = {"vert": ["x", "y", "z"], "edge": ["srce", "trgt", "face"]}


def _snapshot(eptm, t: float) -> dict:
    out = {}
    for el in ("vert", "edge", "face"):
        df = eptm.datasets[el]
        df = df[SNAP_COLUMNS[el]] if el in SNAP_COLUMNS else df
        df = df.copy()
        df.index.name = el
        df = df.reset_index()
        df["time"] = t
        out[el] = df
    return out


def run(core, spec: dict, target: int = TARGET_CELLS, tf: float = TF):
    """Advance a spec one FRAME_DT at a time, snapshotting the mesh after each.
    Stops at ``target`` cells, at the first fold, or at ``tf``.
    Returns (datasets, cell-cycle process, final epithelium, outcome)."""
    from process_bigraph import Composite

    sim = Composite({"state": copy.deepcopy(spec)}, core=core)
    eptm = sim.state["Tyssue"]["instance"].eptm
    frames = [_snapshot(eptm, 0.0)]
    t, outcome = 0.0, "tf"
    while t < tf - 1e-9:
        sim.run(FRAME_DT)
        t = round(t + FRAME_DT, 6)
        eptm = sim.state["Tyssue"]["instance"].eptm
        frames.append(_snapshot(eptm, t))
        if not eptm.find_intersections(strict=True).clean:
            outcome = "folded"
            break
        if eptm.Nf >= target:
            outcome = "target"
            break
    datasets = {el: pd.concat([f[el] for f in frames], ignore_index=True)
                for el in ("vert", "edge", "face")}
    return datasets, sim.state["Cell Cycle"]["instance"], eptm, outcome


def _sanitize_for_hdf(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns -> numeric where they all convert, else str (PyTables)."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype != object:
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        notnull = df[c].notna()
        if notnull.any() and (conv.notna() | ~notnull).all():
            df[c] = conv
        else:
            df[c] = df[c].astype(str)
    return df


def save_history(datasets: dict, path: Path) -> int:
    """Archive the snapshots, one table per element with a ``time`` column (the
    layout of tyssue's ``HistoryHdf5``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with pd.HDFStore(path, "a", complevel=5, complib="blosc:zstd") as store:
        for key, df in datasets.items():
            store.append(key=key, value=_sanitize_for_hdf(df), data_columns=["time"])
    return len(datasets["face"]["time"].unique())


def summarise(datasets: dict) -> pd.DataFrame:
    """Per recorded time: cells, divisions so far, cycling cells, area, strain and
    the tissue's area."""
    faces = datasets["face"]
    rows = []
    n0 = None
    for t, fd in faces.groupby("time"):
        if "is_alive" in fd.columns:
            fd = fd[fd["is_alive"] > 0]
        n = len(fd)
        n0 = n if n0 is None else n0
        cycling = (t - fd["cc_last_division"]) < CYCLING_WINDOW
        rows.append({
            "time": t, "n_cells": n, "n_cycling": int(cycling.sum()),
            "cycling_fraction": float(cycling.mean()),
            "n_growing": int(fd["cc_phase"].isin(
                [PHASE_NAMES.index(p) for p in GROWTH_PHASES]).sum()),
            "divisions": n - n0,
            "mean_area": float(fd["area"].mean()),
            "mean_strain": float((fd["area"] / fd["prefered_area"]).mean()),
            "tissue_area": float(fd["area"].sum()),
        })
    return pd.DataFrame(rows)


def main():
    sys.path.insert(0, str(REPO))
    from vivarium_tyssue.core import build_core

    out = OUT_DIR / SCENARIO
    out.mkdir(parents=True, exist_ok=True)
    core = build_core()
    seed_all()
    print(f"[{SCENARIO}] target {TARGET_CELLS} cells, viscosity {VISCOSITY}, dt {DT}, "
          f"time_scale={TIME_SCALE:.3f}", flush=True)
    datasets, proc, eptm, outcome = run(core, build_spec(ensure_dataset()))
    n_frames = save_history(datasets, out / "history.hf5")
    summary = summarise(datasets)
    summary.to_csv(out / "summary.csv", index=False)

    report = eptm.find_intersections(strict=True)
    last = summary.iloc[-1]
    print(f"[{SCENARIO}] stopped ({outcome}) at t={last['time']:g}; {n_frames} frames -> "
          f"{out / 'history.hf5'}; divisions emitted {proc.n_divisions} "
          f"(skipped {proc.n_skipped}); cells {int(summary.iloc[0]['n_cells'])} -> "
          f"{int(last['n_cells'])}; final mesh: {len(report.folded)} folded, "
          f"{len(report.contained)} invading vertices, {len(report.overlaps)} "
          f"overlapping pairs", flush=True)
    print("Run sbml_cell_cycle_analysis.ipynb for the model description, figures and GIFs.")


if __name__ == "__main__":
    main()
