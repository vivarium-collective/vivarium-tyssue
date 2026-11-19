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

Two scenarios on the flat square sheet (``test_square.hf5``, 206 cells) with a
free border (``model_factory``):

  * **growth**: growth factor GF = 1 in every cell, grown from 206 to
    ``TARGET_CELLS`` cells.
  * **senescence**: replicative aging. Each division lowers a lineage's mitogen
    sensitivity, GF = ``SENESCENCE_DECAY`` ** generation, read per cell from the
    ``generation`` face column that ``split_cells`` increments. The model's G1/S
    switch turns that decline into permanent arrest (G0), so the tissue grows for
    a few cycles and then stops at a fixed size.

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
    python sbml_cell_cycle.py                         # growth
    python sbml_cell_cycle.py senescence              # -> outputs/senescence/
    python sbml_cell_cycle.py senescence --seed 2     # -> outputs/senescence_seed2/
"""
from __future__ import annotations

import os

# Pin BLAS to one thread before numpy loads: multithreaded reductions make
# same-seed runs diverge through T1 flips.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
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
MODEL_PERIOD = 23.6                 # hours of model time at GF = 1
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
# Rosette detachment (tyssue's reconnect): a rank-4 vertex splits at rate p_4, a
# rank-5+ one at p_5p, per unit sim time. tyssue's defaults (0.1, 0.01) leave a
# 4-fold vertex waiting ~12 h of model time, against minutes to an hour for real
# junction remodelling, and 4-fold vertices pile up (~55 per 1000 cells). A 10 min
# mean wait clears them (~1 per 1000) without changing growth or fold onset
# (SWEEPS.md, section 5). p_5p keeps tyssue's 1:10 ratio; rank-5+ vertices hardly
# form either way.
ROSETTE_WAIT_HOURS = 1.0 / 6.0
P_4 = TIME_SCALE / ROSETTE_WAIT_HOURS        # TIME_SCALE = model hours per sim unit
P_5P = P_4 / 10.0
TARGET_CELLS = 1200    # stop here: clean on 9/10 seeds at this viscosity (SWEEPS.md)
FRAME_DT = 0.25        # snapshot (and fold check) interval
TF = 80.0              # safety cap; the target is reached near t = 60

# Senescence scenario: replicative aging. Every division multiplies a cell's
# mitogen sensitivity by SENESCENCE_DECAY, GF = SENESCENCE_DECAY ** generation.
# Full cycles need GF > ~0.072, so a generation-1 cell is already below threshold;
# the commitment lag (decision ~3 h before mitotic exit) still lets most lineages
# finish 2-3 divisions in all, i.e. ~5x the initial cells (mechanics-free check).
SENESCENCE_DECAY = 0.05
# Cyclin D/Cdk4-6 tracks GF within hours (0.185 at GF = 1, ~0.03 at GF = 0.05,
# -> 0 below), while E2F and the other cyclins of an arrested cell damp out over
# 60-80 h. So G0 is read from cyclin D alone, after the S/M/G2 rules: a cell
# finishing a committed cycle is still labelled S, G2 or M. G0 cells never divide.
G0_RULE = "G0: cyclin_D_Cdk4_6 < 0.04"
# Growth in S and G2, not G1 and G2. Cyclin D falls within hours of each division,
# so in a lineage's last (committed) cycle G1 already reads as G0. With G1 growth
# such a cell skipped it, divided undergrown and left half-size daughters; the
# sheet packed to mean area ~0.88 and folded at ~540 cells. S and G2 are read from
# cyclins E and B, independent of the mitogen arm, so every dividing cell has
# doubled its target. S + G2 is 56% of the GF = 1 cycle (13.2 of 23.6 h). A
# committed cell whose G1 reads as G0 grows there too and reaches the cap sooner.
# G0 cells grow too, up to SENESCENCE_MAX_AREA: senescent cells enlarge and
# flatten. Without it the arrested cells sit packed at area ~0.75-0.85 and the late
# divisions among them fold the sheet (4 of 5 seeds, at 445-1033 cells; every fold
# was a just-divided pair against small G0 neighbours).
SENESCENCE_GROWTH_PHASES = ["S", "G2", "G0"]
SENESCENCE_GROWTH_RATE = np.log(2.0) / (0.56 * CYCLE_TIME)
# Growth stops at twice the birth area. With the per-cell noise on vse and Vde, a
# few cells (~0.1-0.3% here; ~3% of the GF = 1 run) fall into the skeleton model's
# runaway state: cyclin E synthesis outruns its saturating degradation, E2F stays
# on, and the cell never leaves S. S is a growth phase here, so without a cap such
# a cell inflated to 13x its target and folded a plateaued sheet (seed 2).
SENESCENCE_MAX_AREA = 2.0 * BIRTH_AREA
# At GF = 0.05 the cycle's amplitude drops and cyclin A peaks right at the M
# rule's 2.0. A cell that has just split, and its daughter with the same state,
# then flips M -> G2 -> M and splits again 0.5-2.5 later: 3-5% of divisions, all
# spurious (the GF = 1 run's shortest cycle is 15.5). No real cycle is this short.
SENESCENCE_MIN_DIVISION_INTERVAL = 0.5 * CYCLE_TIME
SCENARIOS = {
    "growth": {"phases": PHASES, "growth_phases": GROWTH_PHASES,
               "growth_rate": GROWTH_RATE, "target": TARGET_CELLS, "tf": TF},
    # Run on well past the last division to show the plateau; the cap only guards
    # against a runaway (the free sheet folds near ~1500 cells).
    "senescence": {"phases": PHASES + [G0_RULE], "growth_phases": SENESCENCE_GROWTH_PHASES,
                   "growth_rate": SENESCENCE_GROWTH_RATE, "target": 1500, "tf": 120.0},
}


def phase_names(scenario: str = SCENARIO) -> list:
    """SBMLCellCycle.phase_names for a scenario: the default phase, then the rules'."""
    return ["G1"] + [r.split(":")[0].strip() for r in SCENARIOS[scenario]["phases"]]

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


def solver_config(dataset_path: Path, scenario: str = SCENARIO) -> dict:
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
    if scenario == "senescence":
        face_params["generation"] = 0.0   # incremented on both halves by split_cells
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
        "auto_reconnect": {"enabled": True, "p_4": P_4, "p_5p": P_5P},
        "bounds": None,
        "output_columns": {},
        "history_columns": {},
        "maps": {},
        "backend": "python",
        "substeps": 1,
        "max_displacement": 0.0,
        "record_history": False,   # run() snapshots every FRAME_DT instead
    }


def cell_cycle_config(scenario: str = SCENARIO) -> dict:
    config = {
        "model_source": str(MODEL_SBML),
        "time_scale": TIME_SCALE,
        "parameters": {"GF": GROWTH_FACTOR},
        "phases": SCENARIOS[scenario]["phases"],
        "growth_phases": SCENARIOS[scenario]["growth_phases"],
        "division_phase": DIVISION_PHASE,
        "growth_rate": SCENARIOS[scenario]["growth_rate"],
        "birth_area": BIRTH_AREA,
        "parameter_noise": PARAMETER_NOISE,
        "geom": "SheetGeometry",
        "initial_state": "limit_cycle",
    }
    if scenario == "senescence":
        # GF = SENESCENCE_DECAY ** generation  (exponential_gradient: a**x + c).
        config.update({
            "input_columns": {"GF": "generation"},
            "input_functions": {"GF": "exponential"},
            "input_args": {"GF": {"a": SENESCENCE_DECAY, "c": 0.0}},
            "max_prefered_area": SENESCENCE_MAX_AREA,
            "min_division_interval": SENESCENCE_MIN_DIVISION_INTERVAL,
        })
    return config


def build_spec(dataset_path: Path, scenario: str = SCENARIO) -> dict:
    return {
        "Tyssue": {
            "_type": "process",
            "address": "local:EulerSolver",
            "config": solver_config(dataset_path, scenario),
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
            "config": cell_cycle_config(scenario),
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


def summarise(datasets: dict, scenario: str = SCENARIO) -> pd.DataFrame:
    """Per recorded time: cells, divisions so far, cycling cells, cells per phase,
    area, strain, the tissue's area and (senescence) the generation reached."""
    faces = datasets["face"]
    names = phase_names(scenario)
    rows = []
    n0 = None
    for t, fd in faces.groupby("time"):
        if "is_alive" in fd.columns:
            fd = fd[fd["is_alive"] > 0]
        n = len(fd)
        n0 = n if n0 is None else n0
        cycling = (t - fd["cc_last_division"]) < CYCLING_WINDOW
        row = {
            "time": t, "n_cells": n, "n_cycling": int(cycling.sum()),
            "cycling_fraction": float(cycling.mean()),
            "n_growing": int(fd["cc_phase"].isin(
                [names.index(p) for p in SCENARIOS[scenario]["growth_phases"]]).sum()),
            "divisions": n - n0,
            "mean_area": float(fd["area"].mean()),
            "mean_strain": float((fd["area"] / fd["prefered_area"]).mean()),
            "tissue_area": float(fd["area"].sum()),
        }
        row.update({f"n_{p}": int((fd["cc_phase"] == i).sum()) for i, p in enumerate(names)})
        if "generation" in fd.columns:
            row["mean_generation"] = float(fd["generation"].mean())
            row["max_generation"] = int(fd["generation"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def output_dir(scenario: str = SCENARIO, seed: int = SEED) -> Path:
    """outputs/<scenario> for the default seed, outputs/<scenario>_seed<N> otherwise."""
    return OUT_DIR / (scenario if seed == SEED else f"{scenario}_seed{seed}")


def main():
    sys.path.insert(0, str(REPO))
    from vivarium_tyssue.core import build_core

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scenario", nargs="?", default=SCENARIO, choices=sorted(SCENARIOS))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    scenario, cfg = args.scenario, SCENARIOS[args.scenario]

    out = output_dir(scenario, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    core = build_core()
    seed_all(args.seed)
    print(f"[{scenario}] seed {args.seed}, target {cfg['target']} cells, tf {cfg['tf']}, "
          f"viscosity {VISCOSITY}, dt {DT}, time_scale={TIME_SCALE:.3f}", flush=True)
    datasets, proc, eptm, outcome = run(core, build_spec(ensure_dataset(), scenario),
                                        target=cfg["target"], tf=cfg["tf"])
    n_frames = save_history(datasets, out / "history.hf5")
    summary = summarise(datasets, scenario)
    summary.to_csv(out / "summary.csv", index=False)

    report = eptm.find_intersections(strict=True)
    last = summary.iloc[-1]
    print(f"[{scenario}] stopped ({outcome}) at t={last['time']:g}; {n_frames} frames -> "
          f"{out / 'history.hf5'}; divisions emitted {proc.n_divisions} "
          f"(skipped {proc.n_skipped}, refractory {proc.n_refractory}); cells {int(summary.iloc[0]['n_cells'])} -> "
          f"{int(last['n_cells'])}; final mesh: {len(report.folded)} folded, "
          f"{len(report.contained)} invading vertices, {len(report.overlaps)} "
          f"overlapping pairs", flush=True)
    print("Run sbml_cell_cycle_analysis.ipynb for the model description, figures and GIFs.")


if __name__ == "__main__":
    main()
