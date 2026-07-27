"""Discrete-event experiments — random divisions, random deaths, Gillespie (SIMULATION).

Three scenarios exercising the discrete-event processes:

  * **divisions** — a ``CellDivisions`` process fires cell divisions as a Poisson
    process (rate-based random times) on a plain **flat square sheet**
    (``test_square.hf5``, ``SheetGeometry``). Every cell starts ``"normal"``; a cell
    being actively grown toward division is flagged ``"dividing"``.
  * **deaths** — a ``CellDeaths`` process fires apoptotic extrusions as a Poisson
    process on the same flat square; a dying cell is flagged ``"extruding"``.
  * **gillespie** — the full Gillespie biochemistry (``Gillespie`` process) on the
    3-D crypt cylinder (``crypt_cylinder.hf5``) exactly as in ``tests/tests.py`` /
    ``Notebooks/simulation_walkthrough.ipynb`` (``tf=72``, ``dt=0.005``).

This script **only runs the simulations and archives their data**: each scenario's
full-resolution (gillespie: capped, see ``GILL_ARCHIVE_FRAMES``) ``History`` to a
compressed HDF5 file (``outputs/<scenario>/history.hf5``), plus — for gillespie —
the emitted discrete events to ``outputs/gillespie/events.csv`` (events come from
the process emitter, not the History, so they are saved separately).

All visualisation (2-D / 3-D colour-coded GIFs and stills) and analysis (cell-type
distributions over time and along z, event-type distribution along z) now live in
the companion notebook ``discrete_events_analysis.ipynb``, which reopens the
archives with ``tyssue``'s ``HistoryHdf5.from_archive``. Re-analyse without
re-simulating, and re-simulate without disturbing previous analysis.

Everything lands under ``outputs/`` (input meshes under ``data/``), both git-ignored.
Run from the repo's ``vivarium-tyssue`` conda env:

    conda activate vivarium-tyssue
    cd Experiments/discrete_events
    python discrete_events.py            # runs all three
    python discrete_events.py divisions  # or a single scenario: divisions|deaths|gillespie
"""
from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
FLAT_DATASET = "test_square.hf5"        # divisions / deaths (flat sheet)
CRYPT_DATASET = "crypt_cylinder.hf5"    # gillespie (3-D crypt)

COORDS_3D = ["x", "y", "z"]
COORDS_2D = ["x", "y"]
SEED = 20260715

# Discrete-event scenarios on the flat square (Poisson processes).
DIV_TF, DIV_DT = 25.0, 0.05
DIV_RATE, DIV_CRIT, DIV_GROWTH = 0.4, 2.0, 0.3           # CellDivisions
DEATH_TF, DEATH_DT = 25.0, 0.05
DEATH_RATE, DEATH_CRIT, DEATH_SHRINK = 0.4, 0.3, 0.3     # CellDeaths

# Gillespie scenario (identical to tests.py / walkthrough).
GILL_TF, GILL_DT = 72.0, 0.005

# Archiving. The flat-sheet scenarios (~500 frames) are archived in full. The
# gillespie crypt runs ~14400 solver steps, so its History is capped to a still-dense
# subsample (the GIF only needs ~120 frames and the cell-type distributions stay
# smooth). The discrete *events* are always saved in full (from the emitter). Set a
# scenario's cap to ``None`` to keep every frame.
FLAT_ARCHIVE_FRAMES: int | None = None
GILL_ARCHIVE_FRAMES: int | None = 1500
ARCHIVE_COMPLIB = "blosc:zstd"
ARCHIVE_COMPLEVEL = 5


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def ensure_dataset(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_DIR / name
    if not dst.exists():
        src = REPO / "workspace" / "datasets" / name
        if not src.exists():
            raise FileNotFoundError(f"source mesh not found: {src}")
        shutil.copy(src, dst)
        print(f"copied {src} -> {dst}")
    return dst


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
def flat_config(dataset_path: Path) -> dict:
    """Plain flat square sheet (SheetGeometry / model_factory_bound). Every cell
    starts ``cell_type="normal"`` so the division / extrusion behaviours can flag the
    active cell for colour-coding."""
    return {
        "name": "Flat Square",
        "eptm": str(dataset_path),
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {
                "area_elasticity": 1.0,
                "prefered_area": 1.0,
                "perimeter_elasticity": 0.1,
                "prefered_perimeter": 3.6,
                "is_alive": 1.0,
                "cell_type": "normal",
            },
            "edge_df": {"line_tension": 0.0, "is_active": 1.0},
            "vert_df": {"viscosity": 1.0, "is_alive": 1.0},
        },
        "geom": "SheetGeometry",
        "effectors": ["LineTension", "FaceAreaElasticity", "PerimeterElasticity"],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory_bound",
        "settings": {"threshold_length": 0.03},
        "auto_reconnect": True,
        "bounds": None,
        "output_columns": {},
        "history_columns": {},
        "maps": {},
        # topology-mutating behaviours (division / extrusion) run safest on python.
        "backend": "python",
        "substeps": 1,
        "max_displacement": 0.0,
        "record_history": True,
    }


def crypt_config(dataset_path: Path) -> dict:
    """The crypt-cylinder EulerSolver config used by the Gillespie model
    (VesselGeometry / model_factory_vessel)."""
    return {
        "name": "Crypt Cylinder",
        "eptm": str(dataset_path),
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {
                "area_elasticity": 1.0,
                "prefered_area": 1.0,
                "perimeter_elasticity": 0.5,
                "prefered_perimeter": 3.5,
            },
            "edge_df": {"line_tension": 0.0, "is_active": 1.0},
            "vert_df": {
                "viscosity": 0.05,
                "vessel_elasticity": 1.0,
                "prefered_radius": 2.5,
                "is_alive": 1.0,
                "surface_elasticity": 0.1,
            },
        },
        "geom": "VesselGeometry",
        "effectors": ["FaceAreaElasticity", "PerimeterElasticity", "LineTension", "VesselSurfaceElasticity"],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory_vessel",
        "settings": {"threshold_length": 0.03, "radius": 2.5, "axis": "z"},
        "auto_reconnect": True,
        "bounds": None,
        "output_columns": {},
        "history_columns": {},
        "maps": {},
        # VesselGeometry + model_factory_vessel run on the rust compositional
        # gradient path (kernels.rust_model_gradient) with rust geometry — ~2x
        # faster than python here. Topology-change staleness (division / extrusion
        # / rosette detach) is handled, so long crypt runs stay stable.
        "backend": "rust",
        "substeps": 1,
        "max_displacement": 0.0,
        "record_history": True,
    }


def _solver_node(config: dict, dt: float) -> dict:
    return {
        "Tyssue": {
            "_type": "process",
            "address": "local:EulerSolver",
            "config": config,
            "inputs": {"behaviors": ["Behaviors"], "global_time": ["global_time"]},
            "outputs": {
                "datasets": ["Datasets"],
                "network_changed": ["Network Changed"],
                "behaviors_update": ["Behaviors"],
            },
            "interval": dt,
        },
        "Network Changed": False,
        "Behaviors": {},
    }


def build_divisions_spec(dataset_path: Path) -> dict:
    spec = _solver_node(flat_config(dataset_path), DIV_DT)
    spec["Divisions"] = {
        "_type": "process",
        "address": "local:CellDivisions",
        "config": {
            "rate": DIV_RATE,
            "geom": "SheetGeometry",
            "crit_area": DIV_CRIT,
            "growth_rate": DIV_GROWTH,
        },
        "inputs": {"global_time": ["global_time"], "datasets": ["Datasets"]},
        "outputs": {"behaviors": ["Behaviors"]},
        "interval": DIV_DT,
    }
    return spec


def build_deaths_spec(dataset_path: Path) -> dict:
    spec = _solver_node(flat_config(dataset_path), DEATH_DT)
    spec["Deaths"] = {
        "_type": "process",
        "address": "local:CellDeaths",
        "config": {
            "rate": DEATH_RATE,
            "geom": "SheetGeometry",
            "crit_area": DEATH_CRIT,
            "shrink_rate": DEATH_SHRINK,
        },
        "inputs": {"global_time": ["global_time"], "datasets": ["Datasets"]},
        "outputs": {"behaviors": ["Behaviors"]},
        "interval": DEATH_DT,
    }
    return spec


def build_gillespie_spec(dataset_path: Path) -> dict:
    from vivarium_tyssue.models.crypt_gillespie.crypt_params import cell_types
    from vivarium_tyssue.models.crypt_gillespie.jump_rates import (
        rates_max, K, k, regulations, regulation_loc,
    )

    spec = _solver_node(crypt_config(dataset_path), GILL_DT)
    spec["Gillespie"] = {
        "_type": "process",
        "address": "local:Gillespie",
        "config": {
            "cell_types": cell_types,
            "rates_max": rates_max,
            "michaelis_constants": K,
            "transition_lengths": k,
            "geom": "VesselGeometry",
            "global_interval": GILL_DT,
            "growth_rate": 0.02,
            "shrink_rate": 0.02,
            "division_crit": 1.2,
            "apoptosis_crit": 0.1,
            "regulations": regulations,
            "regulation_loc": regulation_loc,
        },
        "inputs": {
            "datasets": ["Datasets"],
            "behaviors": ["Behaviors"],
            "global_time": ["global_time"],
        },
        "outputs": {
            "behaviors": ["Behaviors"],
            "gillespie_trigger": ["Gillespie Trigger"],
        },
        "interval": GILL_DT,
    }
    return spec


def run(core, spec: dict, tf: float, capture_behaviors: bool = False):
    """Run a spec; return (history, events). ``events`` is a list of emitted
    behaviour dicts (only when ``capture_behaviors``)."""
    from process_bigraph import Composite
    from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

    spec = copy.deepcopy(spec)
    if capture_behaviors:
        spec["emitter"] = emitter_from_wires({
            "global_time": ["global_time"],
            "behaviors": ["Behaviors"],
        })
    sim = Composite({"state": spec}, core=core)
    sim.run(tf)
    history = sim.state["Tyssue"]["instance"].history
    history.update_datasets()

    events = []
    if capture_behaviors:
        results = gather_emitter_results(sim)[("emitter",)]
        events = _collect_events(results)
    return history, events


def _collect_events(results) -> list:
    """Flatten emitter frames into distinct (time, func, cell_uid) event records."""
    seen = set()
    events = []
    for frame in results:
        t = float(frame.get("global_time", 0.0))
        behaviors = frame.get("behaviors", [])
        if isinstance(behaviors, dict):
            behaviors = list(behaviors.values())
        for b in behaviors or []:
            if not isinstance(b, dict) or "func" not in b:
                continue
            key = (round(t, 5), b.get("func"), b.get("cell_uid"))
            if key in seen:
                continue
            seen.add(key)
            events.append({"time": t, "func": b["func"], "cell_uid": b.get("cell_uid")})
    return events


# ---------------------------------------------------------------------------
# History archiving (compressed; optionally thinned)
# ---------------------------------------------------------------------------
def _sanitize_for_hdf(df: pd.DataFrame) -> pd.DataFrame:
    """Make an element dataframe serialisable by PyTables' table format.

    The crypt's ``cell`` element carries object-dtype columns whose contents are
    plain integers (e.g. the ``cell`` index column); PyTables refuses to serialise
    an object column that is neither all-string nor a real numeric dtype. Coerce each
    object column to numeric where every non-null value converts, otherwise to str.
    Non-object columns are left untouched, so already-clean archives are unchanged.
    """
    df = df.copy()
    for c in df.columns:
        if df[c].dtype != object:
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        notnull = df[c].notna()
        if notnull.any() and (conv.notna() | ~notnull).all():
            df[c] = conv            # genuinely numeric (ints/floats stored as object)
        else:
            df[c] = df[c].astype(str)   # strings (e.g. cell_type)
    return df


def save_history(history, path: Path, keep_frames: int | None):
    """Archive ``history.datasets`` to a compressed HDF5 file (same per-element keys
    as tyssue's ``History.to_archive`` so the notebook can reopen it with
    ``HistoryHdf5.from_archive``). ``keep_frames`` optionally thins to that many
    subsampled timepoints; ``None`` keeps the full history. The live history is not
    mutated."""
    datasets = history.datasets
    times = np.array(list(history.time_stamps))
    if keep_frames is not None and times.size > keep_frames:
        idx = np.unique(np.round(np.linspace(0, times.size - 1, keep_frames)).astype(int))
        keep = set(times[idx].tolist())
        datasets = {k: df[df["time"].isin(keep)] for k, df in datasets.items()}

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with pd.HDFStore(path, "a", complevel=ARCHIVE_COMPLEVEL, complib=ARCHIVE_COMPLIB) as store:
        for key, df in datasets.items():
            df = _sanitize_for_hdf(df)
            kwargs = {"data_columns": ["time"]}
            if "segment" in df.columns:
                kwargs["min_itemsize"] = {"segment": 7}
            store.append(key=key, value=df, **kwargs)


# ---------------------------------------------------------------------------
# Scenario drivers
# ---------------------------------------------------------------------------
def run_divisions(core):
    dataset = ensure_dataset(FLAT_DATASET)
    out = OUT_DIR / "divisions"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[divisions] flat square, tf={DIV_TF} dt={DIV_DT} rate={DIV_RATE} ...", flush=True)
    history, _ = run(core, build_divisions_spec(dataset), DIV_TF)
    save_history(history, out / "history.hf5", FLAT_ARCHIVE_FRAMES)
    print(f"[divisions] archived {len(list(history.time_stamps))} frames -> {out / 'history.hf5'}", flush=True)


def run_deaths(core):
    dataset = ensure_dataset(FLAT_DATASET)
    out = OUT_DIR / "deaths"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[deaths] flat square, tf={DEATH_TF} dt={DEATH_DT} rate={DEATH_RATE} ...", flush=True)
    history, _ = run(core, build_deaths_spec(dataset), DEATH_TF)
    save_history(history, out / "history.hf5", FLAT_ARCHIVE_FRAMES)
    print(f"[deaths] archived {len(list(history.time_stamps))} frames -> {out / 'history.hf5'}", flush=True)


def run_gillespie(core):
    dataset = ensure_dataset(CRYPT_DATASET)
    out = OUT_DIR / "gillespie"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[gillespie] crypt, tf={GILL_TF} dt={GILL_DT} ...", flush=True)
    history, events = run(core, build_gillespie_spec(dataset), GILL_TF, capture_behaviors=True)
    print(f"[gillespie] captured {len(events)} events", flush=True)
    save_history(history, out / "history.hf5", GILL_ARCHIVE_FRAMES)
    pd.DataFrame(events).to_csv(out / "events.csv", index=False)
    print(f"[gillespie] archived {len(list(history.time_stamps))} frames -> {out / 'history.hf5'}", flush=True)
    print(f"[gillespie] wrote {out / 'events.csv'}", flush=True)


SCENARIOS = {"divisions": run_divisions, "deaths": run_deaths, "gillespie": run_gillespie}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in SCENARIOS and which != "all":
        raise SystemExit(f"unknown scenario '{which}'; choose from {list(SCENARIOS)} or 'all'")

    np.random.seed(SEED)
    sys.path.insert(0, str(REPO))
    from vivarium_tyssue.core import build_core

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core = build_core()

    todo = list(SCENARIOS) if which == "all" else [which]
    for name in todo:
        SCENARIOS[name](core)

    print(f"\ndone — archives under {OUT_DIR}")
    print("Run discrete_events_analysis.ipynb to visualise and analyse the scenarios.")


if __name__ == "__main__":
    main()
