"""Cell-jamming and parameter-gradient migration experiments (SIMULATION).

Two single-cell migration scenarios on a flat epithelial square sheet, reproduced
exactly from ``Notebooks/simulation_walkthrough.ipynb`` (specs 05 and 06) — same
config, same ``tf``/``dt``, no parameter sweep. One cell (face index 96) is given
an ``ActiveMigration`` drive and migrates through the tissue while stochastic
line-tension fluctuations act on every edge.

  * **Jamming** — the ``CellJamming`` process fires at ``t = 300`` (run to
    ``t = 400``, ``dt = 0.1``), ramping every cell's preferred perimeter down so
    the tissue solidifies and arrests the migrating cell.
  * **Gradient** — the ``ParameterGradient`` step imposes a linear
    ``prefered_perimeter`` gradient along x (run to ``t = 400``, ``dt = 0.05``), so
    the migrating cell moves through a stiffness gradient.

This script **only runs the simulations and archives each scenario's
full-resolution ``History`` to a compressed HDF5 file**
(``outputs/<jamming|gradient>/history.hf5``). All visualisation (colour-coded GIF,
still frames) and analysis (migrating-cell trace, a combined over-time figure per
scenario — displacement + circularity for jamming, displacement + preferred
perimeter for the gradient — velocity vs x, circularity binned along x) live in the
companion notebook
``jamming_gradient_analysis.ipynb``, which reopens the archives with ``tyssue``'s
``HistoryHdf5.from_archive``.

Everything lands under ``outputs/`` (input mesh under ``data/``), both git-ignored.
Run from the repo's ``vivarium-tyssue`` conda env:

    conda activate vivarium-tyssue
    cd Experiments/jamming_gradient
    python jamming_gradient.py
"""
from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / configuration (values fixed to the walkthrough notebook)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
DATASET_NAME = "test_square.hf5"

N_FACES = 206
MIGRATING_FACE = 96          # the cell given an ActiveMigration drive
TAU, SIGMA = 0.2, 0.1        # stochastic line-tension (shared by both scenarios)

# Jamming scenario (notebook spec 05)
JAM_TF, JAM_DT = 400.0, 0.1
JAM_TRIGGER = 300.0          # CellJamming.trigger_time
JAM_RATE, JAM_LIMITS = -0.05, [3.0, 4.2]
JAM_PREF_PERIM = 3.8

# Gradient scenario (notebook spec 06). Notebook runs to t=300; we extend by 100.
GRAD_TF, GRAD_DT = 400.0, 0.05
GRAD_ARGS = {"m": -0.1, "c": 4.6}   # prefered_perimeter = m*x + c

COORDS = ["x", "y"]
# One seed per scenario, applied immediately before that run, so either scenario
# reproduces its own archive when re-run on its own. (With a single seed at the top
# of main(), the gradient run depended on the jamming run having consumed the stream
# first, so re-running one scenario silently changed the other.)
SEEDS = {"jamming": 20260811, "gradient": 20260714}

# Colour range the analysis notebook uses for the prefered-perimeter colour map;
# defined here so both scenarios share one source of truth.
PP_RANGE = (3.0, 4.2)

# Archive the FULL-resolution history (every solver step) so the notebook can
# reproduce the exact migrating-cell trace / circularity. Compression keeps the size
# in check (see ``save_history``). Set to a positive int to thin every archive to
# that many subsampled frames (trades analysis fidelity for disk); ``None`` = full.
ARCHIVE_FRAMES: int | None = None
ARCHIVE_COMPLIB = "blosc:zstd"
ARCHIVE_COMPLEVEL = 5


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def ensure_dataset() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_DIR / DATASET_NAME
    if not dst.exists():
        src = REPO / "workspace" / "datasets" / DATASET_NAME
        if not src.exists():
            raise FileNotFoundError(f"source mesh not found: {src}")
        shutil.copy(src, dst)
        print(f"copied {src} -> {dst}")
    return dst


# ---------------------------------------------------------------------------
# Specs (adapted from Notebooks/simulation_walkthrough.ipynb specs 05 & 06)
# ---------------------------------------------------------------------------
def _flat_config(dataset_path: Path) -> dict:
    """get_test_config_flat with the migrating cell at face ``MIGRATING_FACE``."""
    return {
        "name": "Test Square",
        "eptm": str(dataset_path),
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {
                "area_elasticity": 1,
                "prefered_area": 1,
                "perimeter_elasticity": 0.1,
                "prefered_perimeter": 3.6,
                "migration_strength": [0.1 if i == MIGRATING_FACE else 0.0 for i in range(N_FACES)],
                "is_alive": 1,
                "mx": 1,
                "mz": 0,
                "my": 0,  # migrate along x only (notebook sets my = 0)
            },
            "edge_df": {"line_tension": 0, "is_active": 1},
            "vert_df": {"viscosity": 1, "is_alive": 1},
        },
        "geom": "SheetGeometry",
        # ActiveMigration is not rust-supported, so these demos run on python.
        "effectors": ["LineTension", "FaceAreaElasticity", "PerimeterElasticity", "ActiveMigration"],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory_bound",
        "settings": {"threshold_length": 0.03},
        "auto_reconnect": True,
        "bounds": None,
        "output_columns": {},
        "history_columns": {},
        "maps": {},
        "backend": "python",
        "substeps": 1,
        "max_displacement": 0.0,
        "record_history": True,
    }


def _base_spec(config: dict, dt: float) -> dict:
    spec = {
        "Tyssue": {
            "_type": "process",
            "address": "local:EulerSolver",
            "config": config,
            "inputs": {"behaviors": ["Behaviors"], "global_time": ["global_time"]},
            "outputs": {
                "datasets": ["Tissue State"],
                "network_changed": ["Network Changed"],
                "behaviors_update": ["Behaviors"],
            },
            "interval": dt,
        },
        "Stochastic": {
            "_type": "process",
            "address": "local:StochasticLineTension",
            "config": {"tau": TAU, "sigma": SIGMA},
            "inputs": {"datasets": ["Tissue State"]},
            "outputs": {"behaviors": ["Behaviors"]},
            "interval": dt,
        },
        "Network Changed": False,
        "Behaviors": {},
    }
    return spec


def build_jamming_spec(dataset_path: Path) -> dict:
    cfg = _flat_config(dataset_path)
    cfg["parameters"]["face_df"]["prefered_perimeter"] = JAM_PREF_PERIM
    spec = _base_spec(cfg, JAM_DT)
    spec["Jamming"] = {
        "_type": "process",
        "address": "local:CellJamming",
        "config": {"trigger_time": JAM_TRIGGER, "rate": JAM_RATE, "limits": JAM_LIMITS},
        "inputs": {"global_time": ["global_time"], "datasets": ["Tissue State"]},
        "outputs": {"behaviors": ["Behaviors"]},
        "interval": JAM_DT,
    }
    return spec


def build_gradient_spec(dataset_path: Path) -> dict:
    cfg = _flat_config(dataset_path)
    spec = _base_spec(cfg, GRAD_DT)
    spec["Gradient"] = {
        "_type": "step",
        "address": "local:ParameterGradient",
        "config": {
            "gradient_type": "linear",
            "axis": "x",
            "args": GRAD_ARGS,
            "model_parameters": {"prefered_perimeter": "face"},
        },
        "inputs": {"datasets": ["Tissue State"]},
        "outputs": {"behaviors": ["Behaviors"]},
    }
    return spec


def run(core, spec: dict, tf: float):
    from process_bigraph import Composite
    sim = Composite({"state": copy.deepcopy(spec)}, core=core)
    sim.run(tf)
    history = sim.state["Tyssue"]["instance"].history
    history.update_datasets()
    return history


# ---------------------------------------------------------------------------
# History archiving (compressed; optionally thinned)
# ---------------------------------------------------------------------------
def _sanitize_for_hdf(df: pd.DataFrame) -> pd.DataFrame:
    """Make an element dataframe serialisable by PyTables' table format: coerce each
    object-dtype column to numeric where every non-null value converts, otherwise to
    str. Non-object columns are untouched (this flat sheet has none, but the helper
    keeps every experiment's archive path robust)."""
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


def save_history(history, path: Path, keep_frames: int | None = ARCHIVE_FRAMES):
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
# Driver
# ---------------------------------------------------------------------------
def main():
    sys.path.insert(0, str(REPO))
    from vivarium_tyssue.core import build_core

    dataset = ensure_dataset()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core = build_core()

    # ---- Jamming -------------------------------------------------------
    jam_dir = OUT_DIR / "jamming"
    jam_dir.mkdir(parents=True, exist_ok=True)
    print(f"[jamming] running tf={JAM_TF} dt={JAM_DT} (python backend) ...", flush=True)
    np.random.seed(SEEDS["jamming"])
    hist_j = run(core, build_jamming_spec(dataset), JAM_TF)
    save_history(hist_j, jam_dir / "history.hf5")
    print(f"[jamming] archived {len(list(hist_j.time_stamps))} frames -> {jam_dir / 'history.hf5'}", flush=True)

    # ---- Gradient ------------------------------------------------------
    grad_dir = OUT_DIR / "gradient"
    grad_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gradient] running tf={GRAD_TF} dt={GRAD_DT} (python backend) ...", flush=True)
    np.random.seed(SEEDS["gradient"])
    hist_g = run(core, build_gradient_spec(dataset), GRAD_TF)
    save_history(hist_g, grad_dir / "history.hf5")
    print(f"[gradient] archived {len(list(hist_g.time_stamps))} frames -> {grad_dir / 'history.hf5'}", flush=True)

    print(f"\ndone — archives under {OUT_DIR}")
    print("Run jamming_gradient_analysis.ipynb to visualise and analyse the runs.")


if __name__ == "__main__":
    main()
