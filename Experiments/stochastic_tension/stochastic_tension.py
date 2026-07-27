"""Stochastic line-tension experiments — a tau/sigma parameter sweep (SIMULATION).

A flat epithelial square sheet is driven by an Ornstein-Uhlenbeck fluctuation of
per-edge line tension (the ``StochasticLineTension`` process). This script sweeps
the OU relaxation timescale ``tau`` and noise amplitude ``sigma`` and, for each
cell of the grid, **runs the simulation and archives the full-resolution
``History`` to a compressed HDF5 file** — nothing else.

All visualisation and analysis (colour-coded videos, still frames, the
vertex-movement / T1-transition metrics and the sweep summary heatmaps) now live
in the companion notebook ``stochastic_tension_analysis.ipynb``, which reads these
archives back with ``tyssue``'s ``HistoryHdf5.from_archive`` so you can re-analyse
without re-simulating (and re-simulate without disturbing previous analysis).

Each run's archive lands at ``outputs/<tau..._sigma...>/history.hf5`` (git-ignored),
the input mesh under ``data/``.

Run (from the repo's ``vivarium-tyssue`` conda env; the .venv/uv are broken):

    conda activate vivarium-tyssue
    cd Experiments/stochastic_tension
    python stochastic_tension.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / sweep configuration
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
DATASET_NAME = "test_square.hf5"

# 4 x 4 sweep. tau = OU relaxation timescale, sigma = noise amplitude.
# Baseline in the repo is tau=0.2, sigma=0.1 (kept in the grid).
TAUS = [0.1, 0.2, 0.5, 1.0]
SIGMAS = [0.05, 0.1, 0.2, 0.4]

TF = 15.0          # total simulated time — kept low so archived data stays small
DT = 0.1           # emit / solver interval (~150 steps per run)
SEED = 20260714    # OU noise is otherwise unseeded; fix it for a reproducible sweep

COORDS = ["x", "y"]

# The archive keeps the FULL-resolution history (every solver step) so the analysis
# notebook can reproduce the exact per-step vertex-movement / T1 metrics. Compression
# (see ``save_history``) keeps each ~150-frame run to a few MB. Set a positive integer
# here to instead thin every archive to that many subsampled frames (trades analysis
# fidelity for disk); ``None`` keeps everything.
ARCHIVE_FRAMES: int | None = None

# HDF5 compression for the archives (pandas/PyTables blosc:zstd). ~3-4x smaller than
# tyssue's uncompressed ``History.to_archive`` with no loss.
ARCHIVE_COMPLIB = "blosc:zstd"
ARCHIVE_COMPLEVEL = 5


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def ensure_dataset() -> Path:
    """Copy the flat-sheet mesh into ``data/`` (git-ignored) if not already there,
    sourcing it from the repo's tracked ``workspace/datasets`` copy."""
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
# Simulation spec (adapted from tests/tests.py get_test_config_flat +
# get_test_stochastic_spec — self-contained, no emitter so nothing is written to
# out/parquet; the History is archived to HDF5 instead).
# ---------------------------------------------------------------------------
def build_spec(tau: float, sigma: float, dt: float, dataset_path: Path) -> dict:
    tyssue_config = {
        "name": "Stochastic Square",
        "eptm": str(dataset_path),
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {
                "area_elasticity": 1.0,
                "prefered_area": 1.0,
                "perimeter_elasticity": 0.1,
                "prefered_perimeter": 3.6,
                "migration_strength": 0.0,
                "is_alive": 1.0,
                "mx": 1.0,
                "mz": 0.0,
                "my": 1.0,
            },
            "edge_df": {
                "line_tension": 0.0,
                "is_active": 1.0,
            },
            "vert_df": {
                "viscosity": 1.0,
                "is_alive": 1.0,
            },
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
        "backend": "rust",
        "substeps": 1,
        "max_displacement": 0.0,
        "record_history": True,  # build an in-memory History (records all columns)
    }

    return {
        "Tyssue": {
            "_type": "process",
            "address": "local:EulerSolver",
            "config": tyssue_config,
            "inputs": {
                "behaviors": ["Behaviors"],
                "global_time": ["global_time"],
            },
            "outputs": {
                "datasets": ["Datasets"],
                "network_changed": ["Network Changed"],
                "behaviors_update": ["Behaviors"],
            },
            "interval": dt,
        },
        "Stochastic": {
            "_type": "process",
            "address": "local:StochasticLineTension",
            "config": {"tau": tau, "sigma": sigma},
            "inputs": {"datasets": ["Datasets"]},
            "outputs": {"behaviors": ["Behaviors"]},
            "interval": dt,
        },
        "Network Changed": False,
        "Behaviors": {},
    }


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
    """Archive ``history.datasets`` to a compressed HDF5 file.

    Mirrors tyssue's ``History.to_archive`` (same per-element keys, so the notebook
    can reopen it with ``HistoryHdf5.from_archive``) but writes through a
    blosc:zstd-compressed ``HDFStore``. If ``keep_frames`` is a positive int the
    archive is thinned to that many subsampled timepoints; ``None`` keeps the full
    history. The live in-memory history is never mutated.
    """
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


def run_one(core, tau: float, sigma: float, tf: float, dt: float):
    """Run one (tau, sigma) simulation; return its updated tyssue History."""
    from process_bigraph import Composite

    dataset_path = DATA_DIR / DATASET_NAME
    spec = build_spec(tau, sigma, dt, dataset_path)
    sim = Composite({"state": spec}, core=core)
    sim.run(tf)
    history = sim.state["Tyssue"]["instance"].history
    history.update_datasets()
    return history


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    sys.path.insert(0, str(REPO))
    from vivarium_tyssue.core import build_core

    ensure_dataset()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core = build_core()

    for tau in TAUS:
        for sigma in SIGMAS:
            tag = f"tau{tau:0.2f}_sigma{sigma:0.2f}"
            run_dir = OUT_DIR / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run] {tag} ...", flush=True)

            history = run_one(core, tau, sigma, TF, DT)
            save_history(history, run_dir / "history.hf5")
            n = len(list(history.time_stamps))
            print(f"       archived {n} frames -> {run_dir / 'history.hf5'}", flush=True)

    print(f"\ndone — {len(TAUS) * len(SIGMAS)} runs archived under {OUT_DIR}")
    print("Run stochastic_tension_analysis.ipynb to visualise and analyse the sweep.")


if __name__ == "__main__":
    main()
