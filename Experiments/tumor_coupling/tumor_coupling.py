"""Tumor-coupling experiment — a COPASI population ODE drives tissue mechanics (SIMULATION).

A non-spatial breast-cancer **population ODE** (BioModels ``BIOMD0000000903``,
integrated in COPASI) is coupled to a flat 2-D tyssue epithelial sheet
(``test_square.hf5``, ``SheetGeometry``) through the ``TumorCoupling`` process.
Each step the process reads the SBML model's per-reaction birth/death fluxes and
fires ``floor(flux * scale * dt)`` discrete vertex-model events on the mesh:

  * **births**  -> real ``cell_division`` (the cell splits, the mesh gains a face),
  * **deaths**  -> real ``apoptosis_extrusion`` (the cell shrinks and is removed),
  * **tumor induction** -> ``differentiation`` of a cancer stem cell into a tumor cell.

A compact **cancer-stem-cell** focus is seeded at the sheet centre, matching the
SBML model's nonzero initial stem-cell population. The stem cells self-renew, commit
their first cell to tumor, and the tumor then grows outward into one contiguous clone
as the coupled fluxes drive divisions, while healthy cells are progressively
displaced. This mirrors ``vivarium_tyssue/composites/tumor.composite.yaml`` and the
``get_test_tumor_*`` helpers in ``tests/tests.py``.

This script **only runs the simulation and archives its data**: the tyssue
``History`` (thinned to ``TUMOR_ARCHIVE_FRAMES`` subsampled timepoints) is written
to a compressed HDF5 file (``outputs/history.hf5``). All visualisation (the
colour-coded 2-D GIF and stills) and analysis (tumor-vs-healthy population, tumor
area, and the face-area overlap diagnostic) live in the companion notebook
``tumor_coupling_analysis.ipynb``, which reopens the archive with ``tyssue``'s
``HistoryHdf5.from_archive``. Re-analyse without re-simulating.

Run from the repo's ``vivarium-tyssue`` conda env:

    conda activate vivarium-tyssue
    cd Experiments/tumor_coupling
    python tumor_coupling.py

The coupling advances by ``TUMOR_DT`` each step, so the tumor grows over
``TUMOR_TF / TUMOR_DT`` COPASI-driven updates, reaching a clear tumor takeover with
a persistent central stem-cell core. See the timescale-coupling note in README.md
and ``calibrate_timescale.py``.
"""
from __future__ import annotations

import contextlib
import copy
import io
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
FLAT_DATASET = "test_square.hf5"            # flat epithelial sheet
SBML_MODEL = "BIOMD0000000903.xml"          # COPASI breast-cancer population ODE

COORDS_2D = ["x", "y"]
SEED = 20260720

# Run length. TUMOR_TF is elapsed global (tyssue) time; the coupling steps at
# TUMOR_DT. The timescale coupling that stops divisions overrunning the sheet is
# GROWTH_RATE (each cell inflates gradually, see build_tumor_spec) plus the event
# SCALES (fewer, better-spaced divisions); the tumor grows slowly, so TUMOR_TF is
# long enough to reach a clear takeover. COPASI_TIME (alpha) = the tumor-model clock
# relative to tyssue time; kept at 1.0 (alpha<1 delays tumor induction and starves
# the seed). See calibrate_timescale.py for tau_mech.
TUMOR_TF, TUMOR_DT = 300.0, 0.01
COPASI_TIME = 1.0

# Exact COPASI reaction (flux) keys in BIOMD0000000903, per target cell type.
BIRTH_FLUXES = {
    "tumor": "Induction of tumor cell",
    "healthy": "Increase in the healthy cell in the system",
    "stem": "Formation of Stem cell",
}
DEATH_FLUXES = {
    "tumor": "Removal of tumor cell y immune cell and other immune response",
    "healthy": "Decrase of healthy cell due to cancer",
    "stem": "Removal of stem cell from the system",
}
# Convert SBML fluxes (O(1e3-1e7)) into mesh events: an event fires when
# flux * scale * interval accumulates past 1.0. Births and deaths are scaled
# together so the birth:death balance (and hence net tumor growth) is set by
# their ratio, while the overall magnitude sets how often divisions fire.
# stem_births > stem_deaths so the seeded cancer-stem-cell pool self-renews into a
# persistent central core (the raw SBML stem death flux is far larger, which would
# wipe out the handful of seeded stem cells).
SCALES = {
    "tumor_births": 1.0e-6, "tumor_deaths": 4.0e-7,
    "healthy_births": 3.0e-8, "healthy_deaths": 6.0e-8,
    "stem_births": 3.0e-6, "stem_deaths": 1.0e-6,
}

# Archiving. The run is TUMOR_TF/TUMOR_DT ~= 16000 solver steps; the History is
# thinned to a still-dense subsample (the GIF needs ~120 frames and the population
# / area / overlap curves stay smooth at this density). Set to ``None`` to keep
# every frame.
TUMOR_ARCHIVE_FRAMES: int | None = 2000
ARCHIVE_COMPLIB = "blosc:zstd"
ARCHIVE_COMPLEVEL = 5


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def ensure_dataset(name: str) -> Path:
    """Copy a tracked workspace/datasets file into the git-ignored local data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_DIR / name
    if not dst.exists():
        src = REPO / "workspace" / "datasets" / name
        if not src.exists():
            raise FileNotFoundError(f"source dataset not found: {src}")
        shutil.copy(src, dst)
        print(f"copied {src} -> {dst}")
    return dst


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
def tumor_config(dataset_path: Path) -> dict:
    """The flat-sheet EulerSolver config. Every cell starts cell_type='healthy';
    the shape index prefered_perimeter/sqrt(prefered_area)=3.6 keeps the sheet in
    the solid/jammed regime so cells stay compact as the tumor focus expands."""
    return {
        "name": "Tumor Epithelium 2D",
        "eptm": str(dataset_path),
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {
                "area_elasticity": 1.0,
                "prefered_area": 1.0,
                "perimeter_elasticity": 0.1,
                "prefered_perimeter": 3.6,
                "cell_type": "healthy",
                "is_alive": 1.0,
            },
            "edge_df": {"line_tension": 0.0, "is_active": 1.0},
            "vert_df": {"viscosity": 1.0, "is_alive": 1.0},
        },
        "geom": "SheetGeometry",
        "effectors": ["LineTension", "FaceAreaElasticity", "PerimeterElasticity"],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory",
        # A generous threshold_length makes T1 neighbour-swap reconnections fire
        # readily, letting the crowded tumor core rearrange instead of locking
        # into a tangle as the clone grows.
        "settings": {"threshold_length": 0.15},
        "auto_reconnect": True,
        "bounds": None,
        "output_columns": {},
        "history_columns": {},
        "maps": {},
        "backend": "rust",       # rust hot kernels (safe, bit-identical)
        "substeps": 1,
        "max_displacement": 0.0,
        "record_history": True,  # drives the gif / population-over-time counts
    }


def build_tumor_spec(mesh_path: Path, model_path: Path) -> dict:
    """EulerSolver (flat sheet) + a TumorCoupling process wired over Behaviors."""
    spec = {
        "Tyssue": {
            "_type": "process",
            "address": "local:EulerSolver",
            "config": tumor_config(mesh_path),
            "inputs": {"behaviors": ["Behaviors"], "global_time": ["global_time"]},
            "outputs": {
                "datasets": ["Tissue State"],
                "network_changed": ["Network Changed"],
                "behaviors_update": ["Behaviors"],
            },
            "interval": TUMOR_DT,
        },
        "Network Changed": False,
        "Behaviors": {},
    }
    spec["TumorCoupling"] = {
        "_type": "process",
        "address": "local:TumorCoupling",
        "config": {
            # COPASI is owned internally: the process steps the SBML model each
            # update and reads its per-reaction fluxes directly.
            "model_source": str(model_path),
            "copasi_time": COPASI_TIME,   # alpha: tumor-model time per unit tyssue time
            "copasi_intervals": 10,
            "birth_fluxes": dict(BIRTH_FLUXES),
            "death_fluxes": dict(DEATH_FLUXES),
            "scales": dict(SCALES),
            "geom": "SheetGeometry",
            # dt is inert: growth is integrated at the real solver step, so
            # growth_rate is a true per-tyssue-time rate. growth_rate=0.5 -> a cell
            # inflates over tau_grow = ln2/0.5 ~= 1.4 units (~4x tau_mech ~= 0.36):
            # slow enough that the sheet fully relaxes around each division (the
            # smallest face area never drops below its t=0 value -> no overlap), while
            # the compact seed keeps the clone growing.
            "dt": 1.0,
            "growth_rate": 0.5,
            "shrink_rate": 0.5,
            # A cell must grow past area 2.0 before it splits (daughters start near
            # the 1.0 prefered area) and shrink below 2.0 before it extrudes.
            "division_crit": 2.0,
            "apoptosis_crit": 2.0,
            # Seed a compact central cancer-stem-cell focus (6 cells, one contiguous
            # patch), like the SBML model's nonzero initial stem population. The stem
            # cells commit their first cell to tumor (C->T) and self-renew, so the
            # clone always has an eligible cell to grow from and a persistent CSC core.
            "seed": {"tumor": 0, "stem": 6},
            # Real vertex-model topology (cell_division / remove_face); requires the
            # pandas-3-compatible forked tyssue.
            "topology_ops": True,
        },
        "inputs": {"datasets": ["Tissue State"], "global_time": ["global_time"]},
        "outputs": {"behaviors": ["Behaviors"]},
        "interval": TUMOR_DT,
    }
    return spec


def run(core, spec: dict, tf: float):
    """Run a spec to elapsed global time ``tf``; return the tyssue History.

    tyssue's division prints a line per new cell; those are silenced here so the
    experiment's own progress log stays readable."""
    from process_bigraph import Composite

    spec = copy.deepcopy(spec)
    sim = Composite({"state": spec}, core=core)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run(tf)
    history = sim.state["Tyssue"]["instance"].history
    history.update_datasets()
    return history


# ---------------------------------------------------------------------------
# History archiving (compressed; optionally thinned)
# ---------------------------------------------------------------------------
def _sanitize_for_hdf(df: pd.DataFrame) -> pd.DataFrame:
    """Make an element dataframe serialisable by PyTables' table format: coerce each
    object column to numeric where every non-null value converts (ints/floats stored
    as object), otherwise to str (e.g. the ``cell_type`` label). Non-object columns
    are untouched, so already-clean archives are unchanged."""
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


def save_history(history, path: Path, keep_frames: int | None):
    """Archive ``history.datasets`` to a compressed HDF5 file (same per-element keys
    as tyssue's ``History.to_archive`` so the notebook can reopen it with
    ``HistoryHdf5.from_archive``). ``keep_frames`` optionally thins to that many
    subsampled timepoints; ``None`` keeps the full history."""
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
def _composition(history, t) -> dict:
    """{cell_type: count} of living cells at recorded time ``t`` (for the log)."""
    face = history.datasets["face"]
    df = face[face["time"] == t]
    if "is_alive" in df.columns:
        df = df[df["is_alive"] > 0]
    return df["cell_type"].value_counts().to_dict()


def main():
    np.random.seed(SEED)
    sys.path.insert(0, str(REPO))
    from vivarium_tyssue.core import build_core

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = ensure_dataset(FLAT_DATASET)
    model = ensure_dataset(SBML_MODEL)
    core = build_core()

    print(f"[tumor] flat sheet + COPASI BIOMD0000000903, tf={TUMOR_TF} dt={TUMOR_DT} "
          f"(~{int(TUMOR_TF / TUMOR_DT)} coupled updates) ...", flush=True)
    history = run(core, build_tumor_spec(mesh, model), TUMOR_TF)

    out = OUT_DIR / "history.hf5"
    save_history(history, out, TUMOR_ARCHIVE_FRAMES)
    times = list(history.time_stamps)
    print(f"[tumor] archived {len(times)} frames -> {out}", flush=True)
    if times:
        print(f"[tumor] composition start {_composition(history, times[0])} "
              f"-> end {_composition(history, times[-1])}", flush=True)
    print("\ndone — analyse with tumor_coupling_analysis.ipynb")


if __name__ == "__main__":
    main()
