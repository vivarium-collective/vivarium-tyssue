"""Stochastic line-tension experiments — a tau/sigma parameter sweep.

A flat epithelial square sheet is driven by an Ornstein-Uhlenbeck fluctuation of
per-edge line tension (the ``StochasticLineTension`` process). This script sweeps
the OU relaxation timescale ``tau`` and noise amplitude ``sigma`` and, for each
cell of the grid:

  * runs the simulation (``EulerSolver`` + ``StochasticLineTension``),
  * renders a colour-coded video (edges coloured by their live ``line_tension``),
  * saves still frames at a few timepoints,
  * archives the simulation ``History`` to an HDF5 file, and
  * measures two outcomes -- the *degree of stochastic vertex movement* and the
    *number of T1 transitions* (cell rearrangements where cells swap neighbours).

Finally it builds sweep-level summary figures (heatmaps of movement and T1 count
over the tau/sigma grid, plus a movement-vs-sigma line plot).

Everything generated lands under ``outputs/`` (and the input mesh under
``data/``), both of which are git-ignored -- nothing here is tracked or pushed.

Run (from the repo's ``vivarium-tyssue`` conda env; the .venv/uv are broken):

    conda activate vivarium-tyssue
    cd Experiments/stochastic_tension
    python stochastic_tension.py
"""
from __future__ import annotations

import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — we only ever savefig
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
NUM_GIF_FRAMES = 60
STILL_FRACTIONS = [0.0, 0.33, 0.66, 1.0]
FIG_DPI = 300      # publication-quality raster resolution for stills / summary figures
SAVE_HISTORY = True  # archive each run's History to HDF5. Flip off if disk is
                     # tight — metrics/gif/stills are unaffected.
ARCHIVE_FRAMES = 30  # frames kept per archived History (subsampled from the full
                     # ~150). Analysis always uses the full-resolution in-memory
                     # history; only the on-disk archive is thinned to keep the
                     # whole sweep well under a gigabyte.
SEED = 20260714      # OU noise is otherwise unseeded; fix it for a reproducible sweep

COORDS = ["x", "y"]

# Colourblind-safe categorical order (Okabe-Ito) for the tau series in the line
# plot; viridis (perceptually-uniform sequential) for the magnitude heatmaps.
TAU_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
EDGE_CMAP = "coolwarm"   # diverging: line tension is signed around 0


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
# out/parquet; analysis reads the in-memory History directly).
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


def archive_thinned(history, path: Path, keep_frames: int):
    """Archive the History to HDF5, keeping only ``keep_frames`` subsampled frames.

    ``to_archive`` writes ``history.datasets`` (all frames stacked, with a ``time``
    column). We temporarily swap in time-filtered copies so the archive is small
    while the live in-memory history (used for analysis) is untouched.
    """
    times = np.array(list(history.time_stamps))
    if times.size > keep_frames:
        idx = np.round(np.linspace(0, times.size - 1, keep_frames)).astype(int)
        keep = set(times[np.unique(idx)].tolist())
    else:
        keep = set(times.tolist())

    orig = history.datasets
    try:
        history.datasets = {k: df[df["time"].isin(keep)] for k, df in orig.items()}
        if path.exists():
            path.unlink()
        history.to_archive(str(path))
    finally:
        history.datasets = orig


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
# Visualisation — colour-coded gif + stills
# ---------------------------------------------------------------------------
def _edge_rgba(edge_df, color_range):
    """(Ne, 4) RGBA colouring each edge by its ``line_tension``."""
    cmap = plt.get_cmap(EDGE_CMAP)
    cmin, cmax = color_range
    vals = edge_df["line_tension"].to_numpy().astype(float)
    normed = np.clip((vals - cmin) / (cmax - cmin), 0.0, 1.0)
    return cmap(normed)


def save_gif(history, out_path: Path, sigma: float):
    """Colour-coded animation: edges tinted by their live line tension."""
    from tyssue import config
    from tyssue.draw import create_gif
    from vivarium_tyssue.draw import line_tension_edge_kwds

    crange = (-3.0 * sigma, 3.0 * sigma)  # scaled to sigma so hue is comparable within a run
    draw_specs = config.draw.sheet_spec()
    draw_specs["face"]["visible"] = True
    draw_specs["face"]["color"] = "#dddddd"
    draw_specs["face"]["alpha"] = 0.6
    draw_specs["vert"]["visible"] = False
    draw_specs.update(line_tension_edge_kwds(color_range=crange, colormap=EDGE_CMAP, width=1.2))
    create_gif(history, str(out_path), coords=COORDS, num_frames=NUM_GIF_FRAMES, dpi=110, **draw_specs)


def save_stills(history, out_dir: Path, sigma: float):
    """Save still frames coloured by line tension at a few timepoints."""
    from tyssue.draw import sheet_view
    from tyssue.geometry.sheet_geometry import SheetGeometry

    crange = (-3.0 * sigma, 3.0 * sigma)
    times = list(history.time_stamps)
    if not times:
        return
    for frac in STILL_FRACTIONS:
        t = times[int(round(frac * (len(times) - 1)))]
        sheet = history.retrieve(t)
        SheetGeometry.update_all(sheet)
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        sheet_view(
            sheet,
            coords=COORDS,
            ax=ax,
            face={"visible": True, "color": "#dddddd", "alpha": 0.6},
            edge={"visible": True, "color": _edge_rgba(sheet.edge_df, crange), "width": 1.2},
        )
        ax.set_title(f"t = {float(t):.1f}", fontsize=9)
        ax.set_aspect("equal")
        fig.savefig(out_dir / f"still_t{float(t):04.1f}.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def vertex_movement(history):
    """Degree of stochastic vertex movement.

    Vertices are matched by ``unique_id`` between consecutive frames (identity is
    stable across reconnections; the positional index is not). Returns:
      * mean_step_rms   — RMS per-step vertex displacement, averaged over the run
      * mean_path_length — total distance travelled per vertex, averaged over verts
    """
    times = list(history.time_stamps)
    prev = None
    per_step_rms = []
    path = defaultdict(float)
    for t in times:
        sheet = history.retrieve(t)
        vdf = sheet.vert_df
        uids = vdf["unique_id"].to_numpy()
        pos = vdf[COORDS].to_numpy(dtype=float)
        cur = {int(u): pos[i] for i, u in enumerate(uids)}
        if prev is not None:
            common = sorted(set(cur) & set(prev))
            if common:
                deltas = np.array([np.linalg.norm(cur[u] - prev[u]) for u in common])
                per_step_rms.append(float(np.sqrt(np.mean(deltas ** 2))))
                for u, d in zip(common, deltas):
                    path[u] += float(d)
        prev = cur
    mean_step_rms = float(np.mean(per_step_rms)) if per_step_rms else 0.0
    mean_path_length = float(np.mean(list(path.values()))) if path else 0.0
    return mean_step_rms, mean_path_length


def _face_adjacency(sheet):
    """Set of neighbouring face pairs, keyed by face ``unique_id``.

    Two faces are neighbours when they share at least two vertices (i.e. an edge).
    Keying by unique_id makes the set comparable across frames despite the vertex/
    face re-indexing that tyssue performs after every reconnection.
    """
    edf = sheet.edge_df
    vuid = sheet.vert_df["unique_id"].to_numpy()
    fuid = sheet.face_df["unique_id"].to_numpy()

    face_vsets = defaultdict(set)
    for f, s in zip(edf["face"].to_numpy(), edf["srce"].to_numpy()):
        face_vsets[int(f)].add(int(vuid[int(s)]))

    vert_faces = defaultdict(set)
    for fpos, vset in face_vsets.items():
        for v in vset:
            vert_faces[v].add(fpos)

    shared = defaultdict(int)
    for faces in vert_faces.values():
        fl = sorted(faces)
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                shared[(fl[i], fl[j])] += 1

    adj = set()
    for (a, b), n in shared.items():
        if n >= 2:
            adj.add(frozenset((int(fuid[a]), int(fuid[b]))))
    return adj


def count_t1_transitions(history):
    """Estimate the number of T1 transitions (neighbour exchanges) over the run.

    A T1 swaps one face-adjacency for another, so between consecutive frames the
    symmetric difference of the adjacency set changes by two pairs per event. We
    accumulate ``len(symmetric_difference) / 2`` across the run. (tyssue/eulersolver
    keep no transition counter, so this adjacency-diff is the measurement.)
    """
    times = list(history.time_stamps)
    prev = None
    total = 0.0
    for t in times:
        adj = _face_adjacency(history.retrieve(t))
        if prev is not None:
            total += len(prev ^ adj) / 2.0
        prev = adj
    return float(total)


# ---------------------------------------------------------------------------
# Summary figures
# ---------------------------------------------------------------------------
def _heatmap(ax, grid, title, cbar_label):
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(SIGMAS)), [f"{s:g}" for s in SIGMAS])
    ax.set_yticks(range(len(TAUS)), [f"{t:g}" for t in TAUS])
    ax.set_xlabel("sigma (noise amplitude)")
    ax.set_ylabel("tau (relaxation time)")
    ax.set_title(title, fontsize=11)
    thresh = np.nanmin(grid) + 0.6 * (np.nanmax(grid) - np.nanmin(grid))
    for i in range(len(TAUS)):
        for j in range(len(SIGMAS)):
            v = grid[i, j]
            ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=8,
                    color="white" if v < thresh else "black")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=9)


def build_summary(df: pd.DataFrame, out_dir: Path):
    move = df.pivot(index="tau", columns="sigma", values="mean_step_rms").reindex(index=TAUS, columns=SIGMAS)
    t1 = df.pivot(index="tau", columns="sigma", values="t1_count").reindex(index=TAUS, columns=SIGMAS)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    _heatmap(ax, move.to_numpy(), "Stochastic vertex movement", "mean per-step RMS displacement")
    fig.tight_layout()
    fig.savefig(out_dir / "movement_heatmap.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    _heatmap(ax, t1.to_numpy(), "T1 transitions (neighbour exchanges)", "estimated T1 count")
    fig.tight_layout()
    fig.savefig(out_dir / "t1_heatmap.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # Movement vs sigma, one line per tau (identity by colour + direct-labelled).
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for k, tau in enumerate(TAUS):
        sub = df[df["tau"] == tau].sort_values("sigma")
        ax.plot(sub["sigma"], sub["mean_step_rms"], "-o", color=TAU_COLORS[k % len(TAU_COLORS)],
                markersize=5, linewidth=2, label=f"tau = {tau:g}")
    ax.set_xlabel("sigma (noise amplitude)")
    ax.set_ylabel("mean per-step RMS displacement")
    ax.set_title("Vertex movement vs noise amplitude", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "metric_vs_sigma.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


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

    rows = []
    for tau in TAUS:
        for sigma in SIGMAS:
            tag = f"tau{tau:0.2f}_sigma{sigma:0.2f}"
            run_dir = OUT_DIR / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run] {tag} ...", flush=True)

            history = run_one(core, tau, sigma, TF, DT)

            save_gif(history, run_dir / "stochastic.gif", sigma)
            save_stills(history, run_dir, sigma)
            if SAVE_HISTORY:
                archive_thinned(history, run_dir / "history.hf5", ARCHIVE_FRAMES)

            mean_step_rms, mean_path_length = vertex_movement(history)
            t1_count = count_t1_transitions(history)
            print(f"       movement(rms)={mean_step_rms:.4g} path={mean_path_length:.4g} "
                  f"T1={t1_count:.0f}", flush=True)
            rows.append({
                "tau": tau, "sigma": sigma,
                "mean_step_rms": mean_step_rms,
                "mean_path_length": mean_path_length,
                "t1_count": t1_count,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "metrics.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'metrics.csv'}")

    summary_dir = OUT_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    build_summary(df, summary_dir)
    print(f"wrote summary figures to {summary_dir}")


if __name__ == "__main__":
    main()
