"""Cell-jamming and parameter-gradient migration experiments.

Two single-cell migration scenarios on a flat epithelial square sheet, reproduced
exactly from ``Notebooks/simulation_walkthrough.ipynb`` (specs 05 and 06) — same
config, same ``tf``/``dt``, no parameter sweep. One cell (face index 96) is given
an ``ActiveMigration`` drive and migrates through the tissue while stochastic
line-tension fluctuations act on every edge.

  * **Jamming** — the ``CellJamming`` process fires at ``t = 300`` (run to
    ``t = 400``, ``dt = 0.1``), ramping every cell's preferred perimeter down so
    the tissue solidifies and arrests the migrating cell.
  * **Gradient** — the ``ParameterGradient`` step imposes a linear
    ``prefered_perimeter`` gradient along x (run to ``t = 300``, ``dt = 0.05``), so
    the migrating cell moves through a stiffness gradient.

For each scenario we render a colour-coded GIF (faces by preferred perimeter, the
migrating cell's junctions highlighted) and still snapshots, then analyse the
migrating cell's motion:

  * Jamming: displacement of the migrating cell from its initial position over
    time, with a dotted line marking the jamming-transition timepoint.
  * Gradient: the same displacement-over-time plot, plus instantaneous velocity as
    a function of x-position, plus the migrating cell's preferred perimeter over
    time.

Everything generated lands under ``outputs/`` (input mesh under ``data/``), both
git-ignored. Run from the repo's ``vivarium-tyssue`` conda env:

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

NUM_GIF_FRAMES = 200
N_STILLS = 5
FIG_DPI = 300                # publication-quality raster resolution for stills / plots
ARCHIVE_FRAMES = 40          # subsampled frames per archived History (keeps hf5 small)
COORDS = ["x", "y"]
SEED = 20260714


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
                "datasets": ["Datasets"],
                "network_changed": ["Network Changed"],
                "behaviors_update": ["Behaviors"],
            },
            "interval": dt,
        },
        "Stochastic": {
            "_type": "process",
            "address": "local:StochasticLineTension",
            "config": {"tau": TAU, "sigma": SIGMA},
            "inputs": {"datasets": ["Datasets"]},
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
        "inputs": {"global_time": ["global_time"], "datasets": ["Datasets"]},
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
        "inputs": {"datasets": ["Datasets"]},
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
# Drawing — GIF + stills (faces by prefered_perimeter, migrating cell highlighted)
# ---------------------------------------------------------------------------
def _face_edge_specs(pp_range):
    """The notebook's face_param + migrating-cell edge draw kwds."""
    from vivarium_tyssue.draw import face_param_kwds, migrating_cell_edge_kwds
    face = face_param_kwds("prefered_perimeter", color_range=pp_range)["face"]
    edge = migrating_cell_edge_kwds(
        highlight_color="cyan", highlight_alpha=0.6,
        base_color="black", base_alpha=0.8, width=1.5,
    )["edge"]
    return face, edge


def save_gif(history, out_path: Path, pp_range):
    from tyssue import config
    from tyssue.draw import create_gif
    face, edge = _face_edge_specs(pp_range)
    ds = config.draw.sheet_spec()
    ds["face"].update(face)
    ds["edge"].update(edge)
    ds["axis"].update({
        "color_bar": True, "color_bar_cmap": "Reds", "color_bar_range": tuple(pp_range),
        "color_bar_label": "prefered perimeter", "color_bar_target": "face",
    })
    create_gif(history, str(out_path), coords=COORDS, num_frames=NUM_GIF_FRAMES, dpi=110, **ds)


def save_stills(history, out_dir: Path, pp_range):
    from tyssue.draw import sheet_view
    from tyssue.geometry.sheet_geometry import SheetGeometry
    face, edge = _face_edge_specs(pp_range)
    face_color_fn, edge_color_fn = face["color"], edge["color"]
    times = list(history.time_stamps)
    if not times:
        return
    fracs = np.linspace(0.0, 1.0, N_STILLS)
    for frac in fracs:
        t = times[int(round(frac * (len(times) - 1)))]
        sheet = history.retrieve(t)
        SheetGeometry.update_all(sheet)
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        sheet_view(
            sheet, coords=COORDS, ax=ax,
            face={"visible": True, "color": face_color_fn(sheet), "alpha": 0.9},
            edge={"visible": True, "color": edge_color_fn(sheet), "width": 1.5},
        )
        ax.set_title(f"t = {float(t):.1f}", fontsize=9)
        ax.set_aspect("equal")
        fig.savefig(out_dir / f"still_t{float(t):05.1f}.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)


def archive_thinned(history, path: Path, keep_frames: int):
    """Archive History to HDF5 keeping only ``keep_frames`` subsampled frames."""
    times = np.array(list(history.time_stamps))
    if times.size > keep_frames:
        idx = np.unique(np.round(np.linspace(0, times.size - 1, keep_frames)).astype(int))
        keep = set(times[idx].tolist())
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


# ---------------------------------------------------------------------------
# Analysis — migrating-cell trace
# ---------------------------------------------------------------------------
def migrating_trace(history) -> pd.DataFrame:
    """Per-timepoint state of the migrating cell (the face with the largest
    ``migration_strength``), read straight from the stacked face dataframe.

    Columns: time, x, y (centroid), prefered_perimeter, displacement (from t0),
    speed (instantaneous |velocity|).
    """
    face = history.datasets["face"]
    mig = face[face["migration_strength"] > 0].sort_values("time").drop_duplicates("time", keep="first")
    mig = mig.reset_index(drop=True)
    x0, y0 = float(mig.loc[0, "x"]), float(mig.loc[0, "y"])
    out = pd.DataFrame({
        "time": mig["time"].astype(float).to_numpy(),
        "x": mig["x"].astype(float).to_numpy(),
        "y": mig["y"].astype(float).to_numpy(),
        "prefered_perimeter": mig["prefered_perimeter"].astype(float).to_numpy(),
    })
    out["displacement"] = np.hypot(out["x"] - x0, out["y"] - y0)
    dt = out["time"].diff()
    dist = np.hypot(out["x"].diff(), out["y"].diff())
    out["speed"] = dist / dt
    return out


def _face_circularity(history):
    """All alive faces across all frames with a ``circularity`` column.

    Circularity = 4*pi*area / perimeter**2 (1.0 for a perfect circle, lower for
    elongated / irregular cells).
    """
    face = history.datasets["face"]
    df = face[(face["is_alive"] > 0) & (face["perimeter"] > 0) & (face["area"] > 0)].copy()
    df["circularity"] = 4.0 * np.pi * df["area"].astype(float) / df["perimeter"].astype(float) ** 2
    return df


def circularity_over_time(history) -> pd.DataFrame:
    """Mean cell circularity at each timepoint. Columns: time, circularity."""
    df = _face_circularity(history)
    return df.groupby("time")["circularity"].mean().reset_index()


def circularity_along_x(history, nbins: int = 30) -> pd.DataFrame:
    """Mean cell circularity binned by face x-position, pooled over all timepoints.
    Columns: xcenter, circularity (mean), std, count.
    """
    df = _face_circularity(history)
    x = df["x"].astype(float)
    bins = np.linspace(x.min(), x.max(), nbins + 1)
    df["xbin"] = pd.cut(x, bins, include_lowest=True)
    g = df.groupby("xbin", observed=True)["circularity"].agg(["mean", "std", "count"]).reset_index()
    g["xcenter"] = g["xbin"].apply(lambda b: float(b.mid))
    return g[["xcenter", "mean", "std", "count"]].rename(columns={"mean": "circularity"})


# ---------------------------------------------------------------------------
# Analysis plots
# ---------------------------------------------------------------------------
LINE_COLOR = "#0072B2"     # Okabe-Ito blue
EVENT_COLOR = "#D55E00"    # Okabe-Ito vermillion (jamming-transition marker)


def plot_displacement(trace, out_path: Path, title, jamming_time=None):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(trace["time"], trace["displacement"], "-", color=LINE_COLOR, linewidth=2,
            label="migrating cell")
    if jamming_time is not None:
        ax.axvline(jamming_time, linestyle=":", color=EVENT_COLOR, linewidth=1.8)
        ax.text(jamming_time, ax.get_ylim()[1], "  jamming", color=EVENT_COLOR,
                va="top", ha="left", fontsize=9)
    ax.set_xlabel("time")
    ax.set_ylabel("displacement from initial position")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_vs_x(trace, out_path: Path, title):
    t = trace.dropna(subset=["speed"])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    sc = ax.scatter(t["x"], t["speed"], c=t["time"], cmap="viridis", s=14)
    ax.plot(t["x"], t["speed"], "-", color="0.7", linewidth=0.7, zorder=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("time", fontsize=9)
    ax.set_xlabel("migrating cell x-position")
    ax.set_ylabel("instantaneous speed")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pref_perimeter_vs_time(trace, out_path: Path, title):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(trace["time"], trace["prefered_perimeter"], "-", color=LINE_COLOR, linewidth=2)
    ax.set_xlabel("time")
    ax.set_ylabel("prefered perimeter (migrating cell)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_circularity_over_time(circ, out_path: Path, title, jamming_time=None):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(circ["time"], circ["circularity"], "-", color=LINE_COLOR, linewidth=2)
    if jamming_time is not None:
        ax.axvline(jamming_time, linestyle=":", color=EVENT_COLOR, linewidth=1.8)
        ax.text(jamming_time, ax.get_ylim()[1], "  jamming", color=EVENT_COLOR,
                va="top", ha="left", fontsize=9)
    ax.set_xlabel("time")
    ax.set_ylabel("mean cell circularity  (4πA / P²)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_circularity_along_x(circ, out_path: Path, title):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    m = circ["circularity"].to_numpy()
    s = circ["std"].to_numpy()
    x = circ["xcenter"].to_numpy()
    ax.fill_between(x, m - s, m + s, color=LINE_COLOR, alpha=0.15, linewidth=0)
    ax.plot(x, m, "-o", color=LINE_COLOR, linewidth=2, markersize=4)
    ax.set_xlabel("face x-position")
    ax.set_ylabel("mean cell circularity  (4πA / P²)")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    np.random.seed(SEED)
    sys.path.insert(0, str(REPO))
    from vivarium_tyssue.core import build_core

    dataset = ensure_dataset()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core = build_core()

    # ---- Jamming -------------------------------------------------------
    jam_dir = OUT_DIR / "jamming"
    jam_dir.mkdir(parents=True, exist_ok=True)
    print(f"[jamming] running tf={JAM_TF} dt={JAM_DT} (python backend) ...", flush=True)
    hist_j = run(core, build_jamming_spec(dataset), JAM_TF)
    save_gif(hist_j, jam_dir / "jamming.gif", pp_range=(3.0, 4.2))
    save_stills(hist_j, jam_dir, pp_range=(3.0, 4.2))
    archive_thinned(hist_j, jam_dir / "history.hf5", ARCHIVE_FRAMES)
    trace_j = migrating_trace(hist_j)
    trace_j.to_csv(jam_dir / "migrating_trace.csv", index=False)
    plot_displacement(trace_j, jam_dir / "migrating_displacement.png",
                      "Jamming: migrating-cell displacement", jamming_time=JAM_TRIGGER)
    circ_j = circularity_over_time(hist_j)
    circ_j.to_csv(jam_dir / "circularity_over_time.csv", index=False)
    plot_circularity_over_time(circ_j, jam_dir / "circularity_over_time.png",
                               "Jamming: mean cell circularity over time", jamming_time=JAM_TRIGGER)
    print(f"[jamming] final displacement = {trace_j['displacement'].iloc[-1]:.3f}", flush=True)

    # ---- Gradient ------------------------------------------------------
    grad_dir = OUT_DIR / "gradient"
    grad_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gradient] running tf={GRAD_TF} dt={GRAD_DT} (python backend) ...", flush=True)
    hist_g = run(core, build_gradient_spec(dataset), GRAD_TF)
    save_gif(hist_g, grad_dir / "gradient.gif", pp_range=(3.0, 4.2))
    save_stills(hist_g, grad_dir, pp_range=(3.0, 4.2))
    archive_thinned(hist_g, grad_dir / "history.hf5", ARCHIVE_FRAMES)
    trace_g = migrating_trace(hist_g)
    trace_g.to_csv(grad_dir / "migrating_trace.csv", index=False)
    plot_displacement(trace_g, grad_dir / "migrating_displacement.png",
                      "Gradient: migrating-cell displacement")
    plot_velocity_vs_x(trace_g, grad_dir / "migrating_velocity_vs_x.png",
                       "Gradient: migrating-cell velocity vs x-position")
    plot_pref_perimeter_vs_time(trace_g, grad_dir / "prefered_perimeter_vs_time.png",
                                "Gradient: migrating-cell prefered perimeter over time")
    circ_gx = circularity_along_x(hist_g)
    circ_gx.to_csv(grad_dir / "circularity_along_x.csv", index=False)
    plot_circularity_along_x(circ_gx, grad_dir / "circularity_along_x.png",
                             "Gradient: mean cell circularity along x (all timepoints)")
    print(f"[gradient] final displacement = {trace_g['displacement'].iloc[-1]:.3f}", flush=True)

    print(f"\ndone — outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
