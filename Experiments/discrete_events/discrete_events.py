"""Discrete-event experiments — random divisions, random deaths, Gillespie.

Three scenarios exercising the discrete-event processes:

  * **divisions** — a ``CellDivisions`` process fires cell divisions as a Poisson
    process (rate-based random times) on a plain **flat square sheet**
    (``test_square.hf5``, ``SheetGeometry``), exactly the tissue used by the other
    Experiments. There are no biological cell types here — every cell starts
    ``"normal"``; a cell being actively grown toward division is highlighted
    ``"dividing"`` (magenta). This just demonstrates that random divisions work.
    Renders a colour-coded 2-D GIF + stills.
  * **deaths** — a ``CellDeaths`` process fires apoptotic extrusions as a Poisson
    process on the same flat square, using the same ``apoptosis_extrusion``
    behaviour the Gillespie process drives; a dying cell is highlighted
    ``"extruding"`` (black). Colour-coded 2-D GIF + stills.
  * **gillespie** — the full Gillespie biochemistry (``Gillespie`` process) on the
    3-D crypt cylinder (``crypt_cylinder.hf5``) exactly as in ``tests/tests.py`` /
    ``Notebooks/simulation_walkthrough.ipynb`` (``tf=72``, ``dt=0.005``). Renders
    the cell-type colour-coded 3-D GIF and three analyses:
      1. distribution of cell types over time,
      2. spatial distribution of cell types along z (crypt length),
      3. spatial distribution of event types (division / differentiation /
         extrusion) along z.

Everything generated lands under ``outputs/`` (input meshes under ``data/``), both
git-ignored. Run from the repo's ``vivarium-tyssue`` conda env (needs ImageMagick
``magick`` on PATH for GIF rendering):

    conda activate vivarium-tyssue
    cd Experiments/discrete_events
    python discrete_events.py            # runs all three
    python discrete_events.py divisions  # or a single scenario: divisions|deaths|gillespie
"""
from __future__ import annotations

import copy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

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
FIG_DPI = 300          # publication-quality raster resolution for stills / plots
GIF_DPI = 120          # animations (kept lower — many frames)
NUM_GIF_FRAMES = 120
N_STILLS = 5
SEED = 20260715

# Discrete-event scenarios on the flat square (Poisson processes).
DIV_TF, DIV_DT = 25.0, 0.05
DIV_RATE, DIV_CRIT, DIV_GROWTH = 0.4, 2.0, 0.3           # CellDivisions
DEATH_TF, DEATH_DT = 25.0, 0.05
DEATH_RATE, DEATH_CRIT, DEATH_SHRINK = 0.4, 0.3, 0.3     # CellDeaths

# Flat-sheet cell "states": a neutral background plus the two event highlights,
# using the Gillespie dividing/extruding colours so the palette reads the same.
FLAT_TYPE_COLORS = {"normal": "#CFCFCF", "dividing": "#C71FE0", "extruding": "#000000"}

# Gillespie scenario (identical to tests.py / walkthrough).
GILL_TF, GILL_DT = 72.0, 0.005

Z_NBINS = 12       # event-type histogram along z
Z_NBINS_CT = 48    # finer bins for the cell-type-along-z line plot


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
    """Plain flat square sheet (SheetGeometry / model_factory_bound), as used by
    the other Experiments. Every cell starts ``cell_type="normal"`` so the division
    / extrusion behaviours can flag the active cell for colour-coding."""
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
    (VesselGeometry / model_factory_vessel, python backend)."""
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
# Drawing — 2-D flat sheet (divisions / deaths), faces by cell "state"
# ---------------------------------------------------------------------------
def _flat_face_color(sheet):
    """(Nf, 4) RGBA: normal cells grey, dividing magenta, dying/extruding black."""
    return np.array([
        mcolors.to_rgba(FLAT_TYPE_COLORS.get(ct, FLAT_TYPE_COLORS["normal"]))
        for ct in sheet.face_df["cell_type"]
    ])


def _flat_legend_handles():
    return [Patch(facecolor=FLAT_TYPE_COLORS[k], edgecolor="#808080", label=k)
            for k in ("normal", "dividing", "extruding")]


def _frame_limits(history, times, coords):
    """Fixed (min, max) per-axis limits from the first frame, with a 5% margin."""
    sheet0 = history.retrieve(times[0])
    bounds = sheet0.vert_df[coords].describe().loc[["min", "max"]]
    margin = (bounds.loc["max"] - bounds.loc["min"]).max() * 0.05
    return {c: (bounds.loc["min", c] - margin, bounds.loc["max", c] + margin) for c in coords}


def _draw_2d_frame(sheet, title, lims):
    """Draw one flat-sheet frame (faces by cell state) into a matplotlib Axes.
    Returns the Figure, or None if the frame can't be drawn."""
    from tyssue.draw import sheet_view
    try:
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        sheet_view(
            sheet, coords=COORDS_2D, ax=ax,
            face={"visible": True, "color": _flat_face_color(sheet), "alpha": 1.0},
            edge={"visible": True, "color": "#808080", "width": 1.0},
        )
        ax.set_aspect("equal")
        ax.set_xlim(*lims["x"])
        ax.set_ylim(*lims["y"])
        ax.set_title(title, fontsize=10)
        ax.legend(handles=_flat_legend_handles(), loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    except Exception as exc:  # noqa: BLE001
        print(f"frame {title} failed ({type(exc).__name__}: {exc}); skipping")
        plt.close("all")
        return None
    return fig


def save_gif_2d(history, out_path: Path):
    times = list(history.time_stamps)
    if not times:
        return
    lims = _frame_limits(history, times, COORDS_2D)
    idx = np.unique(np.round(np.linspace(0, len(times) - 1,
                                         min(NUM_GIF_FRAMES, len(times)))).astype(int))
    tmp = Path(tempfile.mkdtemp())
    n = 0
    try:
        for i in idx:
            t = times[int(i)]
            fig = _draw_2d_frame(history.retrieve(t), f"t = {float(t):.1f}", lims)
            if fig is None:
                continue
            fig.savefig(tmp / f"frame_{n:04d}.png", dpi=GIF_DPI, bbox_inches="tight")
            plt.close(fig)
            n += 1
        if n == 0:
            print(f"no renderable frames for {out_path.name}; skipping GIF")
            return
        subprocess.run(["magick", "-delay", "12", "-loop", "0",
                        (tmp / "frame_*.png").as_posix(), str(out_path)], check=True)
        print(f"  wrote {out_path.name} ({n} frames)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def save_stills_2d(history, out_dir: Path):
    times = list(history.time_stamps)
    if not times:
        return
    lims = _frame_limits(history, times, COORDS_2D)
    for frac in np.linspace(0.0, 1.0, N_STILLS):
        t = times[int(round(frac * (len(times) - 1)))]
        fig = _draw_2d_frame(history.retrieve(t), f"t = {float(t):.1f}", lims)
        if fig is None:
            continue
        fig.savefig(out_dir / f"still_t{float(t):06.1f}.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Drawing — 3-D crypt (gillespie), faces by cell_type
#
# We render every frame ourselves (rather than via tyssue.create_gif_3d) because
# that helper calls savefig OUTSIDE its per-frame try/except, so a single frame
# whose matplotlib-3D projection goes singular — which happens on the crypt after
# a division/extrusion reindexes the mesh — aborts the whole GIF. Here each frame
# is guarded, so a bad frame is skipped and the rest of the animation survives.
# ---------------------------------------------------------------------------
def _draw_3d_frame(sheet, lims, title, figsize=None):
    """Draw one crypt frame (faces by cell_type) into a fresh matplotlib Axes3D,
    the same path create_gif_3d uses per frame. Returns the Figure, or None."""
    from tyssue import config
    from tyssue.draw.plt_draw import (
        sheet_view_3d, patch_2d_collections_to_3d, _auto_tick_fontsize_3d,
    )
    from vivarium_tyssue.draw import crypt_cell_type_kwds, CELL_TYPE_COLORS

    ds = config.draw.sheet_spec()
    ds["face"]["visible"] = True
    ds["face"]["alpha"] = 1.0
    # Grey (not black) edges so the black "extruding" faces are distinguishable
    # from the cell outlines — a black-on-black mesh hides dying cells entirely.
    ds["edge"]["color"] = "#808080"
    ds["face"]["color"] = crypt_cell_type_kwds(sheet)["face"]["color"]
    try:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=30, azim=45)
        fig, ax = sheet_view_3d(
            sheet, coords=COORDS_3D, ax=ax,
            legend=CELL_TYPE_COLORS, cull_back_edges=True, **ds,
        )
        patch_2d_collections_to_3d(ax)
        ax.set(xlim=lims["x"], ylim=lims["y"], zlim=lims["z"])
        # sheet_view_3d set the 3-D box aspect from THIS frame's auto-scaled data
        # extent; we just overrode the limits with the fixed ones, so recompute the
        # box aspect from those fixed ranges. Otherwise the crypt's z-axis proportion
        # (it is much longer than x/y) drifts frame to frame — squashed on some,
        # correct on others — as divisions/extrusions shift the per-frame z-extent.
        ax.set_box_aspect((lims["x"][1] - lims["x"][0],
                           lims["y"][1] - lims["y"][0],
                           lims["z"][1] - lims["z"][0]))
        # sheet_view_3d also sized the tick labels from this frame's own auto-scaled
        # ranges, so the font jumps frame to frame. Recompute it from the FIXED limits
        # so every frame gets the same tick-label size.
        _auto_tick_fontsize_3d(ax, base_size=8, min_size=4)
        ax.set_title(title, fontsize=9)
        # Force the (occasionally singular) 3-D projection now, so a bad frame
        # raises here inside the guard rather than later at savefig.
        fig.canvas.draw()
    except Exception as exc:  # noqa: BLE001
        print(f"frame {title} failed ({type(exc).__name__}: {exc}); skipping")
        plt.close("all")
        return None
    return fig


def save_gif_3d(history, out_path: Path):
    times = list(history.time_stamps)
    if not times:
        return
    lims = _frame_limits(history, times, COORDS_3D)
    idx = np.unique(np.round(np.linspace(0, len(times) - 1,
                                         min(NUM_GIF_FRAMES, len(times)))).astype(int))
    tmp = Path(tempfile.mkdtemp())
    n = 0
    try:
        for i in idx:
            t = times[int(i)]
            fig = _draw_3d_frame(history.retrieve(t), lims, f"t = {float(t):.2f}", figsize=(5.0, 8.0))
            if fig is None:
                continue
            fig.savefig(tmp / f"frame_{n:04d}.png", dpi=GIF_DPI)
            plt.close(fig)
            n += 1
        if n == 0:
            print(f"no renderable frames for {out_path.name}; skipping GIF")
            return
        subprocess.run(["magick", "-delay", "12", "-loop", "0",
                        (tmp / "frame_*.png").as_posix(), str(out_path)], check=True)
        print(f"  wrote {out_path.name} ({n} frames)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def save_stills_3d(history, out_dir: Path):
    times = list(history.time_stamps)
    if not times:
        return
    lims = _frame_limits(history, times, COORDS_3D)
    for frac in np.linspace(0.0, 1.0, N_STILLS):
        t = times[int(round(frac * (len(times) - 1)))]
        # Same fixed figsize as the gif (save_gif_3d) and NO bbox_inches="tight":
        # tight-cropping would resize each still to its own content, so the crypt's
        # z-axis would look a different length in each snapshot. A constant figsize
        # keeps every still identical in size, box_aspect keeps it proportional.
        fig = _draw_3d_frame(history.retrieve(t), lims, f"t = {float(t):.2f}", figsize=(5.0, 8.0))
        if fig is None:
            continue
        fig.savefig(out_dir / f"still_t{float(t):06.2f}.png", dpi=FIG_DPI)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis (Gillespie)
# ---------------------------------------------------------------------------
BIO_TYPES = ["sc", "pc", "ent", "gc"]                          # true crypt cell types
CELL_TYPE_ORDER = BIO_TYPES + ["dividing", "extruding"]        # + transient states


def _type_palette():
    from vivarium_tyssue.draw import CELL_TYPE_COLORS
    return CELL_TYPE_COLORS


def cell_type_over_time(history) -> pd.DataFrame:
    """Count of each cell type at every recorded timepoint (wide: time × types)."""
    face = history.datasets["face"]
    df = face[face["is_alive"] > 0] if "is_alive" in face.columns else face
    counts = df.groupby(["time", "cell_type"]).size().unstack(fill_value=0)
    counts = counts.reindex(sorted(counts.index)).reset_index()
    return counts


def cell_type_along_z(history, nbins: int = Z_NBINS_CT) -> pd.DataFrame:
    """Mean count of each cell type per z-bin, averaged over all timepoints
    (long: zcenter, cell_type, count)."""
    face = history.datasets["face"]
    df = (face[face["is_alive"] > 0] if "is_alive" in face.columns else face).copy()
    z = df["z"].astype(float)
    bins = np.linspace(z.min(), z.max(), nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    df["zbin"] = pd.cut(z, bins, include_lowest=True, labels=centers)
    n_frames = df["time"].nunique()
    g = df.groupby(["zbin", "cell_type"], observed=True).size().reset_index(name="count")
    g["count"] = g["count"] / max(n_frames, 1)          # per-frame mean occupancy
    g["zcenter"] = g["zbin"].astype(float)
    return g[["zcenter", "cell_type", "count"]]


def events_along_z(history, events: list, nbins: int = Z_NBINS) -> pd.DataFrame:
    """Count of each event type per z-bin. Each event's z is the mean z of its
    cell (by unique_id) across the recorded history (long: zcenter, func, count)."""
    face = history.datasets["face"]
    z_by_uid = face.groupby("unique_id")["z"].mean()
    z_all = face["z"].astype(float)
    bins = np.linspace(z_all.min(), z_all.max(), nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    rows = []
    for e in events:
        uid = e["cell_uid"]
        if uid is None or uid not in z_by_uid.index:
            continue
        rows.append({"func": e["func"], "z": float(z_by_uid.loc[uid])})
    if not rows:
        return pd.DataFrame(columns=["zcenter", "func", "count"])
    edf = pd.DataFrame(rows)
    edf["zbin"] = pd.cut(edf["z"], bins, include_lowest=True, labels=centers)
    g = edf.groupby(["zbin", "func"], observed=True).size().reset_index(name="count")
    g["zcenter"] = g["zbin"].astype(float)
    return g[["zcenter", "func", "count"]]


# ---------------------------------------------------------------------------
# Analysis plots  (legends placed OUTSIDE the axes to avoid overlap)
# ---------------------------------------------------------------------------
def _legend_outside(ax, **kw):
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8, **kw)


def plot_cell_type_over_time(counts: pd.DataFrame, out_path: Path):
    palette = _type_palette()
    types = [c for c in CELL_TYPE_ORDER if c in counts.columns]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.stackplot(
        counts["time"],
        *[counts[t].to_numpy() for t in types],
        labels=types,
        colors=[palette.get(t, "#999999") for t in types],
        alpha=0.9,
    )
    ax.set_xlabel("time")
    ax.set_ylabel("cell count")
    ax.set_title("Gillespie crypt: cell-type distribution over time", fontsize=11)
    ax.margins(x=0)
    _legend_outside(ax)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cell_type_along_z(g: pd.DataFrame, out_path: Path):
    """One line per (biological) cell type: mean occupancy vs z."""
    palette = _type_palette()
    wide = g.pivot(index="zcenter", columns="cell_type", values="count").fillna(0.0).sort_index()
    types = [c for c in BIO_TYPES if c in wide.columns]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    z = wide.index.to_numpy()
    for t in types:
        ax.plot(z, wide[t].to_numpy(), "-", color=palette.get(t, "#999999"),
                linewidth=1.8, label=t)
    ax.set_xlabel("z position (crypt length)")
    ax.set_ylabel("mean cells per z-bin")
    ax.set_title("Gillespie crypt: cell-type spatial distribution along z", fontsize=11)
    ax.margins(x=0)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    _legend_outside(ax)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


EVENT_LABELS = {
    "division": "division",
    "differentiation": "differentiation",
    "apoptosis_extrusion": "extrusion (death)",
}
EVENT_COLORS = {
    "division": "#C71FE0",
    "differentiation": "#0072B2",
    "apoptosis_extrusion": "#000000",
}


def plot_events_along_z(g: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    if g.empty:
        ax.text(0.5, 0.5, "no events recorded", ha="center", va="center", transform=ax.transAxes)
    else:
        wide = g.pivot(index="zcenter", columns="func", values="count").fillna(0.0).sort_index()
        funcs = [f for f in EVENT_COLORS if f in wide.columns] + \
                [f for f in wide.columns if f not in EVENT_COLORS]
        z = wide.index.to_numpy()
        width = (z[1] - z[0]) * 0.8 / max(len(funcs), 1) if len(z) > 1 else 0.5
        for i, f in enumerate(funcs):
            ax.bar(z + (i - (len(funcs) - 1) / 2) * width, wide[f].to_numpy(), width=width,
                   color=EVENT_COLORS.get(f, "#999999"), label=EVENT_LABELS.get(f, f), alpha=0.9)
        _legend_outside(ax)
    ax.set_xlabel("z position (crypt length)")
    ax.set_ylabel("event count")
    ax.set_title("Gillespie crypt: event-type spatial distribution along z", fontsize=11)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scenario drivers
# ---------------------------------------------------------------------------
def run_divisions(core):
    dataset = ensure_dataset(FLAT_DATASET)
    out = OUT_DIR / "divisions"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[divisions] flat square, tf={DIV_TF} dt={DIV_DT} rate={DIV_RATE} ...", flush=True)
    history, _ = run(core, build_divisions_spec(dataset), DIV_TF)
    save_gif_2d(history, out / "divisions.gif")
    save_stills_2d(history, out)
    print("[divisions] done", flush=True)


def run_deaths(core):
    dataset = ensure_dataset(FLAT_DATASET)
    out = OUT_DIR / "deaths"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[deaths] flat square, tf={DEATH_TF} dt={DEATH_DT} rate={DEATH_RATE} ...", flush=True)
    history, _ = run(core, build_deaths_spec(dataset), DEATH_TF)
    save_gif_2d(history, out / "deaths.gif")
    save_stills_2d(history, out)
    print("[deaths] done", flush=True)


def run_gillespie(core):
    dataset = ensure_dataset(CRYPT_DATASET)
    out = OUT_DIR / "gillespie"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[gillespie] crypt, tf={GILL_TF} dt={GILL_DT} (python) ...", flush=True)
    history, events = run(core, build_gillespie_spec(dataset), GILL_TF, capture_behaviors=True)
    print(f"[gillespie] captured {len(events)} events", flush=True)
    save_gif_3d(history, out / "gillespie.gif")
    save_stills_3d(history, out)

    counts = cell_type_over_time(history)
    counts.to_csv(out / "cell_type_over_time.csv", index=False)
    plot_cell_type_over_time(counts, out / "cell_type_over_time.png")

    ctz = cell_type_along_z(history)
    ctz.to_csv(out / "cell_type_along_z.csv", index=False)
    plot_cell_type_along_z(ctz, out / "cell_type_along_z.png")

    evz = events_along_z(history, events)
    evz.to_csv(out / "events_along_z.csv", index=False)
    plot_events_along_z(evz, out / "events_along_z.png")

    pd.DataFrame(events).to_csv(out / "events.csv", index=False)
    print("[gillespie] done", flush=True)


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

    print(f"\ndone — outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
