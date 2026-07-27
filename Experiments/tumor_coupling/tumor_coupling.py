"""Tumor-coupling experiment — a COPASI population ODE drives tissue mechanics.

A non-spatial breast-cancer **population ODE** (BioModels ``BIOMD0000000903``,
integrated in COPASI) is coupled to a flat 2-D tyssue epithelial sheet
(``test_square.hf5``, ``SheetGeometry``) through the ``TumorCoupling`` process.
Each step the process reads the SBML model's per-reaction birth/death fluxes and
fires ``floor(flux * scale * dt)`` discrete vertex-model events on the mesh:

  * **births**  -> real ``cell_division`` (the cell splits, the mesh gains a face),
  * **deaths**  -> real ``apoptosis_extrusion`` (the cell shrinks and is removed),
  * **tumor induction** -> ``differentiation`` of a healthy/stem cell into a tumor cell.

A SINGLE tumor cell is seeded at the sheet centre and grows outward into one
contiguous clone as the coupled fluxes drive divisions, while healthy cells are
progressively displaced. This mirrors ``vivarium_tyssue/composites/
tumor.composite.yaml`` and the ``get_test_tumor_*`` helpers in ``tests/tests.py``.

Outputs (all under ``outputs/``, git-ignored):
  * ``tumor.gif``               — 2-D animation, faces coloured by cell_type
  * ``still_t*.png``            — evenly spaced stills (same renderer as the gif)
  * ``population_over_time.png`` — tumor vs healthy (and stem/dead) cell counts vs time
  * ``population_over_time.csv`` — the underlying per-frame counts
  * ``tumor_area_over_time.png`` — total tumor size (summed face area) vs time
  * ``tumor_area_over_time.csv`` — per-frame tumor / total area and area fraction
  * ``face_area_floor_over_time.png`` — min / 5th-percentile face area vs time
    (overlap diagnostic — the floors must stay above 0)
  * ``face_area_floor_over_time.csv`` — the underlying per-frame area floors

Run from the repo's ``vivarium-tyssue`` conda env (needs ImageMagick ``magick`` on
PATH for the GIF):

    conda activate vivarium-tyssue
    cd Experiments/tumor_coupling
    python tumor_coupling.py

The coupling advances by ``dt`` each step, so the tumor grows over ``TUMOR_TF / dt``
COPASI-driven updates (the default ~3-4 min run reaches a clear tumor takeover).
See the timescale-coupling note in README.md and ``calibrate_timescale.py``.
"""
from __future__ import annotations

import contextlib
import copy
import io
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
FLAT_DATASET = "test_square.hf5"            # flat epithelial sheet
SBML_MODEL = "BIOMD0000000903.xml"          # COPASI breast-cancer population ODE

COORDS_2D = ["x", "y"]
FIG_DPI = 300          # publication-quality raster resolution for stills / plots
GIF_DPI = 120          # animation (kept lower — many frames)
NUM_GIF_FRAMES = 120
N_STILLS = 6
SEED = 20260720

# Run length. TUMOR_TF is elapsed global (tyssue) time; the coupling steps at
# TUMOR_DT. The timescale coupling that stops divisions overrunning the sheet is
# GROWTH_RATE (each cell inflates gradually, see build_tumor_spec) plus the halved
# event SCALES (fewer, better-spaced divisions); the tumor then grows slowly, so
# TUMOR_TF is extended to 200 to reach a clear takeover. COPASI_TIME (alpha) = the
# tumor-model clock relative to tyssue time; kept at 1.0 (alpha<1 delays tumor
# induction and starves the seed). See calibrate_timescale.py for tau_mech.
TUMOR_TF, TUMOR_DT = 160.0, 0.01
COPASI_TIME = 1.0

# Face palette by cell_type (matches vivarium_tyssue.visualizations.tissue_gif).
CELL_TYPE_COLORS = {
    "healthy": "#4a90d9",    # blue — normal epithelium
    "tumor": "#c0392b",      # red — tumor cells
    "stem": "#8e44ad",       # purple — cancer stem cells
    "dead": "#2b2b2b",       # near-black — cells killed under SBML control
    "dividing": "#feeda3",   # yellow — mid-division (transient)
    "extruding": "#000000",  # black — mid-extrusion (transient)
}
# Cell types drawn in the population-over-time analysis, in legend order.
POP_TYPES = ["healthy", "tumor", "stem", "dead"]

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
# flux * scale * interval accumulates past 1.0. ALL scales halved (2026-07-20) —
# births AND deaths together, so the birth:death balance holds (the tumor still
# grows) while the discrete-event RATE halves: ~half as many, better-spaced
# divisions per unit time.
SCALES = {
    "tumor_births": 1.0e-6, "tumor_deaths": 4.0e-7,
    "healthy_births": 3.0e-8, "healthy_deaths": 6.0e-8,
    "stem_births": 3.0e-7, "stem_deaths": 1.0e-5,
}


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
        # threshold_length raised 0.03 -> 0.15 so T1 neighbour-swap reconnections
        # fire more readily, letting the crowded tumor core rearrange instead of
        # locking into a tangle as the clone grows. This is a topology setting, not
        # the energy model — viscosity / elasticity / effectors are unchanged.
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
                "datasets": ["Datasets"],
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
            # Seed a compact central focus (6 cells, one contiguous patch). A single
            # seed stalls under gradual growth: while it is mid-division no free tumor
            # cell exists for the next flux-driven birth, so the clone dies out. 6
            # cells always leave an eligible cell, so the tumor grows steadily.
            "seed": {"tumor": 6, "stem": 0},
            # Real vertex-model topology (cell_division / remove_face); requires the
            # pandas-3-compatible forked tyssue.
            "topology_ops": True,
        },
        "inputs": {"datasets": ["Datasets"], "global_time": ["global_time"]},
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
# Drawing — 2-D flat sheet, faces by cell_type
# ---------------------------------------------------------------------------
def _face_color(sheet) -> np.ndarray:
    """(Nf, 4) RGBA per face, coloured by cell_type."""
    return np.array([
        mcolors.to_rgba(CELL_TYPE_COLORS.get(ct, "#cccccc"))
        for ct in sheet.face_df["cell_type"]
    ])


# Legend entries always shown, even on frames with none of that type present (so
# e.g. "dividing" stays in the key across the whole gif, not just on frames with an
# actively-dividing cell). Other types are added only if they appear during the run.
_ALWAYS_LEGEND = ("healthy", "tumor", "dividing")


def _legend_handles(types) -> list:
    """Legend patches for the given cell types, in the palette's canonical order."""
    keys = [k for k in CELL_TYPE_COLORS if k in set(types)]
    return [Patch(facecolor=CELL_TYPE_COLORS[k], edgecolor="#808080", label=k) for k in keys]


def _run_legend_types(history, times) -> list:
    """Cell types to key the whole animation on: the union of types seen across the
    run, plus the always-shown core — so the legend is identical on every frame."""
    seen = set(_ALWAYS_LEGEND)
    for t in times:
        seen.update(map(str, history.retrieve(t).face_df["cell_type"].unique()))
    return [k for k in CELL_TYPE_COLORS if k in seen]


def _frame_limits(history, times):
    """Fixed (min, max) per-axis limits from the first frame, with a 5% margin."""
    sheet0 = history.retrieve(times[0])
    bounds = sheet0.vert_df[COORDS_2D].describe().loc[["min", "max"]]
    margin = (bounds.loc["max"] - bounds.loc["min"]).max() * 0.05
    return {c: (bounds.loc["min", c] - margin, bounds.loc["max", c] + margin) for c in COORDS_2D}


def _draw_frame(sheet, title, lims, legend_types):
    """Draw one flat-sheet frame (faces by cell_type) into a Figure, or None if
    the frame can't be rendered. ``legend_types`` is the fixed run-wide legend."""
    from tyssue.draw import sheet_view
    try:
        fig, ax = plt.subplots(figsize=(6.6, 5.0))
        sheet_view(
            sheet, coords=COORDS_2D, ax=ax,
            face={"visible": True, "color": _face_color(sheet), "alpha": 1.0},
            edge={"visible": True, "color": "#808080", "width": 0.8},
        )
        ax.set_aspect("equal")
        ax.set_xlim(*lims["x"])
        ax.set_ylim(*lims["y"])
        ax.set_title(title, fontsize=10)
        ax.legend(handles=_legend_handles(legend_types),
                  loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    except Exception as exc:  # noqa: BLE001
        print(f"frame {title} failed ({type(exc).__name__}: {exc}); skipping")
        plt.close("all")
        return None
    return fig


def save_gif(history, out_path: Path):
    times = list(history.time_stamps)
    if not times:
        return
    lims = _frame_limits(history, times)
    idx = np.unique(np.round(np.linspace(0, len(times) - 1,
                                         min(NUM_GIF_FRAMES, len(times)))).astype(int))
    legend_types = _run_legend_types(history, [times[int(i)] for i in idx])
    tmp = Path(tempfile.mkdtemp())
    n = 0
    try:
        for i in idx:
            t = times[int(i)]
            fig = _draw_frame(history.retrieve(t), f"t = {float(t):.1f}", lims, legend_types)
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


def save_stills(history, out_dir: Path):
    times = list(history.time_stamps)
    if not times:
        return
    lims = _frame_limits(history, times)
    legend_types = _run_legend_types(history, times)
    for frac in np.linspace(0.0, 1.0, N_STILLS):
        t = times[int(round(frac * (len(times) - 1)))]
        fig = _draw_frame(history.retrieve(t), f"t = {float(t):.1f}", lims, legend_types)
        if fig is None:
            continue
        fig.savefig(out_dir / f"still_t{float(t):06.1f}.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Analysis — tumor vs healthy cell population over time
# ---------------------------------------------------------------------------
def population_over_time(history) -> pd.DataFrame:
    """Count of each cell type at every recorded timepoint (wide: time x types).

    Counts come from the mesh history rather than the coupling's scalar ``*_count``
    stores: those stores accumulate additively across steps (a process-bigraph
    map[float] quirk), so the mesh is the reliable source for instantaneous counts.
    """
    face = history.datasets["face"]
    df = face[face["is_alive"] > 0] if "is_alive" in face.columns else face
    counts = df.groupby(["time", "cell_type"]).size().unstack(fill_value=0)
    return counts.reindex(sorted(counts.index)).reset_index()


def plot_population_over_time(counts: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    types = [t for t in POP_TYPES if t in counts.columns]
    for t in types:
        ax.plot(counts["time"], counts[t].to_numpy(), "-", linewidth=1.9,
                color=CELL_TYPE_COLORS[t], label=t)
    # Transient states: a cell mid-division is labelled "dividing" and mid-extrusion
    # "extruding" until the topology op fires. Under the slower clock only ~1 cell is
    # in flight at a time, so these are thin reference lines (counts stay accurate).
    for t in ("dividing", "extruding"):
        if t in counts.columns and counts[t].to_numpy().max() > 0:
            ax.plot(counts["time"], counts[t].to_numpy(), ":", linewidth=1.0,
                    color=CELL_TYPE_COLORS[t], label=t, alpha=0.8)
    # Total living tissue size — grows as the tumor divides.
    total = counts[[c for c in counts.columns if c != "time"]].sum(axis=1)
    ax.plot(counts["time"], total.to_numpy(), "--", linewidth=1.3,
            color="#555555", label="total (all types)")
    ax.set_xlabel("time")
    ax.set_ylabel("cell count")
    ax.set_title("Tumor coupling: tumor vs healthy cell population over time", fontsize=11)
    ax.margins(x=0)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def face_area_floor_over_time(history) -> pd.DataFrame:
    """Smallest face areas at every recorded timepoint (long: time, min_area,
    p5_area, median_area). This is the overlap diagnostic: when divisions outrun
    mechanical relaxation, freshly-split faces collapse toward zero area, so the
    minimum / 5th-percentile face area dips toward 0. A well-separated timescale
    keeps these floors comfortably above 0."""
    face = history.datasets["face"]
    df = face[face["is_alive"] > 0] if "is_alive" in face.columns else face
    g = df.groupby("time")["area"]
    out = pd.DataFrame({
        "time": sorted(df["time"].unique()),
        "min_area": g.min().to_numpy(),
        "p5_area": g.quantile(0.05).to_numpy(),
        "median_area": g.median().to_numpy(),
    })
    return out


def plot_face_area_floor(area: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(area["time"], area["median_area"], "-", linewidth=1.6,
            color="#4a90d9", label="median face area")
    ax.plot(area["time"], area["p5_area"], "-", linewidth=1.8,
            color="#e67e22", label="5th-percentile face area")
    ax.plot(area["time"], area["min_area"], "-", linewidth=1.8,
            color="#c0392b", label="minimum face area")
    ax.axhline(0.0, color="#000000", linewidth=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("face area (mesh units²)")
    ax.set_title("Tumor coupling: smallest face areas over time (overlap check)", fontsize=11)
    ax.set_ylim(bottom=min(0.0, float(area["min_area"].min())))
    ax.margins(x=0)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def tumor_area_over_time(history) -> pd.DataFrame:
    """Total tumor size at every recorded timepoint: the summed face ``area`` of
    all tumor cells (long: time, tumor_area, total_area, tumor_area_fraction).

    This is a size/morphology signal complementary to the cell-count population:
    it captures both proliferation (more tumor faces) and per-cell growth (each
    face inflating toward its prefered area before it divides)."""
    face = history.datasets["face"]
    df = face[face["is_alive"] > 0] if "is_alive" in face.columns else face
    total = df.groupby("time")["area"].sum()
    tumor = (df[df["cell_type"] == "tumor"].groupby("time")["area"].sum()
             .reindex(total.index, fill_value=0.0))
    out = pd.DataFrame({
        "time": total.index,
        "tumor_area": tumor.to_numpy(),
        "total_area": total.to_numpy(),
    })
    out["tumor_area_fraction"] = out["tumor_area"] / out["total_area"].where(out["total_area"] > 0)
    return out.reset_index(drop=True)


def plot_tumor_area_over_time(area: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.fill_between(area["time"], area["tumor_area"], color=CELL_TYPE_COLORS["tumor"],
                    alpha=0.25, linewidth=0)
    ax.plot(area["time"], area["tumor_area"], "-", linewidth=2.0,
            color=CELL_TYPE_COLORS["tumor"], label="tumor area (Σ face area)")
    ax.plot(area["time"], area["total_area"], "--", linewidth=1.3,
            color="#555555", label="total tissue area")
    ax.set_xlabel("time")
    ax.set_ylabel("area (mesh units²)")
    ax.set_title("Tumor coupling: tumor size (area) over time", fontsize=11)
    ax.margins(x=0)
    ax.grid(True, alpha=0.25, linewidth=0.6)

    # Secondary axis: tumor area as a fraction of the whole sheet.
    ax2 = ax.twinx()
    ax2.plot(area["time"], 100.0 * area["tumor_area_fraction"], ":", linewidth=1.6,
             color="#8e44ad", label="tumor area fraction")
    ax2.set_ylabel("tumor area fraction (%)", color="#8e44ad")
    ax2.tick_params(axis="y", labelcolor="#8e44ad")
    ax2.set_ylim(bottom=0)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines],
              loc="upper left", bbox_to_anchor=(1.08, 1.0), frameon=False, fontsize=9)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
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

    save_gif(history, OUT_DIR / "tumor.gif")
    save_stills(history, OUT_DIR)

    counts = population_over_time(history)
    counts.to_csv(OUT_DIR / "population_over_time.csv", index=False)
    plot_population_over_time(counts, OUT_DIR / "population_over_time.png")

    area = tumor_area_over_time(history)
    area.to_csv(OUT_DIR / "tumor_area_over_time.csv", index=False)
    plot_tumor_area_over_time(area, OUT_DIR / "tumor_area_over_time.png")

    floor = face_area_floor_over_time(history)
    floor.to_csv(OUT_DIR / "face_area_floor_over_time.csv", index=False)
    plot_face_area_floor(floor, OUT_DIR / "face_area_floor_over_time.png")

    start, end = counts.iloc[0], counts.iloc[-1]
    fmt = lambda row: {t: int(row[t]) for t in POP_TYPES if t in counts.columns}
    print(f"[tumor] population start {fmt(start)} -> end {fmt(end)}", flush=True)
    print(f"[tumor] tumor area {area['tumor_area'].iloc[0]:.2f} -> "
          f"{area['tumor_area'].iloc[-1]:.2f} "
          f"({100 * area['tumor_area_fraction'].iloc[-1]:.1f}% of tissue)", flush=True)
    print(f"[tumor] smallest face area (overlap check): min {floor['min_area'].min():.3f}, "
          f"5th-pct floor {floor['p5_area'].min():.3f} (both should stay > 0)", flush=True)
    print(f"\ndone — outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
