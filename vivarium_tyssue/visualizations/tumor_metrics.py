"""Mesh-derived metric plots for the tumor studies (Path C: read runs.db).

These complement the cell-type snapshots/animation by quantifying the geometry the
snapshots only show qualitatively — the two things the 2026-06-17 review asked us to
demonstrate:

  * ``TumorCloneGrowth`` — the seeded focus is a SINGLE patch that grows OUTWARD: it
    plots the tumor cell count against the clone's spatial spread radius (mean distance
    of tumor faces from their own centroid). One tight, steadily growing radius ⇒ one
    expanding focus (a scatter of foci would show a large, jumpy radius from step one).

  * ``CellAreaOverTime`` — the mesh stays RELAXED (no overlap): it plots the median
    face area per cell type plus the 5th-percentile area of the whole sheet against the
    prefered area (1.0) and the division critical area (2.0). Overlap shows up as faces
    collapsing toward zero area; here the smallest faces stay well above zero.

Both read the per-step tyssue datasets the runner stores in runs.db (decimated), via the
shared ``_load_frames`` loader, so they need no extra emitted observables.
"""
from __future__ import annotations

import io

from pbg_superpowers.visualization import Visualization
from vivarium_tyssue.visualizations.tissue_gif import (
    _load_frames, _empty_html, _embed_figure, CELL_TYPE_COLORS,
)

_TUMOR = CELL_TYPE_COLORS.get("tumor", "#c0392b")
_HEALTHY = CELL_TYPE_COLORS.get("healthy", "#4a90d9")
_SPREAD = "#e67e22"   # clone-radius accent (orange)
_SMALL = "#7f8c8d"    # 5th-percentile area (grey)


def _cols(face: dict, *names):
    """Return aligned python lists for the requested face_df columns (empty-safe)."""
    return tuple(list(face.get(n) or []) for n in names)


def _frame_series(frames: list) -> dict:
    """Per-frame metrics over the loaded mesh frames.

    Returns parallel lists keyed by metric. ``clone_radius`` is the mean distance of
    tumor faces from the tumor centroid (an effective focus radius); it is None when
    there are no tumor cells yet so the line starts when the focus appears.
    """
    import numpy as np

    out = {k: [] for k in ("t", "n_tumor", "n_total", "clone_radius",
                           "healthy_med_area", "tumor_med_area", "p05_area")}
    for fr in frames:
        face = fr.get("datasets", {}).get("face_df") or {}
        xs, ys, areas, ctypes = _cols(face, "x", "y", "area", "cell_type")
        if not ctypes:
            continue
        xs = np.asarray(xs, float); ys = np.asarray(ys, float)
        areas = np.asarray(areas, float); ctypes = np.asarray([str(c) for c in ctypes])
        tmask = ctypes == "tumor"; hmask = ctypes == "healthy"
        out["t"].append(float(fr.get("t", 0.0)))
        out["n_tumor"].append(int(tmask.sum()))
        out["n_total"].append(int(len(ctypes)))
        if tmask.sum() >= 1:
            tx, ty = xs[tmask], ys[tmask]
            cx, cy = tx.mean(), ty.mean()
            out["clone_radius"].append(float(np.hypot(tx - cx, ty - cy).mean()))
        else:
            out["clone_radius"].append(None)
        finite = areas[np.isfinite(areas)]
        out["tumor_med_area"].append(float(np.median(areas[tmask])) if tmask.sum() else None)
        out["healthy_med_area"].append(float(np.median(areas[hmask])) if hmask.sum() else None)
        out["p05_area"].append(float(np.percentile(finite, 5)) if finite.size else None)
    return out


def _fig_to_html(fig, title: str) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return _embed_figure(buf.getvalue(), "image/png", title)


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class TumorCloneGrowth(Visualization):
    """Tumor cell count vs the clone's spatial spread radius over time."""

    config_schema = {
        **Visualization.config_schema,
        "_runs_db_path": {"_type": "string", "_default": ""},
    }

    def inputs(self) -> dict:
        return {}

    def _render_html(self) -> str:
        cfg = getattr(self, "config", None) or {}
        source = (cfg.get("sources") or [None])[0]
        frames = _load_frames(cfg.get("_runs_db_path") or "", source)
        m = _frame_series(frames)
        if not m["t"]:
            return _empty_html("No clone-growth data: runs.db has no tissue datasets.")
        plt = _setup()
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(m["t"], m["n_tumor"], color=_TUMOR, lw=2.2, label="tumor cells")
        ax1.set_xlabel("time"); ax1.set_ylabel("tumor cells", color=_TUMOR)
        ax1.tick_params(axis="y", labelcolor=_TUMOR)
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        # plot the radius only where defined (focus exists)
        ts = [t for t, r in zip(m["t"], m["clone_radius"]) if r is not None]
        rs = [r for r in m["clone_radius"] if r is not None]
        ax2.plot(ts, rs, color=_SPREAD, lw=2.0, ls="--", label="clone spread radius")
        ax2.set_ylabel("clone spread radius (mean dist. from focus centroid)", color=_SPREAD)
        ax2.tick_params(axis="y", labelcolor=_SPREAD)
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9, framealpha=0.9)
        ax1.set_title(cfg.get("title") or "Tumor clone growth")
        return _fig_to_html(fig, cfg.get("title") or "Tumor clone growth")

    def render(self) -> str:
        return self._render_html()

    def update(self, state: dict) -> dict:
        return {"html": self._render_html()}


class CellAreaOverTime(Visualization):
    """Median face area per cell type + the sheet's 5th-percentile area over time,
    against the prefered area and the division critical area — evidence the mesh stays
    relaxed (cells don't collapse into overlaps)."""

    config_schema = {
        **Visualization.config_schema,
        "prefered_area": {"_type": "float", "_default": 1.0},
        "division_crit": {"_type": "float", "_default": 2.0},
        "_runs_db_path": {"_type": "string", "_default": ""},
    }

    def inputs(self) -> dict:
        return {}

    def _render_html(self) -> str:
        cfg = getattr(self, "config", None) or {}
        source = (cfg.get("sources") or [None])[0]
        frames = _load_frames(cfg.get("_runs_db_path") or "", source)
        m = _frame_series(frames)
        if not m["t"]:
            return _empty_html("No cell-area data: runs.db has no tissue datasets.")
        plt = _setup()
        fig, ax = plt.subplots(figsize=(7, 4))

        def _pairs(key):
            return ([t for t, v in zip(m["t"], m[key]) if v is not None],
                    [v for v in m[key] if v is not None])

        th, vh = _pairs("healthy_med_area")
        tt, vt = _pairs("tumor_med_area")
        tp, vp = _pairs("p05_area")
        ax.plot(th, vh, color=_HEALTHY, lw=2.0, label="healthy median area")
        ax.plot(tt, vt, color=_TUMOR, lw=2.0, label="tumor median area")
        ax.plot(tp, vp, color=_SMALL, lw=1.6, ls=":", label="5th-percentile area (smallest cells)")
        pa = float(cfg.get("prefered_area") or 1.0)
        dc = float(cfg.get("division_crit") or 2.0)
        ax.axhline(pa, color="#444", lw=1.0, ls="--", alpha=0.7)
        ax.axhline(dc, color="#888", lw=1.0, ls="--", alpha=0.6)
        ax.text(m["t"][-1], pa, " prefered area", va="bottom", ha="right", fontsize=8, color="#444")
        ax.text(m["t"][-1], dc, " division crit_area", va="bottom", ha="right", fontsize=8, color="#888")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("time"); ax.set_ylabel("face area")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_title(cfg.get("title") or "Cell-area distribution over time")
        return _fig_to_html(fig, cfg.get("title") or "Cell-area over time")

    def render(self) -> str:
        return self._render_html()

    def update(self, state: dict) -> dict:
        return {"html": self._render_html()}
