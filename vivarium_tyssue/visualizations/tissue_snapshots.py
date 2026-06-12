"""TissueSheetSnapshots — multi-panel static figure of the sheet over time,
faces colored by cell_type. Path C: reads runs.db (shares the loader with
tissue_gif). Renders filled face polygons by grouping edge_df rows per face."""
from __future__ import annotations

import base64
import io
from pathlib import Path

from pbg_superpowers.visualization import Visualization
from vivarium_tyssue.visualizations.tissue_gif import (
    _load_frames, _subsample, _empty_html, CELL_TYPE_COLORS,
)

_DEFAULT_COLOR = "#9aa0a6"


def _cell_type_color(cell_type: str) -> str:
    return CELL_TYPE_COLORS.get(str(cell_type), _DEFAULT_COLOR)


def _face_polygons(datasets: dict, coords: list[str]):
    """Yield (face_uid, cell_type, [(x,y), ...]) for each face from edge_df.

    Uses s<axis>/t<axis> edge endpoint columns grouped by edge_df['face'].
    """
    import pandas as pd
    face = datasets.get("face_df") or {}
    edge = datasets.get("edge_df") or {}
    if not face or not edge:
        return
    fdf = pd.DataFrame(face)
    edf = pd.DataFrame(edge)
    a, b = coords[0], coords[1]
    needed = {"face", f"s{a}", f"s{b}", f"t{a}", f"t{b}"}
    if not needed <= set(edf.columns):
        return
    type_by_idx = dict(zip(range(len(fdf)), fdf.get("cell_type", ["?"] * len(fdf))))
    uid_by_idx = dict(zip(range(len(fdf)), fdf.get("unique_id", list(range(len(fdf))))))
    for face_idx, grp in edf.groupby("face"):
        pts = list(zip(grp[f"s{a}"].tolist(), grp[f"s{b}"].tolist()))
        idx = int(face_idx)
        yield uid_by_idx.get(idx, idx), type_by_idx.get(idx, "?"), pts


def _panels_html(frames: list, coords: list[str], n_panels: int, title: str) -> str:
    if not frames:
        return _empty_html("No snapshots: runs.db has no tissue datasets.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    sampled = _subsample(frames, n_panels)
    ncols = min(len(sampled), 4) or 1
    nrows = (len(sampled) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for i, fr in enumerate(sampled):
        ax = axes[i // ncols][i % ncols]
        any_poly = False
        for _uid, ctype, pts in _face_polygons(fr["datasets"], coords):
            if len(pts) >= 3:
                ax.add_patch(Polygon(pts, closed=True, facecolor=_cell_type_color(ctype),
                                     edgecolor="black", linewidth=0.4, alpha=0.85))
                any_poly = True
        if any_poly:
            ax.autoscale_view()
            ax.set_aspect("equal")
        ax.set_title(f"t = {fr['t']:.2f}", fontsize=9)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (f'<figure style="margin:0;text-align:center"><img alt="{title}" '
            f'style="max-width:100%;height:auto" src="data:image/png;base64,{b64}"/>'
            f'<figcaption style="font-family:system-ui;font-size:0.85rem;color:#666">'
            f'{title} — {len(frames)} steps, {len(sampled)} panels</figcaption></figure>')


class TissueSheetSnapshots(Visualization):
    """Multi-panel snapshots of the sheet over time, faces colored by cell_type."""

    config_schema = {
        **Visualization.config_schema,
        "coords": {"_type": "list[string]", "_default": ["x", "y"]},
        "n_panels": {"_type": "integer", "_default": 6},
        "_runs_db_path": {"_type": "string", "_default": ""},
    }

    def inputs(self) -> dict:
        return {}

    def _render_html(self) -> str:
        cfg = getattr(self, "config", None) or {}
        frames = _load_frames(cfg.get("_runs_db_path") or "")
        return _panels_html(frames, list(cfg.get("coords") or ["x", "y"]),
                            int(cfg.get("n_panels") or 6), cfg.get("title") or "tissue snapshots")

    def render(self) -> str:
        return self._render_html()

    def update(self, state: dict) -> dict:
        return {"html": self._render_html()}
