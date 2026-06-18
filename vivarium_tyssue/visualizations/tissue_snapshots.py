"""TissueSheetSnapshots — multi-panel static figure of the sheet over time,
faces colored by cell_type. Path C: reads runs.db (shares the loader with
tissue_gif).

Primary render path reconstructs a tyssue ``Sheet`` from the emitted datasets and
draws each panel with tyssue's own ``sheet_view`` — the *same* function family the
animation uses, so the stills and the gif are visually consistent. ``sheet_view``
orders the vertices cyclically around each face and maps colors per-face by index,
which the old hand-rolled ``_face_polygons`` path did not: that grouped ``edge_df``
by face without ordering the loop (→ self-intersecting "triangular" polygons) and
mapped colors by positional row (→ wrong colors once topology ops add/remove faces).
The hand-rolled path is kept only as a dependency-light fallback when tyssue's draw
stack is unavailable."""
from __future__ import annotations

import io
from pathlib import Path

from pbg_superpowers.visualization import Visualization
from vivarium_tyssue.visualizations.tissue_gif import (
    _load_frames, _subsample, _empty_html, _build_sheet, _embed_figure, CELL_TYPE_COLORS,
)

_DEFAULT_COLOR = "#9aa0a6"


def _cell_type_color(cell_type: str) -> str:
    return CELL_TYPE_COLORS.get(str(cell_type), _DEFAULT_COLOR)


def _draw_sheet_view_panel(ax, datasets: dict, coords: list[str]) -> bool:
    """Draw one panel with tyssue's sheet_view (faces colored by cell_type).

    Returns True on success. Raises on any reconstruction/draw failure so the
    caller can fall back to the hand-rolled polygon path.
    """
    import numpy as np
    from matplotlib.colors import to_rgba
    from tyssue import SheetGeometry
    from tyssue.draw import sheet_view

    sheet = _build_sheet(datasets)
    SheetGeometry.update_all(sheet)
    ctypes = list(sheet.face_df.get("cell_type", ["?"] * sheet.Nf))
    colors = np.array([to_rgba(_cell_type_color(t)) for t in ctypes])
    sheet_view(
        sheet, coords=coords, ax=ax,
        face={"visible": True, "color": colors, "alpha": 0.9},
        edge={"visible": True, "color": "#333333", "width": 0.3, "alpha": 0.7},
    )
    ax.set_aspect("equal")
    return True


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


def _draw_polygon_panel(ax, datasets: dict, coords: list[str]) -> bool:
    """Fallback panel: hand-rolled face polygons (no tyssue). Less accurate —
    does not order the face loop — but dependency-light."""
    from matplotlib.patches import Polygon
    any_poly = False
    for _uid, ctype, pts in _face_polygons(datasets, coords):
        if len(pts) >= 3:
            ax.add_patch(Polygon(pts, closed=True, facecolor=_cell_type_color(ctype),
                                 edgecolor="black", linewidth=0.4, alpha=0.85))
            any_poly = True
    if any_poly:
        ax.autoscale_view()
        ax.set_aspect("equal")
    return any_poly


def _legend_html(present_types: set[str]) -> str:
    """A small inline swatch legend so reviewers can read the cell-type colors."""
    labels = {"healthy": "healthy epithelium", "tumor": "tumor", "stem": "cancer stem",
              "extruding": "extruding (apoptotic)", "dead": "dead"}
    items = []
    for t in ["healthy", "tumor", "stem", "extruding", "dead"]:
        if t not in present_types:
            continue
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin:0 0.6rem 0 0">'
            f'<span style="width:0.8rem;height:0.8rem;background:{_cell_type_color(t)};'
            f'border:1px solid #888;display:inline-block;margin-right:0.3rem"></span>'
            f'{labels.get(t, t)}</span>'
        )
    if not items:
        return ""
    return ('<div style="font-family:system-ui;font-size:0.8rem;color:#444;'
            'margin-top:0.3rem">' + "".join(items) + "</div>")


def _panels_html(frames: list, coords: list[str], n_panels: int, title: str) -> str:
    if not frames:
        return _empty_html("No snapshots: runs.db has no tissue datasets.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sampled = _subsample(frames, n_panels)
    ncols = min(len(sampled), 4) or 1
    nrows = (len(sampled) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    present_types: set[str] = set()
    used_fallback = False
    for i, fr in enumerate(sampled):
        ax = axes[i // ncols][i % ncols]
        try:
            _draw_sheet_view_panel(ax, fr["datasets"], coords)
        except Exception:  # noqa: BLE001 — degrade to the hand-rolled polygons
            used_fallback = True
            _draw_polygon_panel(ax, fr["datasets"], coords)
        ax.set_axis_off()
        face = fr["datasets"].get("face_df") or {}
        present_types.update(str(t) for t in (face.get("cell_type") or []))
        ax.set_title(f"t = {fr['t']:.2f}", fontsize=9)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    renderer = "matplotlib mesh (fallback)" if used_fallback else "tyssue sheet_view"
    caption = (f'{title} — {len(frames)} steps, {len(sampled)} panels '
               f'(faces colored by cell type, {renderer})')
    figcaption = (f'<figcaption style="font-family:system-ui;font-size:0.85rem;color:#666">'
                  f'{caption}</figcaption>')
    return _embed_figure(buf.getvalue(), "image/png", title,
                         caption_html=figcaption, extra_html=_legend_html(present_types))


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
        source = (cfg.get("sources") or [None])[0]
        frames = _load_frames(cfg.get("_runs_db_path") or "", source)
        return _panels_html(frames, list(cfg.get("coords") or ["x", "y"]),
                            int(cfg.get("n_panels") or 6), cfg.get("title") or "tissue snapshots")

    def render(self) -> str:
        return self._render_html()

    def update(self, state: dict) -> dict:
        return {"html": self._render_html()}
