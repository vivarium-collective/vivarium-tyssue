"""Tissue animation Visualization Steps for pbg-tyssue.

These replace the ``create_gif`` / ``create_gif_3d`` calls that used to live in
``tests/tests.py`` ``__main__``. Each scenario there drew the evolving tyssue
``History`` to a GIF; here that rendering is a first-class, dashboard-discoverable
``Visualization`` Step instead of a one-off script.

Render path: **Path C** (see docs/conventions/visualizations.md) — ``inputs()``
returns ``{}`` and the Step reads the study's ``runs.db`` directly. The per-step
emitted state carries the full tyssue datasets (vert/edge/face dataframes), which
the typed-wire path (B) would truncate, so we read the raw state JSON ourselves.

Two render strategies, in order:
  1. **Faithful** — reconstruct a tyssue ``History`` from the emitted datasets and
     call tyssue's own ``create_gif`` / ``create_gif_3d``, embedding the resulting
     animation as a base64 GIF. Pixel-faithful to the original repo's output.
  2. **Fallback** — if tyssue is unavailable or reconstruction fails, animate the
     edge mesh directly from the emitted ``edge_df`` source/target coordinates with
     matplotlib (+ Pillow). Dependency-light; still shows the tissue evolving.

All heavy imports (tyssue, matplotlib) are deferred into render time so package
discovery / ``allocate_core()`` never needs tyssue installed.
"""
from __future__ import annotations

import base64
import io
import json
import sqlite3
from pathlib import Path

from pbg_superpowers.visualization import Visualization

# Crypt cell-type palette (inlined from pbg_tyssue.draw.kwd_functions so this
# module doesn't import the tyssue-dependent draw package at discovery time).
CELL_TYPE_COLORS = {
    "sc": "#DE8968",
    "pc": "#69E0C3",
    "ent": "#C45454",
    "gc": "#45B53E",
    "extruding": "#000000",
    "dividing": "#feeda3",
}


# --------------------------------------------------------------------------
# runs.db reading (Path C) — shared by both viz classes.
# --------------------------------------------------------------------------
def _load_frames(runs_db_path: str | None) -> list[dict]:
    """Return [{t: float, datasets: {vert_df, edge_df, face_df, cell_df}}] for the
    most recent run in runs.db. Empty list when unavailable.
    """
    if not runs_db_path:
        return []
    p = Path(runs_db_path)
    if not p.is_file():
        return []
    try:
        conn = sqlite3.connect(str(p))
    except sqlite3.Error:
        return []
    conn.row_factory = sqlite3.Row
    try:
        has_history = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='history'"
        ).fetchone()
        if not has_history:
            return []
        # Pick the most recent run if a run_id column exists; otherwise take all.
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(history)").fetchall()}
        if "run_id" in cols:
            last = conn.execute(
                "SELECT run_id FROM history ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            rows = conn.execute(
                "SELECT global_time, state FROM history WHERE run_id=? ORDER BY step ASC",
                (last["run_id"],),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT global_time, state FROM history ORDER BY step ASC"
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    frames: list[dict] = []
    for r in rows:
        try:
            state = json.loads(r["state"]) if r["state"] else {}
        except (json.JSONDecodeError, TypeError):
            continue
        ds = _extract_datasets(state)
        if ds:
            frames.append({"t": r["global_time"], "datasets": ds})
    return frames


def _extract_datasets(state: dict) -> dict:
    """Find the tyssue datasets store within an emitted state dict.

    The emitter may nest the datasets under a store name ("Datasets"/"datasets")
    or splat the dataframes at the top level. Handle all three shapes.
    """
    df_keys = ("vert_df", "edge_df", "face_df", "cell_df")
    for store in ("Datasets", "datasets"):
        node = state.get(store)
        if isinstance(node, dict) and any(k in node for k in df_keys):
            return {k: node[k] for k in df_keys if k in node and node[k]}
    if any(k in state for k in df_keys):
        return {k: state[k] for k in df_keys if k in state and state[k]}
    # Search one level down for any dict carrying the dataframe keys.
    for v in state.values():
        if isinstance(v, dict) and any(k in v for k in df_keys):
            return {k: v[k] for k in df_keys if k in v and v[k]}
    return {}


def _subsample(frames: list[dict], num_frames: int) -> list[dict]:
    if num_frames <= 0 or len(frames) <= num_frames:
        return frames
    step = len(frames) / float(num_frames)
    return [frames[int(i * step)] for i in range(num_frames)]


def _empty_html(msg: str) -> str:
    return (
        '<div style="padding:1rem;font-family:system-ui;color:#555;'
        'border:1px dashed #bbb;border-radius:6px">'
        f"<strong>No animation:</strong> {msg}</div>"
    )


def _gif_html(gif_bytes: bytes, caption: str, div_id: str) -> str:
    b64 = base64.b64encode(gif_bytes).decode("ascii")
    return (
        f'<figure id="{div_id}" style="margin:0;text-align:center">'
        f'<img alt="{caption}" style="max-width:100%;height:auto" '
        f'src="data:image/gif;base64,{b64}"/>'
        f'<figcaption style="font-family:system-ui;font-size:0.85rem;color:#666;'
        f'margin-top:0.4rem">{caption}</figcaption></figure>'
    )


# --------------------------------------------------------------------------
# tyssue History reconstruction (faithful path).
# --------------------------------------------------------------------------
def _build_sheet(datasets: dict):
    """Reconstruct a tyssue Sheet from emitted {vert_df, edge_df, face_df, cell_df}."""
    import pandas as pd
    from tyssue import Sheet

    mapping = {"vert": "vert_df", "edge": "edge_df", "face": "face_df", "cell": "cell_df"}
    ds = {}
    for short, col in mapping.items():
        block = datasets.get(col)
        if block:
            ds[short] = pd.DataFrame(block)
    return Sheet("reconstructed", ds)


def _build_history(frames: list[dict]):
    """Reconstruct a tyssue History by replaying the emitted datasets frame by frame."""
    from tyssue.core.history import History

    sheet = _build_sheet(frames[0]["datasets"])
    history = History(sheet)
    for fr in frames:
        snap = _build_sheet(fr["datasets"])
        for short in ("vert", "edge", "face", "cell"):
            if short in snap.datasets:
                sheet.datasets[short] = snap.datasets[short]
        try:
            history.record(time_stamp=float(fr["t"]))
        except TypeError:
            history.record()
    return history, sheet


# --------------------------------------------------------------------------
# matplotlib fallback (no tyssue) — animate the edge mesh from source/target coords.
# --------------------------------------------------------------------------
def _matplotlib_gif(frames: list[dict], coords: list[str], *, edge_color: str,
                    face_color: str, alpha: float, threed: bool) -> bytes | None:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Collect per-frame edge segments from s<axis>/t<axis> columns.
    segs_per_frame = []
    for fr in frames:
        edge = fr["datasets"].get("edge_df")
        if not edge:
            continue
        df = pd.DataFrame(edge)
        scols = [f"s{a}" for a in coords]
        tcols = [f"t{a}" for a in coords]
        if not all(c in df.columns for c in scols + tcols):
            continue
        s = df[scols].to_numpy()
        t = df[tcols].to_numpy()
        segs_per_frame.append((s, t))
    if not segs_per_frame:
        return None

    import numpy as np
    allpts = np.vstack([np.vstack([s, t]) for s, t in segs_per_frame])
    mins, maxs = allpts.min(axis=0), allpts.max(axis=0)
    pad = 0.05 * (maxs - mins + 1e-9)

    if threed and len(coords) == 3:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = plt.subplots(figsize=(5, 5))

    def draw(i):
        ax.clear()
        s, t = segs_per_frame[i]
        for k in range(len(s)):
            pts = list(zip(s[k], t[k]))  # ([x0,x1],[y0,y1],...)
            ax.plot(*pts, color=edge_color, linewidth=0.6, alpha=0.9)
        ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
        ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
        if threed and len(coords) == 3:
            ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
        ax.set_axis_off()
        ax.set_title(f"t = {frames[i]['t']:.2f}", fontsize=9)

    anim = FuncAnimation(fig, draw, frames=len(segs_per_frame), interval=100)
    # PillowWriter needs a real filename to infer the GIF format — a BytesIO has
    # no extension, so write to a temp .gif and read the bytes back.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        out_path = tmp.name
    try:
        anim.save(out_path, writer=PillowWriter(fps=10))
        data = Path(out_path).read_bytes()
    except Exception:
        data = None
    finally:
        plt.close(fig)
        try:
            Path(out_path).unlink()
        except OSError:
            pass
    return data


def _tyssue_gif(frames: list[dict], *, coords: list[str], num_frames: int,
                face_color: str, edge_color: str, alpha: float,
                threed: bool, crypt: bool, cull_back_edges: bool) -> bytes | None:
    """Faithful path: reconstruct History + call tyssue's create_gif/create_gif_3d."""
    import tempfile
    from tyssue import config as tyssue_config
    from tyssue.draw import create_gif

    history, sheet = _build_history(frames)
    draw_specs = tyssue_config.draw.sheet_spec()
    draw_specs["face"]["visible"] = True
    draw_specs["face"]["alpha"] = alpha
    draw_specs["face"]["color"] = face_color
    draw_specs["edge"]["color"] = edge_color

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        out_path = tmp.name
    try:
        if threed:
            from tyssue.draw import create_gif_3d  # may not exist in all versions
            kwargs = dict(output=out_path, num_frames=num_frames, coords=coords,
                          cull_back_edges=cull_back_edges, **draw_specs)
            if crypt:
                from pbg_tyssue.draw.kwd_functions import crypt_cell_type_kwds
                kwargs["dynamic_draw_kwds"] = [crypt_cell_type_kwds]
                kwargs["legend"] = CELL_TYPE_COLORS
            create_gif_3d(history, **kwargs)
        else:
            create_gif(history, output=out_path, num_frames=num_frames,
                       coords=coords, **draw_specs)
        return Path(out_path).read_bytes()
    finally:
        try:
            Path(out_path).unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------
# Visualization classes.
# --------------------------------------------------------------------------
class TissueSheetGif(Visualization):
    """2D animation of a tyssue sheet evolving over the simulation.

    Reconstructs the run's tyssue History from runs.db and renders it with
    tyssue's ``create_gif`` (faithful), falling back to a matplotlib edge-mesh
    animation when tyssue isn't available. Drives the solver / regulation /
    stochastic / jamming / gradient / anisotropic scenarios — pick the
    projection plane with ``coords`` (``["x","z"]`` for the cylinder,
    ``["x","y"]`` for the flat sheet).
    """

    config_schema = {
        **Visualization.config_schema,
        "coords": {"_type": "list[string]", "_default": ["x", "z"]},
        "num_frames": {"_type": "integer", "_default": 60},
        "face_color": {"_type": "string", "_default": "#4a90d9"},
        "edge_color": {"_type": "string", "_default": "black"},
        "alpha": {"_type": "float", "_default": 0.6},
        "_runs_db_path": {"_type": "string", "_default": ""},
    }

    def inputs(self) -> dict:
        return {}

    def _render_html(self) -> str:
        cfg = getattr(self, "config", None) or {}
        frames = _load_frames(cfg.get("_runs_db_path") or "")
        if not frames:
            return _empty_html(
                "no tissue datasets found in runs.db (the run must emit the "
                "Datasets store: vert_df / edge_df / face_df)."
            )
        coords = list(cfg.get("coords") or ["x", "z"])
        num_frames = int(cfg.get("num_frames") or 60)
        face_color = cfg.get("face_color") or "#4a90d9"
        edge_color = cfg.get("edge_color") or "black"
        alpha = float(cfg.get("alpha") or 0.6)
        title = cfg.get("title") or "tyssue sheet"
        div_id = self.stable_div_id(title)

        sampled = _subsample(frames, num_frames)
        # 1. Faithful tyssue render.
        try:
            gif = _tyssue_gif(sampled, coords=coords, num_frames=num_frames,
                              face_color=face_color, edge_color=edge_color,
                              alpha=alpha, threed=False, crypt=False,
                              cull_back_edges=False)
            if gif:
                return _gif_html(gif, f"{title} — {len(frames)} steps (tyssue create_gif)", div_id)
        except Exception as exc:  # noqa: BLE001 — fall through to matplotlib
            fallback_note = f"tyssue render unavailable ({type(exc).__name__}); using matplotlib mesh."
        else:
            fallback_note = "tyssue render returned no frames; using matplotlib mesh."
        # 2. matplotlib fallback.
        gif = _matplotlib_gif(sampled, coords, edge_color=edge_color,
                              face_color=face_color, alpha=alpha, threed=False)
        if gif:
            return _gif_html(gif, f"{title} — {len(frames)} steps (mesh). {fallback_note}", div_id)
        return _empty_html(f"could not render animation. {fallback_note}")

    def render(self) -> str:
        return self._render_html()

    def update(self, state: dict) -> dict:
        # Path C: data comes from runs.db, not the per-tick wire. Render directly.
        return {"html": self._render_html()}


class TissueCryptGif3D(Visualization):
    """3D animation of the intestinal-crypt vessel, cells colored by type.

    The gillespie scenario's renderer: reconstructs the History and calls
    tyssue's ``create_gif_3d`` with the crypt cell-type coloring
    (``crypt_cell_type_kwds`` + ``CELL_TYPE_COLORS``), falling back to a 3D
    matplotlib edge-mesh animation when tyssue isn't available.
    """

    config_schema = {
        **Visualization.config_schema,
        "coords": {"_type": "list[string]", "_default": ["x", "y", "z"]},
        "num_frames": {"_type": "integer", "_default": 100},
        "cull_back_edges": {"_type": "boolean", "_default": True},
        "edge_color": {"_type": "string", "_default": "black"},
        "_runs_db_path": {"_type": "string", "_default": ""},
    }

    def inputs(self) -> dict:
        return {}

    def _render_html(self) -> str:
        cfg = getattr(self, "config", None) or {}
        frames = _load_frames(cfg.get("_runs_db_path") or "")
        if not frames:
            return _empty_html(
                "no tissue datasets found in runs.db (the gillespie run must "
                "emit the Datasets store with a cell_type column on face_df)."
            )
        coords = list(cfg.get("coords") or ["x", "y", "z"])
        num_frames = int(cfg.get("num_frames") or 100)
        cull = bool(cfg.get("cull_back_edges", True))
        edge_color = cfg.get("edge_color") or "black"
        title = cfg.get("title") or "crypt (3D)"
        div_id = self.stable_div_id(title)

        sampled = _subsample(frames, num_frames)
        try:
            gif = _tyssue_gif(sampled, coords=coords, num_frames=num_frames,
                              face_color="#cccccc", edge_color=edge_color,
                              alpha=1.0, threed=True, crypt=True,
                              cull_back_edges=cull)
            if gif:
                return _gif_html(gif, f"{title} — {len(frames)} steps (tyssue create_gif_3d)", div_id)
        except Exception as exc:  # noqa: BLE001
            fallback_note = f"tyssue 3D render unavailable ({type(exc).__name__}); using matplotlib mesh."
        else:
            fallback_note = "tyssue 3D render returned no frames; using matplotlib mesh."
        gif = _matplotlib_gif(sampled, coords, edge_color=edge_color,
                              face_color="#cccccc", alpha=1.0, threed=True)
        if gif:
            return _gif_html(gif, f"{title} — {len(frames)} steps (3D mesh). {fallback_note}", div_id)
        return _empty_html(f"could not render 3D animation. {fallback_note}")

    def render(self) -> str:
        return self._render_html()

    def update(self, state: dict) -> dict:
        return {"html": self._render_html()}
