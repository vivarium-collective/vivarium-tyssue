"""Generate tumor_coupling_3d_viz.ipynb — an ipyvolume visualization of the saved
3D tumor history. Run: python make_viz_notebook.py  (writes the .ipynb beside it).

Kept as a generator so the notebook is reproducible and reviewable as plain text.
The notebook only LOADS outputs/monolayer_tumor_history.hf5 and renders the tissue
mesh with tyssue's `browse_history`; it never re-runs the simulation."""
import json
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / "tumor_coupling_3d_viz.ipynb"

md = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)

cells = []

cells.append(md(
r"""# 3D Tumor Coupling — tissue visualization (ipyvolume)

Interactive 3D view of the tumor grown in **`tumor_coupling_3d.py`** (a COPASI
population ODE driving a volumetric tyssue monolayer). This notebook **loads the
saved history** `outputs/monolayer_tumor_history.hf5` and draws the tissue's
**apical surface** with tyssue's `browse_history` — a single-layer sheet (one
polygon per cell) so the coloured faces and the wireframe cell outlines coincide.
Each cell is coloured by `cell_type`: **tumor cells stand out in red** against the
blue healthy epithelium. Drag the slider to move through time and watch the seeded
focus divide into a 3D mass — the randomized division-plane orientations push it
out of the starting plane, buckling the surface in *z*.

> This notebook does **not** run the simulation — run `python tumor_coupling_3d.py`
> first (from this directory) to produce the history file.

Requires `ipyvolume` (already in the `vivarium-tyssue` env). In JupyterLab you may
need the widgets extension enabled for the 3D canvas to appear."""
))

cells.append(code(
r"""import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

from tyssue.core.history import HistoryHdf5
from tyssue.draw import browse_history

HISTORY = Path("outputs/monolayer_tumor_history.hf5")
assert HISTORY.exists(), (
    f"{HISTORY} not found — run `python tumor_coupling_3d.py` in this directory first."
)

# Same cell-type palette as the experiment. The ORDER defines the integer code each
# type maps to; browse_history colours faces through a matplotlib colormap, so we
# register the palette as a discrete ListedColormap and colour by that code.
TYPES = ["healthy", "tumor", "stem", "dead", "dividing", "extruding"]
CELL_TYPE_COLORS = {
    "healthy":  "#4a90d9",   # blue  — normal epithelium
    "tumor":    "#c0392b",   # red   — tumor cells
    "stem":     "#8e44ad",   # purple
    "dead":     "#2b2b2b",   # near-black — necrotic
    "dividing": "#feeda3",   # yellow — mid-division
    "extruding":"#000000",   # black  — mid-death
}
_CMAP_NAME = "tumor_celltypes"
try:
    matplotlib.colormaps.register(
        ListedColormap([mcolors.to_rgb(CELL_TYPE_COLORS[t]) for t in TYPES], name=_CMAP_NAME))
except ValueError:
    pass  # already registered on a re-run"""
))

cells.append(md(
r"""## 1. Reopen the saved history

`HistoryHdf5.from_archive` reloads the archived apical surface so `browse_history`
can retrieve and redraw any frame. The experiment stores the render columns the
apical sheet needs (vertex + face positions, edge topology + sub-coordinates) and
stamps each face with its cell's `cell_type`, at ~60 evenly-spaced timesteps."""
))

cells.append(code(
r"""history = HistoryHdf5.from_archive(str(HISTORY))
times = history.time_stamps
print(f"{len(times)} frames, t = {times.min():.1f} .. {times.max():.1f}")
last = history.retrieve(times[-1])
print(f"final frame: {last.Nf} apical faces (one per cell), {last.Ne} edges")
print("cell types present:", sorted(last.face_df['cell_type'].astype(str).unique()))"""
))

cells.append(md(
r"""## 2. Colour faces by cell type

`browse_history` accepts a **callable** face colour — re-evaluated on each retrieved
frame — so the colouring follows cells as they divide and change type. Each apical
face already carries its cell's `cell_type`; we turn that into the integer code the
registered `ListedColormap` maps to a palette colour."""
))

cells.append(code(
r"""_CODE = {t: i for i, t in enumerate(TYPES)}

def face_type_code(sheet):
    # Per-face cell_type code (float array of length Nf) for the colormap.
    ftype = sheet.face_df["cell_type"].astype(str).to_numpy()
    return np.array([_CODE.get(t, 0) for t in ftype], dtype=float)"""
))

cells.append(md(
r"""## 3. Browse the 3D tissue

The slider scrubs through time. Each cell's apical face is filled by its type
(**red = tumor**), with a faint grey wireframe for the cell outlines — because this
is a single-layer sheet, the fills and outlines line up exactly. Drag to rotate.
The tumor begins as a small central red focus and builds outward — and, via the
randomized horizontal divisions, upward in *z* — into a 3D mass."""
))

cells.append(code(
r"""browse_history(
    history,
    coords=["x", "y", "z"],
    edge={"visible": True, "color": "#666666"},
    face={
        "visible": True,
        "color": face_type_code,
        "colormap": _CMAP_NAME,
        "color_range": (0, len(TYPES) - 1),
    },
)"""
))

cells.append(md(
r"""## 4. Tumor growth curve

A quick non-3D sanity check from the same history: the tumor-lineage cell count
(tumor + the transient *dividing* cells, which are committed tumor cells) over
time as the coupled COPASI fluxes drive divisions."""
))

cells.append(code(
r"""import pandas as pd
import matplotlib.pyplot as plt

cell = history.retrieve_columns("cell", ["time", "cell_type"])
counts = (cell.groupby(["time", "cell_type"]).size().unstack(fill_value=0)
          .reindex(sorted(cell["time"].unique())))
tumor_lineage = counts.get("tumor", 0) + counts.get("dividing", 0)
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(counts.index, tumor_lineage, color=CELL_TYPE_COLORS["tumor"], lw=2,
        label="tumor (incl. dividing)")
if "healthy" in counts:
    ax.plot(counts.index, counts["healthy"], color=CELL_TYPE_COLORS["healthy"], lw=1.6, label="healthy")
if "dead" in counts and counts["dead"].max() > 0:
    ax.plot(counts.index, counts["dead"], color=CELL_TYPE_COLORS["dead"], lw=1.2, label="dead")
ax.set_xlabel("time"); ax.set_ylabel("cell count")
ax.set_title("3D tumor growth over time"); ax.legend(frameon=False); ax.grid(alpha=0.25)
plt.show()"""
))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (vivarium-tyssue)", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"},
}
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB_PATH}")
