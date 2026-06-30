import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from tyssue.io.hdf5 import load_datasets

CELL_TYPE_COLORS = {
        "sc": "#DE8968",
        "pc": "#69E0C3",
        "ent": "#C45454",
        "gc": "#45B53E",
        "extruding": "#000000",
        "dividing": "#feeda3"
    }

def cell_type_kwds(sheet, alpha=1.0):
    face_df = sheet.face_df
    cell_types = face_df["cell_type"].unique()

    if len(cell_types) <= 10:
        cmap = plt.cm.get_cmap('tab10')
        cell_type_int = {ct: i for i, ct in enumerate(cell_types)}
        cell_ints = face_df["cell_type"].map(cell_type_int).to_numpy()
        colors = cmap(cell_ints)
    else:
        cmap = plt.cm.get_cmap('tab20')
        cell_type_int = {ct: i for i, ct in enumerate(cell_types)}
        cell_ints = face_df["cell_type"].map(cell_type_int).to_numpy()
        colors = cmap(cell_ints)
    kwds = {
        "face": {
            "color": colors,
            "visible": True,
        }
    }
    return kwds

def crypt_cell_type_kwds(sheet, alpha=1.0):
    # Convert each face's cell_type to an RGBA row → shape (Nf, 4)
    # This bypasses the colormap normalisation path in tyssue entirely
    face_colors_rgba = np.array(
        [mcolors.to_rgba(CELL_TYPE_COLORS[ct]) for ct in sheet.face_df["cell_type"]]
    )

    kwds = {
        "face": {
            "color": face_colors_rgba,
        }
    }
    return kwds

def line_tension_edge_kwds(color_range=(-0.3, 0.3), colormap="coolwarm", width=1.5):
    """Draw kwds that colour each edge by its current ``line_tension``.

    ``color`` is a callable so tyssue's ``_parse_edge_specs`` re-evaluates it on
    every frame — the only way to get per-frame edge colours out of the 2-D
    ``create_gif`` (which, unlike ``create_gif_3d``, has no ``dynamic_draw_kwds``
    hook). ``color_range`` fixes the tension->colour mapping across all frames so a
    given colour always means the same tension.
    """
    return {
        "edge": {
            "visible": True,
            "color": lambda sheet: sheet.edge_df["line_tension"].to_numpy(),
            "colormap": colormap,
            "color_range": color_range,
            "width": width,
        }
    }

if __name__ == "__main__":
    from vivarium_tyssue.models.crypt_gillespie.crypt_params import spatial_prob, assign_cell_types
    from tyssue import Sheet

    sheet = Sheet("test", load_datasets("test_square.hf5"))
    sheet.face_df["cell_type"] = assign_cell_types(sheet, "y", spatial_prob_func=spatial_prob)
    kwds = cell_type_kwds(sheet)
    print(kwds)