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
        "dividing": "#C71FE0",
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

def face_param_kwds(parameter, color_range, colormap="Reds", alpha=1.0):
    """Draw kwds that colour each face by a ``face_df`` column (e.g.
    ``prefered_perimeter``) on a light->dark single-hue ramp.

    ``color`` is a callable so tyssue re-evaluates it on every frame; the
    colourmap + ``color_range`` normalisation is baked into an (Nf, 4) RGBA array
    here rather than left to tyssue. This (a) fixes the value->colour mapping to a
    STATIC scale so changes over time are comparable, and (b) bypasses tyssue's
    ``_face_color_from_sequence`` guard that paints a *spatially uniform* field
    flat grey — important because some parameters (e.g. the jammed
    ``prefered_perimeter``) stay uniform across cells while changing over time.
    """
    cmap = plt.get_cmap(colormap)
    cmin, cmax = color_range

    def _face_colors(sheet):
        vals = sheet.face_df[parameter].to_numpy().astype(float)
        normed = np.clip((vals - cmin) / (cmax - cmin), 0.0, 1.0)
        return cmap(normed)

    return {
        "face": {
            "visible": True,
            "color": _face_colors,
            "alpha": alpha,
        }
    }

def migrating_cell_edge_kwds(
    highlight_color="cyan",
    highlight_alpha=0.5,
    base_color="black",
    base_alpha=0.8,
    marker_col="migration_strength",
    width=1.5,
):
    """Edge kwds that paint the migrating cell's edges ``highlight_color`` and all
    other edges ``base_color``.

    The migrating cell is whichever face has ``marker_col`` > 0 (robust to face
    re-indexing). Both the cell's own half-edges *and* their ``opposite``
    half-edges (which belong to the neighbouring faces) are highlighted, so the
    whole shared junction is coloured rather than only one side. Per-edge alpha is
    baked into the RGBA array so the highlighted cell can use a different alpha
    from the rest; ``alpha`` is therefore set to ``None`` so matplotlib honours the
    per-edge RGBA alpha instead of applying one scalar to the whole collection.
    ``color`` is a callable so it is re-evaluated every frame.
    """
    base_rgba = (*mcolors.to_rgb(base_color), base_alpha)
    hi_rgba = (*mcolors.to_rgb(highlight_color), highlight_alpha)

    def _edge_colors(sheet):
        colors = np.tile(np.array(base_rgba), (sheet.Ne, 1))
        if marker_col in sheet.face_df.columns:
            mask = sheet.upcast_face(sheet.face_df[marker_col]).to_numpy() > 0
            if "opposite" in sheet.edge_df.columns:
                # Add the opposite half-edges (the neighbour's side of each shared
                # junction) so both sides of the migrating cell's boundary colour.
                opp = sheet.edge_df["opposite"].to_numpy()
                opp_of_mig = opp[mask]
                opp_of_mig = opp_of_mig[opp_of_mig >= 0]
                opp_pos = sheet.edge_df.index.get_indexer(opp_of_mig)
                mask[opp_pos[opp_pos >= 0]] = True
            colors[mask] = hi_rgba
        return colors

    return {
        "edge": {
            "visible": True,
            "color": _edge_colors,
            "alpha": None,
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