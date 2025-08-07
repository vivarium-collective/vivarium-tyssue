import logging
import warnings

import numpy as np

from tyssue.topology.base_topology import add_vert
from tyssue.topology.sheet_topology import face_division, get_division_edges

logger = logging.getLogger(name=__name__)
MAX_ITER = 100

def cell_division(sheet, mother, geom, angle=None):
    """Causes a cell to divide

    Parameters
    ----------

    sheet : a 'Sheet' instance
    mother : face index of target dividing cell
    geom : a 2D geometry
    angle : division angle for newly formed edge

    Returns
    -------
    daughter: face index of new cell

    Notes
    -----
    - Function checks for perodic boundaries if there are, it checks if dividing cell
      rests on an edge of the periodic boundaries if so, it displaces the boundaries
      by a half a period and moves the target cell in the bulk of the tissue. It then
      performs cell division normally and reverts the periodic boundaries
      to the original configuration
    """

    if sheet.settings.get("boundaries") is not None:
        mother_on_periodic_boundary = False
        if (
            sheet.face_df.loc[mother]["at_x_boundary"]
            or sheet.face_df.loc[mother]["at_y_boundary"]
        ):
            mother_on_periodic_boundary = True
            saved_boundary = sheet.specs["settings"]["boundaries"].copy()
            for u, boundary in sheet.settings["boundaries"].items():
                if sheet.face_df.loc[mother][f"at_{u}_boundary"]:
                    period = boundary[1] - boundary[0]
                    sheet.specs["settings"]["boundaries"][u] = [
                        boundary[0] + period / 2.0,
                        boundary[1] + period / 2.0,
                    ]
            geom.update_all(sheet)

    if not sheet.face_df.loc[mother, "is_alive"]:
        logger.warning("Cell %s is not alive and cannot divide", mother)
        return
    edge_a, edge_b = get_division_edges(sheet, mother, geom, angle=angle, axis="x")
    if edge_a is None:
        return

    vert_a, *_ = add_vert(sheet, edge_a)
    vert_b, *_ = add_vert(sheet, edge_b)
    sheet.vert_df.index.name = "vert"
    daughter = face_division(sheet, mother, vert_a, vert_b)

    if sheet.settings.get("boundaries") is not None and mother_on_periodic_boundary:
        sheet.specs["settings"]["boundaries"] = saved_boundary
        geom.update_all(sheet)

    sheet.network_changed = True

    return daughter

def divide(sheet, division_rate, dt):
    stem_cells = sheet.face_df.loc[sheet.face_df["stem_cell"] == 1]
    n_stem = len(stem_cells)
    cell_ids = list(stem_cells["unique_id"])
    n_divisions = np.random.binomial(n=n_stem, p=division_rate * dt)
    dividing_cells = np.random.choice(cell_ids, size=n_divisions, replace=False)

    for cell_id in dividing_cells:
        cell_idx = int(sheet.face_df[sheet.face_df["unique_id"] == cell_id].index[0])
        daughter = cell_division(sheet, cell_idx, orientation='horizontal')
        sheet.face_df

