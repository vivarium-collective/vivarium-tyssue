"""Vertex-deviation geometry — the columns ``SurfaceElasticity`` reads.

tyssue's ``SurfaceElasticity`` is a *vertex* effector: it penalises

    E = sum_i  k_i / 2 * (dev_i - d0_i)^2

where ``dev_i`` (``vert_df["dev_length"]``) is how far vertex *i* sits from the
centroid of its neighbouring vertices, and ``(dx, dy, dz)`` is the unit vector
along that offset. Pulling each vertex back toward its neighbours' centroid is
Laplacian ("umbrella") smoothing, so the term resists a single cell everting into
a spike while leaving the tissue's in-surface mechanics alone.

The effector reads those columns but does not compute them, and in the tyssue fork
only ``CylinderGeometryInit.update_all`` does — inside a cylinder-specific update
that also wants a boundary index, a z-axis distance and a cylinder-shaped lumen,
none of which apply to a closed sheet. :func:`update_vert_deviation` is the same
quantity computed here, and the mixin below folds it into any geometry's
``update_all``.

Two differences from the fork's version, both deliberate:

* it averages **every** neighbour rather than the first three the edge table
  happens to list, so the centroid is the actual one-ring centroid and does not
  depend on edge ordering (which a T1 changes);
* it is vectorised. The fork's version is a Python loop over vertices that
  re-materialises the whole edge table on each call — on a 400-vertex mesh over a
  2400-step run that alone costs more than the rest of the simulation.
"""
import numpy as np

from tyssue.geometry.sheet_geometry import ClosedSheetGeometry, SheetGeometry

DEVIATION_COLS = ["dx", "dy", "dz"]


def update_vert_deviation(eptm):
    """Write ``dev_length`` and the unit offset ``(dx, dy, dz)`` into ``vert_df``.

    ``dev_length`` is ``|r_i - mean(r_j for j in neighbours(i))|``; the unit columns
    point from the centroid toward the vertex. A vertex with no outgoing half-edge
    (there should be none on a closed sheet) gets zero deviation and a zero vector,
    so it contributes nothing to the energy or its gradient.
    """
    vert_df, edge_df = eptm.vert_df, eptm.edge_df
    n_verts = len(vert_df)
    coords = list(eptm.coords)

    if n_verts == 0 or len(edge_df) == 0:
        vert_df["dev_length"] = 0.0
        vert_df[DEVIATION_COLS[: len(coords)]] = 0.0
        return

    # Positional indices: srce/trgt carry vert_df *labels*, which are 0..Nv-1 only
    # after a reset_index. Map explicitly so a mid-run mesh works either way.
    positions = np.empty(int(vert_df.index.max()) + 1, dtype=np.int64)
    positions[vert_df.index.to_numpy()] = np.arange(n_verts)
    srce = positions[edge_df["srce"].to_numpy()]
    trgt = positions[edge_df["trgt"].to_numpy()]

    pos = vert_df[coords].to_numpy(dtype=float)
    neighbour_sum = np.zeros_like(pos)
    np.add.at(neighbour_sum, srce, pos[trgt])
    degree = np.bincount(srce, minlength=n_verts).astype(float)

    has_neighbours = degree > 0
    centroid = np.zeros_like(pos)
    centroid[has_neighbours] = neighbour_sum[has_neighbours] / degree[has_neighbours, None]

    deviation = np.where(has_neighbours[:, None], pos - centroid, 0.0)
    length = np.linalg.norm(deviation, axis=1)
    unit = np.zeros_like(deviation)
    nonzero = length > 0
    unit[nonzero] = deviation[nonzero] / length[nonzero, None]

    vert_df["dev_length"] = length
    for col, values in zip(DEVIATION_COLS[: len(coords)], unit.T):
        vert_df[col] = values


class VertDeviationMixin:
    """Adds :func:`update_vert_deviation` to a geometry's ``update_all``."""

    @classmethod
    def update_all(cls, eptm):
        super().update_all(eptm)
        update_vert_deviation(eptm)


class ClosedSheetVertDeviationGeometry(VertDeviationMixin, ClosedSheetGeometry):
    """``ClosedSheetGeometry`` (enclosed lumen volume) + the vertex-deviation
    columns, so a closed 2.5-D sheet can carry ``SurfaceElasticity``."""


class SheetVertDeviationGeometry(VertDeviationMixin, SheetGeometry):
    """``SheetGeometry`` + the vertex-deviation columns, for an open sheet."""


LOCAL_GEOMETRIES = {
    "ClosedSheetVertDeviationGeometry": ClosedSheetVertDeviationGeometry,
    "SheetVertDeviationGeometry": SheetVertDeviationGeometry,
}
