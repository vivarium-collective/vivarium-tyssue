"""Vertex-deviation geometry — the columns ``SurfaceElasticity`` reads.

``SurfaceElasticity`` penalises how far each vertex sits from the centroid of its
neighbours, but it only *reads* ``vert_df["dev_length"]`` and the unit offset
``(dx, dy, dz)``; nothing in stock tyssue writes them for a sheet. These tests pin
the geometry that does, and the arithmetic it has to get right — most importantly
that the effector's own energy and gradient come out non-degenerate on a mesh it
has updated.
"""
import numpy as np
import pytest

from tyssue import ClosedSheetGeometry, Sheet, SheetGeometry
from tyssue.generation import three_faces_sheet

from vivarium_tyssue.core_maps import GEOMETRY_MAP
from vivarium_tyssue.geometry import (
    ClosedSheetVertDeviationGeometry,
    SheetVertDeviationGeometry,
    update_vert_deviation,
)

DEV_COLS = ["dx", "dy", "dz"]


def _flat_sheet():
    """Three coplanar faces: every vertex lies in z = 0."""
    sheet = Sheet("test", *three_faces_sheet())
    SheetGeometry.update_all(sheet)
    return sheet


def _sphere(n_faces=40, radius=3.0):
    pytest.importorskip("tyssue.generation.shapes")
    from tyssue.generation.shapes import spherical_sheet

    np.random.seed(0)
    sphere = spherical_sheet(radius=radius, Nf=n_faces)
    ClosedSheetVertDeviationGeometry.update_all(sphere)
    return sphere


def _brute_force_deviation(sheet):
    """The definition, one vertex at a time."""
    lengths, units = [], []
    for label in sheet.vert_df.index:
        neighbours = sheet.edge_df.loc[sheet.edge_df["srce"] == label, "trgt"]
        pos = sheet.vert_df.loc[label, sheet.coords].to_numpy(dtype=float)
        if not len(neighbours):
            lengths.append(0.0)
            units.append(np.zeros(len(sheet.coords)))
            continue
        centroid = sheet.vert_df.loc[neighbours, sheet.coords].to_numpy(dtype=float).mean(axis=0)
        offset = pos - centroid
        length = float(np.linalg.norm(offset))
        lengths.append(length)
        units.append(offset / length if length else np.zeros(len(sheet.coords)))
    return np.array(lengths), np.array(units)


# --------------------------------------------------------------------------
# the columns themselves
# --------------------------------------------------------------------------
def test_matches_the_definition_on_a_sphere():
    sphere = _sphere()
    lengths, units = _brute_force_deviation(sphere)
    np.testing.assert_allclose(sphere.vert_df["dev_length"].to_numpy(), lengths, atol=1e-12)
    np.testing.assert_allclose(sphere.vert_df[DEV_COLS].to_numpy(), units, atol=1e-12)


def test_unit_columns_are_unit_vectors():
    sphere = _sphere()
    norms = np.linalg.norm(sphere.vert_df[DEV_COLS].to_numpy(), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_deviation_points_outward_on_a_sphere():
    """A convex surface bulges away from its neighbours' centroid, so the offset
    has a positive radial component everywhere."""
    sphere = _sphere()
    pos = sphere.vert_df[sphere.coords].to_numpy(dtype=float)
    radial = pos / np.linalg.norm(pos, axis=1)[:, None]
    assert (np.einsum("ij,ij->i", sphere.vert_df[DEV_COLS].to_numpy(), radial) > 0).all()


def test_a_spike_stands_out_from_the_rest():
    sphere = _sphere()
    baseline = sphere.vert_df["dev_length"].max()
    # Push one vertex straight out — an everted cell, in miniature.
    label = sphere.vert_df.index[0]
    sphere.vert_df.loc[label, sphere.coords] *= 1.5
    ClosedSheetVertDeviationGeometry.update_all(sphere)
    assert sphere.vert_df.loc[label, "dev_length"] > 5 * baseline


def test_flat_sheet_interior_vertex_has_no_deviation():
    """The centre vertex of three_faces_sheet is the centroid of its neighbours."""
    sheet = _flat_sheet()
    update_vert_deviation(sheet)
    interior = sheet.edge_df["srce"].value_counts()
    centre = interior.idxmax()
    assert sheet.vert_df.loc[centre, "dev_length"] == pytest.approx(0.0, abs=1e-12)


def test_zero_deviation_gives_a_zero_unit_vector():
    sheet = _flat_sheet()
    update_vert_deviation(sheet)
    zero = sheet.vert_df["dev_length"] < 1e-12
    assert zero.any()
    np.testing.assert_allclose(sheet.vert_df.loc[zero, DEV_COLS].to_numpy(), 0.0)


def test_survives_a_non_contiguous_vertex_index():
    """srce/trgt carry index *labels*; a mesh whose vert_df index has gaps (mid-run,
    before a reset_index) must still map to the right rows."""
    sphere = _sphere()
    expected = sphere.vert_df["dev_length"].to_numpy().copy()

    relabel = {old: old * 3 + 1 for old in sphere.vert_df.index}
    sphere.vert_df.index = [relabel[i] for i in sphere.vert_df.index]
    sphere.edge_df["srce"] = sphere.edge_df["srce"].map(relabel)
    sphere.edge_df["trgt"] = sphere.edge_df["trgt"].map(relabel)

    update_vert_deviation(sphere)
    np.testing.assert_allclose(sphere.vert_df["dev_length"].to_numpy(), expected, atol=1e-12)


# --------------------------------------------------------------------------
# wiring: geometry classes and the effector that consumes them
# --------------------------------------------------------------------------
def test_geometries_are_in_the_geometry_map():
    assert GEOMETRY_MAP["ClosedSheetVertDeviationGeometry"] is ClosedSheetVertDeviationGeometry
    assert GEOMETRY_MAP["SheetVertDeviationGeometry"] is SheetVertDeviationGeometry


def test_closed_variant_still_computes_the_lumen_volume():
    """The mixin must not displace what ClosedSheetGeometry already did."""
    sphere = _sphere()
    expected = sphere.settings["lumen_vol"]
    ClosedSheetGeometry.update_all(sphere)
    assert sphere.settings["lumen_vol"] == pytest.approx(expected)


def test_surface_elasticity_energy_and_gradient_are_finite():
    from tyssue.dynamics.effectors import SurfaceElasticity

    sphere = _sphere()
    sphere.vert_df["surface_elasticity"] = 1.0
    sphere.vert_df["prefered_deviation"] = 0.0
    sphere.vert_df["is_active"] = 1.0

    energy = SurfaceElasticity.energy(sphere)
    grad_srce, grad_trgt = SurfaceElasticity.gradient(sphere)

    assert np.isfinite(energy.to_numpy()).all()
    assert (energy.to_numpy() > 0).all(), "a curved sphere deviates from a flat one"
    assert grad_trgt is None, "SurfaceElasticity is a vertex effector"
    assert grad_srce.shape == (sphere.Nv, len(sphere.coords))
    assert np.isfinite(grad_srce.to_numpy()).all()


def test_zero_energy_at_the_prefered_deviation():
    from tyssue.dynamics.effectors import SurfaceElasticity

    sphere = _sphere()
    sphere.vert_df["surface_elasticity"] = 1.0
    sphere.vert_df["is_active"] = 1.0
    # Target each vertex's own current deviation: the sphere is then at rest.
    sphere.vert_df["prefered_deviation"] = sphere.vert_df["dev_length"]

    np.testing.assert_allclose(SurfaceElasticity.energy(sphere).to_numpy(), 0.0, atol=1e-24)
    grad_srce, _ = SurfaceElasticity.gradient(sphere)
    np.testing.assert_allclose(grad_srce.to_numpy(), 0.0, atol=1e-12)
