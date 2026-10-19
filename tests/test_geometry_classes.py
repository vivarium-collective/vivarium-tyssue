"""Invariants for the viva-lumenoid coupling-hierarchy geometry classes.

Covers the two levels stock tyssue doesn't ship, added in
``vivarium_tyssue.geometries`` and registered in ``GEOMETRY_MAP``:

  * ``CrossSectionGeometry`` — the height-resolving 2D cross-section exposes an
    explicit per-cell height DOF and basal-adhesion tracking.
  * ``LumenoidGeometry``     — the closed 3D lumenoid tracks a lumen volume +
    Laplace pressure and exposes a basal BM-coupling shell.

The whole suite skips gracefully when the tyssue simulator stack is unavailable;
the import-cleanliness test runs regardless (it must not need any optional viz
dep such as vispy).
"""
import importlib

import numpy as np
import pytest


def test_geometries_import_without_viz_deps():
    """The geometry module + the map import cleanly with no optional viz deps."""
    geom = importlib.import_module("vivarium_tyssue.geometries")
    assert hasattr(geom, "CrossSectionGeometry")
    assert hasattr(geom, "LumenoidGeometry")


def test_classes_registered_in_geometry_map():
    pytest.importorskip("tyssue", reason="tyssue geometry stack unavailable")
    from vivarium_tyssue.core_maps import GEOMETRY_MAP

    assert "CrossSectionGeometry" in GEOMETRY_MAP
    assert "LumenoidGeometry" in GEOMETRY_MAP


# --------------------------------------------------------------------------- #
# CrossSectionGeometry — height DOF + basal adhesion
# --------------------------------------------------------------------------- #
def _ring():
    from tyssue.generation import generate_ring

    return generate_ring(Nf=12, R_in=4.0, R_out=6.0)


def test_cross_section_exposes_height_dof():
    pytest.importorskip("tyssue", reason="tyssue geometry stack unavailable")
    from vivarium_tyssue.core_maps import GEOMETRY_MAP

    CS = GEOMETRY_MAP["CrossSectionGeometry"]
    ring = _ring()
    CS.update_all(ring)

    # height is an explicit, per-cell degree of freedom
    assert "height" in ring.face_df.columns
    heights = ring.face_df["height"].to_numpy()
    assert np.isfinite(heights).all()
    assert (heights > 0).all()
    # ring apico-basal extent is R_out - R_in = 2.0
    assert ring.settings["mean_cell_height"] == pytest.approx(2.0, abs=0.3)
    # the base AnnularGeometry lumen area is still tracked
    assert "lumen_volume" in ring.settings and ring.settings["lumen_volume"] > 0


def test_cross_section_basal_adhesion_tracks_substrate():
    pytest.importorskip("tyssue", reason="tyssue geometry stack unavailable")
    from vivarium_tyssue.core_maps import GEOMETRY_MAP

    CS = GEOMETRY_MAP["CrossSectionGeometry"]
    ring = _ring()
    CS.update_all(ring)

    assert "basal_offset" in ring.vert_df.columns
    # at rest the basal surface sits on its reference substrate
    assert ring.settings["basal_adhesion_strain"] == pytest.approx(0.0, abs=1e-6)

    # push the basal ring outward by 20%: height grows and basal strain rises
    h0 = ring.settings["mean_cell_height"]
    basal = (ring.vert_df["segment"] == "basal").to_numpy()
    center = ring.vert_df[["x", "y"]].mean().to_numpy()
    rel = ring.vert_df.loc[basal, ["x", "y"]].to_numpy() - center
    ring.vert_df.loc[basal, ["x", "y"]] = center + rel * 1.2
    CS.update_all(ring)

    assert ring.settings["mean_cell_height"] > h0
    assert ring.settings["basal_adhesion_strain"] > 0.1


# --------------------------------------------------------------------------- #
# LumenoidGeometry — lumen volume/pressure + basal BM shell
# --------------------------------------------------------------------------- #
def _lumenoid():
    from tyssue import Monolayer
    from tyssue.generation import ellipsoid_sheet, extrude

    sheet = ellipsoid_sheet(a=5.0, b=5.0, c=5.0, n_zs=6)
    return Monolayer("lumenoid", extrude(sheet.datasets, method="normals", scale=0.75))


def test_lumenoid_tracks_volume_and_pressure():
    pytest.importorskip("tyssue", reason="tyssue geometry stack unavailable")
    from vivarium_tyssue.core_maps import GEOMETRY_MAP

    LM = GEOMETRY_MAP["LumenoidGeometry"]
    mono = _lumenoid()
    mono.settings["lumen_elasticity"] = 2.0
    LM.update_all(mono)

    V0 = mono.settings["lumen_vol"]
    assert V0 > 0
    # prefered volume seeded from the first observed volume -> rest, zero pressure
    assert mono.settings["lumen_prefered_vol"] == pytest.approx(V0)
    assert mono.settings["lumen_pressure"] == pytest.approx(0.0, abs=1e-6)

    # shrink the shell: lumen volume drops and pressure rises (P = K*(V0 - V) > 0)
    mono.vert_df[["x", "y", "z"]] *= 0.9
    LM.update_all(mono)
    assert mono.settings["lumen_vol"] < V0
    assert mono.settings["lumen_pressure"] > 0
    expected = 2.0 * (V0 - mono.settings["lumen_vol"])
    assert mono.settings["lumen_pressure"] == pytest.approx(expected, rel=1e-6)


def test_lumenoid_exposes_basal_bm_shell():
    pytest.importorskip("tyssue", reason="tyssue geometry stack unavailable")
    from vivarium_tyssue.core_maps import GEOMETRY_MAP

    LM = GEOMETRY_MAP["LumenoidGeometry"]
    mono = _lumenoid()
    LM.update_all(mono)

    shell = LM.basal_shell(mono)
    assert len(shell) > 0
    assert mono.settings["bm_face_count"] == len(shell)
    assert mono.settings["bm_surface_area"] > 0

    surf = LM.bm_surface(mono)
    assert set(surf) == {"face_ids", "centroids", "area"}
    assert len(surf["face_ids"]) == len(shell)
    assert surf["area"] == pytest.approx(mono.settings["bm_surface_area"])
    # the BM shell is the outward surface (complement of the lumen-lining side)
    assert LM._bm_side(mono) != mono.settings.get("lumen_side", "basal")
