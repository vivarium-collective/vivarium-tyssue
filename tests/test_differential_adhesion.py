"""Differential-adhesion cell sorting: the behavior and the process that drives it.

The load-bearing property is that the classification is re-derived from the *live*
mesh every time the behavior runs — a junction rewired by a T1 must pick up the
tension of the pair of cells it separates NOW, not the pair it separated when the
run started. These tests pin that, plus the arithmetic of the classification
itself and the shape of the behavior node the process emits.
"""
import numpy as np
import pytest

from tyssue import Sheet, SheetGeometry
from tyssue.generation import three_faces_sheet

from vivarium_tyssue.behaviors.behaviors import (
    differential_adhesion,
    heterotypic_edges,
    opposite_half_edges,
)

HOMO = 0.1
HET = 0.9


def _three_faces(types=("A", "A", "B")):
    """Three faces around a shared centre vertex, with a free outer boundary."""
    sheet = Sheet("test", *three_faces_sheet())
    SheetGeometry.update_all(sheet)
    sheet.face_df["cell_type"] = list(types)
    sheet.edge_df["line_tension"] = 0.0
    sheet.edge_df["heterotypic"] = 0.0
    return sheet


def _brute_force_opposite(sheet):
    lookup = {(int(s), int(t)): i for i, (s, t)
              in enumerate(zip(sheet.edge_df["srce"], sheet.edge_df["trgt"]))}
    return np.array([lookup.get((int(t), int(s)), -1) for s, t
                     in zip(sheet.edge_df["srce"], sheet.edge_df["trgt"])])


# --------------------------------------------------------------------------
# opposite_half_edges / heterotypic_edges
# --------------------------------------------------------------------------
def test_opposite_matches_brute_force():
    sheet = _three_faces()
    np.testing.assert_array_equal(opposite_half_edges(sheet), _brute_force_opposite(sheet))


def test_open_sheet_has_boundary_half_edges():
    sheet = _three_faces()
    assert (opposite_half_edges(sheet) < 0).any(), "three_faces_sheet has a free border"


def test_closed_sphere_has_no_boundary_half_edges():
    pytest.importorskip("tyssue.generation.shapes")
    from tyssue import ClosedSheetGeometry
    from tyssue.generation.shapes import spherical_sheet

    np.random.seed(0)
    sphere = spherical_sheet(radius=3.0, Nf=40)
    ClosedSheetGeometry.update_all(sphere)
    assert (opposite_half_edges(sphere) >= 0).all()


def test_heterotypic_mask_matches_the_face_pairs():
    sheet = _three_faces(("A", "A", "B"))
    opposite = opposite_half_edges(sheet)
    own = sheet.upcast_face(sheet.face_df["cell_type"]).to_numpy()
    expected = np.array([
        opp >= 0 and own[i] != own[opp] for i, opp in enumerate(opposite)
    ])
    np.testing.assert_array_equal(heterotypic_edges(sheet), expected)


def test_boundary_half_edges_are_not_heterotypic():
    sheet = _three_faces()
    boundary = opposite_half_edges(sheet) < 0
    assert not heterotypic_edges(sheet)[boundary].any()


# --------------------------------------------------------------------------
# the behavior
# --------------------------------------------------------------------------
def test_behavior_sets_the_two_tensions():
    sheet = _three_faces(("A", "A", "B"))
    differential_adhesion(sheet, None, homotypic_tension=HOMO, heterotypic_tension=HET)

    hetero = heterotypic_edges(sheet)
    tension = sheet.edge_df["line_tension"].to_numpy()
    assert hetero.any() and (~hetero).any(), "fixture must have both kinds of junction"
    np.testing.assert_allclose(tension[hetero], HET)
    np.testing.assert_allclose(tension[~hetero], HOMO)


def test_behavior_records_the_classification():
    sheet = _three_faces(("A", "A", "B"))
    differential_adhesion(sheet, None, homotypic_tension=HOMO, heterotypic_tension=HET)
    np.testing.assert_array_equal(
        sheet.edge_df["heterotypic"].to_numpy().astype(bool), heterotypic_edges(sheet))


def test_boundary_tension_is_applied_only_when_asked():
    sheet = _three_faces()
    differential_adhesion(sheet, None, homotypic_tension=HOMO, heterotypic_tension=HET)
    boundary = opposite_half_edges(sheet) < 0
    np.testing.assert_allclose(sheet.edge_df["line_tension"].to_numpy()[boundary], HOMO)

    differential_adhesion(sheet, None, homotypic_tension=HOMO, heterotypic_tension=HET,
                          boundary_tension=0.25)
    np.testing.assert_allclose(sheet.edge_df["line_tension"].to_numpy()[boundary], 0.25)


def test_reclassifies_after_the_cell_types_change():
    """The whole point of doing this in a behavior: rerun it and the tensions
    follow the current state, they are not a table fixed at t=0."""
    sheet = _three_faces(("A", "A", "B"))
    differential_adhesion(sheet, None, homotypic_tension=HOMO, heterotypic_tension=HET)
    before = sheet.edge_df["line_tension"].to_numpy().copy()

    sheet.face_df["cell_type"] = ["A", "A", "A"]      # everything is one type now
    differential_adhesion(sheet, None, homotypic_tension=HOMO, heterotypic_tension=HET)

    assert (before == HET).any(), "the fixture had heterotypic junctions to start with"
    np.testing.assert_allclose(sheet.edge_df["line_tension"].to_numpy(), HOMO)
    assert not sheet.edge_df["heterotypic"].any()


def test_missing_type_column_is_a_no_op():
    sheet = _three_faces()
    sheet.face_df.drop(columns=["cell_type"], inplace=True)
    differential_adhesion(sheet, None, homotypic_tension=HOMO, heterotypic_tension=HET)
    np.testing.assert_allclose(sheet.edge_df["line_tension"].to_numpy(), 0.0)


def test_integer_line_tension_column_is_coerced():
    sheet = _three_faces(("A", "A", "B"))
    sheet.edge_df["line_tension"] = 0            # int64
    differential_adhesion(sheet, None, homotypic_tension=HOMO, heterotypic_tension=HET)
    assert sheet.edge_df["line_tension"].dtype == np.float64
    assert sheet.edge_df["line_tension"].max() == pytest.approx(HET)


# --------------------------------------------------------------------------
# the process
# --------------------------------------------------------------------------
def test_process_is_registered_and_emits_the_behavior():
    from vivarium_tyssue.core import build_core
    from vivarium_tyssue.processes.regulations import DifferentialAdhesion

    core = build_core()
    assert "DifferentialAdhesion" in core.link_registry

    sheet = _three_faces(("A", "A", "B"))
    process = DifferentialAdhesion({
        "homotypic_tension": HOMO,
        "heterotypic_tension": HET,
        "type_column": "cell_type",
        "boundary_tension": 0.0,
        "apply_boundary_tension": False,
        "record_column": "heterotypic",
    }, core=core)

    update = process.update({"datasets": {"edge_df": sheet.edge_df}}, 0.1)
    assert len(update["behaviors"]) == 1
    node = update["behaviors"][0]
    assert node["func"] == "differential_adhesion"
    assert node["homotypic_tension"] == HOMO
    assert node["heterotypic_tension"] == HET
    assert node["boundary_tension"] is None      # apply_boundary_tension is off


def test_process_emits_nothing_for_an_empty_mesh():
    from vivarium_tyssue.core import build_core
    from vivarium_tyssue.processes.regulations import DifferentialAdhesion

    core = build_core()
    process = DifferentialAdhesion({
        "homotypic_tension": HOMO,
        "heterotypic_tension": HET,
        "type_column": "cell_type",
        "boundary_tension": 0.0,
        "apply_boundary_tension": False,
        "record_column": "heterotypic",
    }, core=core)
    assert process.update({"datasets": {"edge_df": []}}, 0.1) == {"behaviors": []}


def test_behavior_is_in_the_behavior_map():
    from vivarium_tyssue.maps import BEHAVIOR_MAP
    assert BEHAVIOR_MAP["differential_adhesion"] is differential_adhesion
