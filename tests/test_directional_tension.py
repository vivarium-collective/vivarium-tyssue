"""Directional (planar-polarised) line tension.

The load-bearing property is the angle: the tension a junction is given must depend
on the **acute** angle between that junction and the user's polarity vector, and on
nothing else. That makes it invariant to three things that are all arbitrary — the
sign of the polarity vector, its magnitude, and which of ``srce``/``trgt`` the mesh
happens to list first — and it makes ``tension_max`` the value at 0 degrees and
``tension_min`` the value at 90.

These tests pin that arithmetic, the three alignment profiles, the sharpness
exponent, the shape of the behavior nodes the process emits, and the guards on a
malformed config.
"""
import numpy as np
import pandas as pd
import pytest

from vivarium_tyssue.behaviors.behaviors import apply_gradient, update_tension
from vivarium_tyssue.processes.regulations import ALIGNMENT_MAP, DirectionalLineTension

LO = 0.1
HI = 0.9

BASE = {
    "polarity": [1.0, 0.0],
    "tension_min": LO,
    "tension_max": HI,
    "profile": "cos2",
    "sharpness": 1.0,
    "coords": [],
    "record_column": "polar_alignment",
}


def make(**overrides):
    """A DirectionalLineTension with its config applied, without a Composite."""
    config = {**BASE, **overrides}
    process = DirectionalLineTension.__new__(DirectionalLineTension)
    process.config = config
    process.initialize(config)
    return process


def edges_at(degrees, length=1.0):
    """An ``edge_df``-shaped frame of unit edges at the given angles from +x."""
    radians = np.radians(np.asarray(degrees, dtype=float))
    return pd.DataFrame({
        "dx": length * np.cos(radians),
        "dy": length * np.sin(radians),
        "unique_id": np.arange(len(radians)),
    })


ANGLES = [0, 15, 30, 45, 60, 75, 90]


# --------------------------------------------------------------------------
# the angle
# --------------------------------------------------------------------------
def test_alignment_is_cos_squared_of_the_acute_angle():
    got = make().edge_alignment(edges_at(ANGLES))
    np.testing.assert_allclose(got, np.cos(np.radians(ANGLES)) ** 2, atol=1e-12)


def test_alignment_is_one_parallel_and_zero_perpendicular():
    got = make().edge_alignment(edges_at([0, 90]))
    assert got[0] == pytest.approx(1.0)
    assert got[1] == pytest.approx(0.0)


def test_obtuse_edges_fold_onto_their_acute_angle():
    """135 deg is the same junction as 45 deg: only the acute angle can matter."""
    got = make().edge_alignment(edges_at([45, 135, -45, 225]))
    np.testing.assert_allclose(got, got[0], atol=1e-12)


def test_reversing_an_edge_changes_nothing():
    """Which endpoint is srce is arbitrary, so d and -d must score the same."""
    process = make()
    forward = edges_at([0, 23, 47, 90])
    backward = forward.assign(dx=-forward["dx"], dy=-forward["dy"])
    np.testing.assert_allclose(process.edge_alignment(forward),
                               process.edge_alignment(backward), atol=1e-12)


def test_polarity_is_normalised_and_sign_invariant():
    unit = make(polarity=[1.0, 1.0])
    np.testing.assert_allclose(unit.polarity, [2 ** -0.5, 2 ** -0.5])
    edges = edges_at(ANGLES)
    for equivalent in ([3.0, 3.0], [-1.0, -1.0], [-0.25, -0.25]):
        np.testing.assert_allclose(make(polarity=equivalent).edge_alignment(edges),
                                   unit.edge_alignment(edges), atol=1e-12)


def test_alignment_is_independent_of_edge_length():
    process = make()
    np.testing.assert_allclose(process.edge_alignment(edges_at(ANGLES, length=0.01)),
                               process.edge_alignment(edges_at(ANGLES, length=17.0)),
                               atol=1e-12)


def test_rotating_the_polarity_rotates_the_response():
    """The law is attached to the polarity vector, not to the x axis."""
    rotated = make(polarity=[np.cos(np.radians(30)), np.sin(np.radians(30))])
    np.testing.assert_allclose(rotated.edge_alignment(edges_at([30, 60, 120])),
                               make().edge_alignment(edges_at([0, 30, 90])), atol=1e-12)


def test_zero_length_edge_takes_the_minimum_not_a_nan():
    """A junction mid-collapse has no direction; a NaN here would poison the solver."""
    edges = pd.DataFrame({"dx": [0.0], "dy": [0.0], "unique_id": [0]})
    process = make()
    assert process.edge_alignment(edges)[0] == 0.0
    tensions = process.update({"datasets": {"edge_df": edges}}, 0.1)
    assert list(tensions["behaviors"][0]["tension_update"].values()) == [LO]


def test_three_dimensional_polarity():
    process = make(polarity=[0.0, 0.0, 2.0])
    assert process.coords == ["x", "y", "z"]
    edges = pd.DataFrame({"dx": [0.0, 1.0, 0.0], "dy": [0.0, 0.0, 1.0],
                          "dz": [1.0, 0.0, 1.0], "unique_id": [0, 1, 2]})
    np.testing.assert_allclose(process.edge_alignment(edges), [1.0, 0.0, 0.5], atol=1e-12)


# --------------------------------------------------------------------------
# the tension
# --------------------------------------------------------------------------
def test_tension_spans_min_to_max_and_is_monotonic():
    tensions = np.array(list(
        make().update({"datasets": {"edge_df": edges_at(ANGLES)}}, 0.1)
        ["behaviors"][0]["tension_update"].values()))
    assert tensions[0] == pytest.approx(HI)     # parallel to the polarity axis
    assert tensions[-1] == pytest.approx(LO)    # perpendicular to it
    assert np.all(np.diff(tensions) < 0)        # strictly stronger the closer to p


def test_equal_extremes_give_isotropic_tension():
    """How the notebook's control run is built."""
    tensions = np.array(list(
        make(tension_min=0.5, tension_max=0.5)
        .update({"datasets": {"edge_df": edges_at(ANGLES)}}, 0.1)
        ["behaviors"][0]["tension_update"].values()))
    np.testing.assert_allclose(tensions, 0.5)


@pytest.mark.parametrize("profile", sorted(ALIGNMENT_MAP))
def test_every_profile_is_a_monotonic_map_from_one_to_zero(profile):
    got = make(profile=profile).edge_alignment(edges_at(ANGLES))
    assert got[0] == pytest.approx(1.0)
    assert got[-1] == pytest.approx(0.0)
    assert np.all(np.diff(got) < 0)


def test_cos2_at_sharpness_one_is_the_classical_nematic_law():
    """Lambda = mean * (1 + alpha cos 2theta) with alpha the anisotropy."""
    mean = 0.5 * (HI + LO)
    alpha = (HI - LO) / (HI + LO)
    radians = np.radians(ANGLES)
    tensions = np.array(list(
        make().update({"datasets": {"edge_df": edges_at(ANGLES)}}, 0.1)
        ["behaviors"][0]["tension_update"].values()))
    np.testing.assert_allclose(tensions, mean * (1 + alpha * np.cos(2 * radians)),
                               atol=1e-12)


def test_sharpness_narrows_the_high_tension_cone_without_moving_the_extremes():
    edges = edges_at(ANGLES)
    soft = np.array(list(make(sharpness=1.0)
                         .update({"datasets": {"edge_df": edges}}, 0.1)
                         ["behaviors"][0]["tension_update"].values()))
    sharp = np.array(list(make(sharpness=4.0)
                          .update({"datasets": {"edge_df": edges}}, 0.1)
                          ["behaviors"][0]["tension_update"].values()))
    assert sharp[0] == pytest.approx(soft[0])       # parallel: still tension_max
    assert sharp[-1] == pytest.approx(soft[-1])     # perpendicular: still tension_min
    assert np.all(sharp[1:-1] < soft[1:-1])         # everything between: lower


# --------------------------------------------------------------------------
# the behavior nodes the process emits
# --------------------------------------------------------------------------
def test_emits_a_tension_update_and_an_alignment_readout():
    behaviors = make().update({"datasets": {"edge_df": edges_at(ANGLES)}}, 0.1)["behaviors"]
    assert [node["func"] for node in behaviors] == ["update_tension", "apply_gradient"]
    assert set(behaviors[0]["tension_update"]) == set(range(len(ANGLES)))
    readout = behaviors[1]["parameter_updates"]["polar_alignment"]
    assert readout["dataframe"] == "edge"
    np.testing.assert_allclose(list(readout["update"].values()),
                               np.cos(np.radians(ANGLES)) ** 2, atol=1e-12)


def test_readout_can_be_switched_off():
    behaviors = make(record_column="").update(
        {"datasets": {"edge_df": edges_at(ANGLES)}}, 0.1)["behaviors"]
    assert [node["func"] for node in behaviors] == ["update_tension"]


def test_empty_mesh_emits_nothing():
    assert make().update({"datasets": {"edge_df": pd.DataFrame()}}, 0.1) == {"behaviors": []}


def test_behaviors_land_on_a_sheet():
    """End to end: what the process emits, executed the way the EventManager does."""
    from tyssue import Sheet, SheetGeometry
    from tyssue.generation import three_faces_sheet

    sheet = Sheet("test", *three_faces_sheet())
    SheetGeometry.update_all(sheet)
    sheet.edge_df["line_tension"] = 0.0
    sheet.edge_df["polar_alignment"] = 0.0

    process = make()
    behaviors = process.update({"datasets": {"edge_df": sheet.edge_df}}, 0.1)["behaviors"]
    update_tension(sheet, None, **{k: v for k, v in behaviors[0].items() if k != "func"})
    apply_gradient(sheet, None, **{k: v for k, v in behaviors[1].items() if k != "func"})

    expected = LO + (HI - LO) * process.edge_alignment(sheet.edge_df)
    np.testing.assert_allclose(sheet.edge_df["line_tension"], expected, atol=1e-12)
    np.testing.assert_allclose(sheet.edge_df["polar_alignment"],
                               process.edge_alignment(sheet.edge_df), atol=1e-12)
    # ...and on this real mesh the tension is a strictly decreasing function of the
    # acute angle: the more parallel the junction, the stronger it pulls.
    angles = np.abs(np.degrees(np.arctan2(sheet.edge_df["dy"], sheet.edge_df["dx"])))
    angles = np.minimum(angles, 180 - angles)
    by_angle = sheet.edge_df["line_tension"].to_numpy()[np.argsort(angles.to_numpy())]
    assert np.all(np.diff(by_angle) <= 1e-12)
    assert by_angle[0] > by_angle[-1]


# --------------------------------------------------------------------------
# config guards
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bad, match", [
    ({"polarity": [0.0, 0.0]}, "non-zero"),
    ({"polarity": []}, "non-zero"),
    ({"polarity": [np.nan, 1.0]}, "non-zero"),
    ({"profile": "quadratic"}, "unknown profile"),
    ({"coords": ["x"]}, "does not match"),
    ({"polarity": [1.0, 0.0, 0.0], "coords": ["x", "y"]}, "does not match"),
])
def test_malformed_config_is_rejected_at_initialisation(bad, match):
    with pytest.raises(ValueError, match=match):
        make(**bad)


def test_process_is_registered():
    from vivarium_tyssue.core import build_core

    core = build_core()
    assert core.access("DirectionalLineTension") is not None
