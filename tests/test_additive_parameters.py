"""Parameter-setting processes ship deltas, and the behaviors add them in place.

The load-bearing property is composition: several processes can drive the same
parameter (directional + stochastic line tension, adhesion + stochastic, a
perimeter gradient + the jamming ramp) and each moves only its own share. Every
process remembers its share instead of reading it back from the mesh, because the
solver executes a behavior a step or two after it is emitted. These tests pin the
arithmetic of the behaviors, the share bookkeeping (including the solver's lag and
rows cloned mid-run), and a short composite run with two tension processes.
"""
from collections import deque

import numpy as np
import pandas as pd
import pytest

from tyssue import Sheet, SheetGeometry
from tyssue.generation import three_faces_sheet

from vivarium_tyssue.behaviors.behaviors import (
    cell_jamming,
    differential_adhesion,
    update_parameter,
)
from vivarium_tyssue.processes.regulations import (
    DirectionalLineTension,
    ParameterGradient,
    StochasticLineTension,
)

HOMO, HET = 0.1, 0.9


def _bare(cls, **config):
    """A process with its config applied, without a Composite (as the directional
    tension tests do), so the schema defaults are given explicitly."""
    process = cls.__new__(cls)
    process.config = config
    process.initialize(config)
    return process


def _stochastic(tau=0.5, sigma=0.2):
    return _bare(StochasticLineTension, tau=tau, sigma=sigma, component="stochastic_tension")


def _directional():
    return _bare(DirectionalLineTension, polarity=[1.0, 0.0], tension_min=0.1,
                 tension_max=0.9, profile="cos2", sharpness=1.0, coords=[],
                 record_column="", component="directional_tension")


def _sheet(types=("A", "A", "B")):
    sheet = Sheet("test", *three_faces_sheet())
    SheetGeometry.update_all(sheet)
    sheet.face_df["cell_type"] = list(types)
    sheet.edge_df["line_tension"] = 0.0
    sheet.edge_df["unique_id"] = np.arange(sheet.Ne)
    sheet.face_df["unique_id"] = np.arange(sheet.Nf)
    return sheet


def _apply(sheet, node):
    kwargs = {k: v for k, v in node.items() if k != "func"}
    assert node["func"] == "update_parameter"
    update_parameter(sheet, None, **kwargs)


# --------------------------------------------------------------------------
# the behaviors
# --------------------------------------------------------------------------
def test_update_parameter_adds_to_line_tension_and_mirrors_the_component():
    sheet = _sheet()
    sheet.edge_df["line_tension"] = 0.5
    update_parameter(sheet, None, "line_tension", "edge",
                     parameter_update={0: 0.25, 3: -0.5}, component="mine")
    lt = sheet.edge_df["line_tension"].to_numpy()
    assert lt[0] == pytest.approx(0.75) and lt[3] == pytest.approx(0.0)
    assert np.allclose(np.delete(lt, [0, 3]), 0.5)          # other edges untouched
    mine = sheet.edge_df["mine"].to_numpy()                 # created at 0
    assert mine[0] == pytest.approx(0.25) and mine[3] == pytest.approx(-0.5)
    assert np.allclose(np.delete(mine, [0, 3]), 0.0)


def test_update_parameter_coerces_an_integer_column():
    sheet = _sheet()
    sheet.edge_df["line_tension"] = 0
    update_parameter(sheet, None, "line_tension", "edge", parameter_update={1: 0.3})
    assert sheet.edge_df["line_tension"].dtype == np.float64
    assert sheet.edge_df["line_tension"].iloc[1] == pytest.approx(0.3)


def test_update_parameter_adds_to_any_element():
    sheet = _sheet()
    sheet.face_df["prefered_perimeter"] = 3.6
    update_parameter(sheet, None, "prefered_perimeter", "face_df",
                     parameter_update={0: 0.4, 2: -0.1}, component="grad")
    np.testing.assert_allclose(sheet.face_df["prefered_perimeter"], [4.0, 3.6, 3.5])
    np.testing.assert_allclose(sheet.face_df["grad"], [0.4, 0.0, -0.1])


def test_adhesion_keeps_tension_other_processes_added():
    sheet = _sheet(("A", "A", "B"))
    noise = np.linspace(-0.05, 0.05, sheet.Ne)
    sheet.edge_df["line_tension"] = noise                    # e.g. a stochastic share
    differential_adhesion(sheet, None, HOMO, HET, record_column=None)
    target = sheet.edge_df["adhesion_tension"].to_numpy()
    assert set(np.round(target, 12)) <= {HOMO, HET}
    np.testing.assert_allclose(sheet.edge_df["line_tension"], noise + target)

    sheet.face_df["cell_type"] = "A"                        # reclassify: all homotypic
    differential_adhesion(sheet, None, HOMO, HET, record_column=None)
    np.testing.assert_allclose(sheet.edge_df["adhesion_tension"], HOMO)
    np.testing.assert_allclose(sheet.edge_df["line_tension"], noise + HOMO)


def test_adhesion_owns_a_line_tension_seeded_with_its_component():
    sheet = _sheet(("A", "A", "B"))
    sheet.edge_df["line_tension"] = HOMO
    sheet.edge_df["adhesion_tension"] = HOMO
    differential_adhesion(sheet, None, HOMO, HET, record_column=None)
    np.testing.assert_allclose(sheet.edge_df["line_tension"], sheet.edge_df["adhesion_tension"])


def test_jamming_ramp_matches_the_multiplicative_ramp_and_stacks():
    sheet = _sheet()
    sheet.face_df["prefered_perimeter"] = [3.8, 3.9, 4.0]
    expected = sheet.face_df["prefered_perimeter"].to_numpy() * (1 - 0.05 * 0.1)

    class Manager:
        def append(self, *args, **kwargs):
            self.requeued = kwargs

    manager = Manager()
    cell_jamming(sheet, manager, rate=-0.05, limits=[3.0, 4.2], dt=0.1, component="jam")
    np.testing.assert_allclose(sheet.face_df["prefered_perimeter"], expected)
    np.testing.assert_allclose(sheet.face_df["jam"], expected - [3.8, 3.9, 4.0])
    assert manager.requeued["component"] == "jam"


# --------------------------------------------------------------------------
# share bookkeeping in the processes
# --------------------------------------------------------------------------
def test_first_deltas_are_the_values_on_a_fresh_mesh():
    sheet = _sheet()
    process = _directional()
    node = process.update({"datasets": {"edge_df": sheet.edge_df}}, 0.1)["behaviors"][0]
    expected = 0.1 + 0.8 * process.edge_alignment(sheet.edge_df)
    np.testing.assert_allclose(list(node["parameter_update"].values()), expected)
    assert (node["parameter"], node["dataframe"]) == ("line_tension", "edge")
    assert node["component"] == "directional_tension"


def test_a_steady_target_ships_zero_deltas():
    sheet = _sheet()
    process = _directional()
    process.update({"datasets": {"edge_df": sheet.edge_df}}, 0.1)
    node = process.update({"datasets": {"edge_df": sheet.edge_df}}, 0.1)["behaviors"][0]
    np.testing.assert_allclose(list(node["parameter_update"].values()), 0.0, atol=1e-15)


@pytest.mark.parametrize("lag", [0, 1, 2, 3])
def test_ou_share_is_exact_whatever_the_solver_lag(lag):
    """x(n+1) = decay x(n) + scale N(0,1) holds for the emitted share even though the
    mesh the process reads trails its deltas by ``lag`` steps."""
    tau, sigma, dt, steps = 0.5, 0.2, 0.1, 40
    sheet = _sheet()
    process = _stochastic(tau, sigma)
    pending = deque()
    np.random.seed(7)
    shares = []
    for _ in range(steps):
        node = process.update({"datasets": {"edge_df": sheet.edge_df.copy()}}, dt)["behaviors"][0]
        pending.append(node)
        shares.append(np.array([process.share.last[u] for u in sheet.edge_df["unique_id"]]))
        while len(pending) > lag:
            _apply(sheet, pending.popleft())
    np.random.seed(7)
    decay = np.exp(-dt / tau)
    scale = sigma * np.sqrt(1 - np.exp(-2 * dt / tau))
    x = np.zeros(sheet.Ne)
    for n in range(steps):
        x = decay * x + scale * np.random.randn(sheet.Ne)
        np.testing.assert_allclose(shares[n], x, atol=1e-12)
    while pending:
        _apply(sheet, pending.popleft())
    np.testing.assert_allclose(sheet.edge_df["line_tension"], x, atol=1e-12)
    np.testing.assert_allclose(sheet.edge_df["stochastic_tension"], x, atol=1e-12)


def test_two_tension_processes_compose_through_a_lagging_solver():
    sheet = _sheet()
    stochastic, directional = _stochastic(), _directional()
    pending = deque()
    np.random.seed(3)
    for _ in range(25):
        state = {"datasets": {"edge_df": sheet.edge_df.copy()}}
        pending.extend(stochastic.update(state, 0.1)["behaviors"])
        pending.extend(directional.update(state, 0.1)["behaviors"])
        while len(pending) > 4:                      # two steps late, two nodes a step
            _apply(sheet, pending.popleft())
    while pending:
        _apply(sheet, pending.popleft())
    edge = sheet.edge_df
    np.testing.assert_allclose(edge["line_tension"],
                               edge["stochastic_tension"] + edge["directional_tension"],
                               atol=1e-12)
    np.testing.assert_allclose(edge["directional_tension"],
                               0.1 + 0.8 * directional.edge_alignment(edge), atol=1e-12)
    assert edge["stochastic_tension"].abs().max() > 0


def test_an_unseen_row_takes_its_share_from_the_component_column():
    """A row cloned mid-run (new uid) copies the parent's share; the process must
    start from it, or the copied share would never relax."""
    sheet = _sheet()
    sheet.edge_df["line_tension"] = 0.0
    process = _directional()
    _apply(sheet, process.update({"datasets": {"edge_df": sheet.edge_df}}, 0.1)["behaviors"][0])
    clone = sheet.edge_df.iloc[[0]].copy()
    clone.index = [sheet.edge_df.index.max() + 1]
    clone["unique_id"] = 999
    clone["dx"], clone["dy"] = 0.0, 1.0                      # perpendicular: minimum
    sheet.edge_df = pd.concat([sheet.edge_df, clone])
    node = process.update({"datasets": {"edge_df": sheet.edge_df}}, 0.1)["behaviors"][0]
    _apply(sheet, node)
    row = sheet.edge_df[sheet.edge_df["unique_id"] == 999].iloc[0]
    assert row["directional_tension"] == pytest.approx(0.1)
    assert row["line_tension"] == pytest.approx(0.1)


def test_gradient_seeded_with_the_baseline_replaces_it_and_stacks_with_jamming():
    sheet = _sheet()
    sheet.face_df["prefered_perimeter"] = 3.6
    sheet.face_df["gradient_prefered_perimeter"] = 3.6        # gradient owns the baseline
    process = _bare(ParameterGradient, gradient_type="linear", axis="x",
                    args={"m": -0.1, "c": 4.6},
                    model_parameters={"prefered_perimeter": "face"},
                    component_prefix="gradient_")
    _apply(sheet, process.update({"datasets": {"face_df": sheet.face_df}})["behaviors"][0])
    target = 4.6 - 0.1 * sheet.face_df["x"].to_numpy()
    np.testing.assert_allclose(sheet.face_df["prefered_perimeter"], target)

    sheet.face_df["prefered_perimeter"] += 0.2               # another process's increment
    _apply(sheet, process.update({"datasets": {"face_df": sheet.face_df}})["behaviors"][0])
    np.testing.assert_allclose(sheet.face_df["prefered_perimeter"], target + 0.2)


def test_gradient_ships_one_update_parameter_node_per_parameter():
    sheet = _sheet()
    sheet.face_df["prefered_area"] = 0.0
    process = _bare(ParameterGradient, gradient_type="linear", axis="x",
                    args={"m": 1.0, "c": 0.0},
                    model_parameters={"prefered_area": "face", "line_tension": "edge"},
                    component_prefix="gradient_")
    nodes = process.update({"datasets": {"face_df": sheet.face_df, "edge_df": sheet.edge_df}})
    nodes = nodes["behaviors"]
    assert [(n["func"], n["parameter"], n["dataframe"], n["component"]) for n in nodes] == [
        ("update_parameter", "prefered_area", "face", "gradient_prefered_area"),
        ("update_parameter", "line_tension", "edge", "gradient_line_tension"),
    ]
    for node in nodes:
        _apply(sheet, node)
    np.testing.assert_allclose(sheet.face_df["prefered_area"], sheet.face_df["x"])
    np.testing.assert_allclose(sheet.edge_df["line_tension"],
                               (sheet.edge_df["sx"] + sheet.edge_df["tx"]) / 2)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------
def test_directional_and_stochastic_compose_in_a_composite():
    from pathlib import Path
    from process_bigraph import Composite
    from vivarium_tyssue.core import build_core

    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    dataset = Path(__file__).parent / "test_square.hf5"
    config = {
        "name": "additive", "eptm": str(dataset), "tissue_type": "Sheet",
        "parameters": {
            "face_df": {"area_elasticity": 1.0, "prefered_area": 1.0,
                        "perimeter_elasticity": 0.1, "prefered_perimeter": 3.6,
                        "is_alive": 1},
            "edge_df": {"line_tension": 0.0, "is_active": 1,
                        "directional_tension": 0.0, "stochastic_tension": 0.0},
            "vert_df": {"viscosity": 1.0, "is_alive": 1},
        },
        "geom": "SheetGeometry",
        "effectors": ["LineTension", "FaceAreaElasticity", "PerimeterElasticity"],
        "ref_effector": "FaceAreaElasticity", "factory": "model_factory",
        "settings": {"threshold_length": 0.03}, "auto_reconnect": {"enabled": True},
        "bounds": None, "output_columns": {}, "history_columns": {}, "maps": {},
        "backend": "python", "substeps": 1, "max_displacement": 0.0,
        "record_history": True,
    }
    dt = 0.1
    spec = {
        "Tyssue": {"_type": "process", "address": "local:EulerSolver", "config": config,
                   "inputs": {"behaviors": ["Behaviors"], "global_time": ["global_time"]},
                   "outputs": {"datasets": ["Tissue State"],
                               "network_changed": ["Network Changed"],
                               "behaviors_update": ["Behaviors"]},
                   "interval": dt},
        "Stochastic": {"_type": "process", "address": "local:StochasticLineTension",
                       "config": {"tau": 0.5, "sigma": 0.05},
                       "inputs": {"datasets": ["Tissue State"]},
                       "outputs": {"behaviors": ["Behaviors"], "datasets": ["Tissue State"]}, "interval": dt},
        "Directional": {"_type": "process", "address": "local:DirectionalLineTension",
                        "config": {"polarity": [1.0, 0.0], "tension_min": 0.02,
                                   "tension_max": 0.2, "record_column": ""},
                        "inputs": {"datasets": ["Tissue State"]},
                        "outputs": {"behaviors": ["Behaviors"], "datasets": ["Tissue State"]}, "interval": dt},
        "Network Changed": False,
        "Behaviors": {},
    }
    np.random.seed(11)
    sim = Composite({"state": spec}, core=build_core())
    sim.run(3.0)
    edge = sim.state["Tyssue"]["instance"].eptm.edge_df
    np.testing.assert_allclose(edge["line_tension"],
                               edge["directional_tension"] + edge["stochastic_tension"],
                               atol=1e-10)
    assert edge["directional_tension"].between(0.02 - 1e-9, 0.2 + 1e-9).all()
    assert edge["directional_tension"].std() > 0.01          # both shares are live
    assert 0.005 < edge["stochastic_tension"].std() < 0.2
