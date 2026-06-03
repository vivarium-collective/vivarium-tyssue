"""Validate the declared composites and visualization Steps.

This is the declarative replacement for the procedural ``get_test_*_spec`` /
``run_test_*`` helpers in ``tests/tests.py``: the composites now live as
``vivarium_tyssue/composites/*.composite.yaml`` and are exercised here.

  - every composite spec parses + loads
  - the workspace core registers all processes, types and visualizations
  - the stock-tyssue composite ('anisotropic') builds + runs end to end
    (the vessel/crypt composites need the custom tyssue fork; see README)
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
COMPOSITES = sorted((ROOT / "vivarium_tyssue" / "composites").glob("*.composite.yaml"))
ALL_NAMES = {"base_solver", "regulation", "stochastic", "jamming", "gradient", "anisotropic", "gillespie"}


def test_all_composites_present():
    names = {p.name.split(".")[0] for p in COMPOSITES}
    assert names == ALL_NAMES, f"missing composites: {ALL_NAMES - names}"


@pytest.mark.parametrize("path", COMPOSITES, ids=lambda p: p.name.split(".")[0])
def test_composite_spec_loads(path):
    from pbg_superpowers.composite_spec import load_spec

    spec = load_spec(path)
    assert spec["name"]
    assert "EulerSolver" in spec["requires"]["processes"]
    assert spec["state"]["Tyssue"]["address"] == "local:EulerSolver"


def test_core_registers_everything():
    import sys
    sys.path.insert(0, str(ROOT))
    from vivarium_tyssue.core import build_core

    core = build_core()
    reg = core.link_registry
    for proc in ["EulerSolver", "TestRegulations", "StochasticLineTension", "CellJamming",
                 "ParameterGradient", "AnisotropicTension", "Gillespie"]:
        assert proc in reg, f"{proc} not registered"
    for viz in ["TissueSheetGif", "TissueCryptGif3D"]:
        assert viz in reg, f"{viz} not registered"
    for typ in ["tyssue_data", "behaviors"]:
        assert typ in core.registry, f"{typ} not registered"


def test_anisotropic_runs_end_to_end():
    """The one composite that needs no fork-only tyssue symbols actually runs."""
    import sys
    sys.path.insert(0, str(ROOT))
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    from pbg_superpowers.composite_spec import load_spec, build_composite_from_spec
    from vivarium_tyssue.core import build_core

    core = build_core()
    spec = load_spec(ROOT / "vivarium_tyssue" / "composites" / "anisotropic.composite.yaml")
    comp = build_composite_from_spec(spec, overrides={"interval": 0.1}, core=core)
    comp.run(2)  # smoke: a couple of solver steps
