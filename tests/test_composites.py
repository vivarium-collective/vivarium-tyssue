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
ALL_NAMES = {"base_solver", "regulation", "stochastic", "jamming", "gradient",
             "anisotropic", "gillespie", "epithelium_2d", "tumor",
             "monolayer_liftoff", "hra_crypt_field", "hra_colon_surface"}


def test_all_composites_present():
    names = {p.name.split(".")[0] for p in COMPOSITES}
    assert names == ALL_NAMES, f"missing composites: {ALL_NAMES - names}"


@pytest.mark.parametrize("path", COMPOSITES, ids=lambda p: p.name.split(".")[0])
def test_composite_spec_loads(path):
    from viva_superpowers.composite_spec import load_spec

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
    for proc in ["EulerSolver", "CellDivisions", "CellDeaths", "StochasticLineTension", "CellJamming",
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
    from viva_superpowers.composite_spec import load_spec, build_composite_from_spec
    from vivarium_tyssue.core import build_core

    core = build_core()
    spec = load_spec(ROOT / "vivarium_tyssue" / "composites" / "anisotropic.composite.yaml")
    comp = build_composite_from_spec(spec, overrides={"interval": 0.1}, core=core)
    comp.run(2)  # smoke: a couple of solver steps


def test_history_file_streams_to_hdf5(tmp_path):
    """history_file makes EulerSolver record via disk-backed HistoryHdf5 (flat
    memory), and the archive still drives retrieve/browse (create_gif)."""
    import copy
    import sys
    sys.path.insert(0, str(ROOT))
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    from pbg_superpowers.composite_spec import load_spec, build_composite_from_spec
    from tyssue.core.history import HistoryHdf5
    from vivarium_tyssue.core import build_core

    hf5 = tmp_path / "hist.hf5"
    spec = copy.deepcopy(load_spec(ROOT / "vivarium_tyssue" / "composites" / "anisotropic.composite.yaml"))
    spec["emitters"] = []
    cfg = spec["state"]["Tyssue"]["config"]
    cfg["record_history"] = True
    cfg["history_file"] = str(hf5)
    def run_once():
        comp = build_composite_from_spec(spec, overrides={"interval": 0.1}, core=build_core())
        proc = comp.state["Tyssue"]["instance"]
        assert isinstance(proc.history, HistoryHdf5)
        comp.run(3)
        return proc

    proc = run_once()
    assert hf5.exists() and hf5.stat().st_size > 0
    assert proc.history.retrieve(0).Nv == proc.eptm.Nv  # reads a frame back from disk
    n_first = len(proc.history.time_stamps)

    # rerun into the same filename: the stale file is cleared, so it holds one
    # simulation's worth of stamps — not the two runs concatenated.
    n_second = len(run_once().history.time_stamps)
    assert n_second == n_first, f"rerun accumulated stamps: {n_first} -> {n_second}"


def test_default_history_stays_in_memory():
    """With recording on but no history_file, the in-RAM History is unchanged."""
    import copy
    import sys
    sys.path.insert(0, str(ROOT))
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    from pbg_superpowers.composite_spec import load_spec, build_composite_from_spec
    from tyssue.core.history import History, HistoryHdf5
    from vivarium_tyssue.core import build_core

    spec = copy.deepcopy(load_spec(ROOT / "vivarium_tyssue" / "composites" / "anisotropic.composite.yaml"))
    spec["state"]["Tyssue"]["config"]["record_history"] = True
    comp = build_composite_from_spec(spec, overrides={"interval": 0.1}, core=build_core())
    hist = comp.state["Tyssue"]["instance"].history
    assert isinstance(hist, History) and not isinstance(hist, HistoryHdf5)


# ---------------------------------------------------------------------------
# Run-one-step guardrail for EVERY composite (not just anisotropic).
#
# The 6 fork-only composites (base_solver, regulation, stochastic, jamming,
# gradient, monolayer_liftoff) previously had *no* run coverage — only
# spec-load. They exercise the EulerSolver step path we are about to swap to a
# Rust backend, so a failure here is the tripwire that says "a demo broke".
#
# gillespie is xfail: it fails NOW (pre-existing) with a pyarrow error — its
# spec still declares a parquet emitter that can't merge the object-dtype
# 'cell_type' column across frames on the crypt mesh. Documented, not silent.
# ---------------------------------------------------------------------------
_GILLESPIE_KNOWN_BROKEN = (
    "pre-existing: gillespie composite declares a parquet emitter that fails to "
    "merge the object-dtype 'cell_type' column across frames (ArrowTypeError)"
)


@pytest.mark.parametrize(
    "name",
    sorted(ALL_NAMES),
    ids=lambda n: n,
)
def test_composite_runs_one_step(name):
    """Each declared composite builds and advances at least one solver step."""
    import sys

    sys.path.insert(0, str(ROOT))
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    from viva_superpowers.composite_spec import load_spec, build_composite_from_spec
    from vivarium_tyssue.core import build_core

    if name == "gillespie":
        pytest.xfail(_GILLESPIE_KNOWN_BROKEN)

    core = build_core()
    spec = load_spec(ROOT / "vivarium_tyssue" / "composites" / f"{name}.composite.yaml")
    comp = build_composite_from_spec(spec, overrides={"interval": 0.01}, core=core)
    comp.run(1)  # smoke: one solver step is enough to catch a broken step path
