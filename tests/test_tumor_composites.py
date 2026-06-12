from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "vivarium_tyssue" / "composites"


def _run(spec_name, steps, interval=1.0):
    import sys
    sys.path.insert(0, str(ROOT))
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    from pbg_superpowers.composite_spec import load_spec, build_composite_from_spec
    from vivarium_tyssue.core import build_core

    core = build_core()
    spec = load_spec(SPECS / f"{spec_name}.composite.yaml")
    comp = build_composite_from_spec(spec, overrides={"interval": interval}, core=core)
    comp.run(steps)
    return comp


def test_baseline_runs_without_behaviors():
    comp = _run("epithelium_2d", 3)
    assert comp is not None  # smoke: flat-sheet mechanics complete


def test_tumor_composite_runs_end_to_end():
    # Smoke: COPASI + TumorCoupling + EulerSolver step together without error.
    comp = _run("tumor", 6)
    assert comp is not None
