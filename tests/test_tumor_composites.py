from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "vivarium_tyssue" / "composites"


def _run(spec_name, steps, interval=0.1):  # 0.1 keeps the vertex mechanics stable
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


def _cell_type_counts(comp):
    from collections import Counter
    face_df = comp.state["Tyssue"]["instance"].eptm.face_df
    return Counter(face_df["cell_type"])


def test_baseline_runs_without_behaviors():
    comp = _run("epithelium_2d", 5)
    # Mechanics-only: every cell stays healthy (no births/deaths).
    counts = _cell_type_counts(comp)
    assert set(counts) == {"healthy"}


def test_tumor_composite_drives_cell_fate():
    # COPASI fluxes drive tumor invasion: after a short run the mesh carries tumor
    # cells (and some dead) that the baseline never produces.
    import numpy as np
    np.random.seed(0)
    comp = _run("tumor", 25)
    counts = _cell_type_counts(comp)
    assert counts.get("tumor", 0) > 0, dict(counts)
    assert counts.get("healthy", 0) < 206, dict(counts)  # some healthy were consumed
