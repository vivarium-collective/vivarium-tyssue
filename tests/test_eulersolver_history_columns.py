"""Contract tests for EulerSolver's ``history_columns`` config and its schema.

These pin the behaviour of the history-column selection so an accidental or
autonomous change to ``vivarium_tyssue/processes/eulersolver.py`` is caught:

  - the config schema keeps its expected keys (add/remove trips the contract);
  - an absent ``history_columns`` records every column (default unchanged);
  - a listed dataframe records only the coords/topology minimum plus the listed
    columns, and the recorded frames still rebuild the epithelium.
"""
import copy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
ANISO = ROOT / "vivarium_tyssue" / "composites" / "anisotropic.composite.yaml"

# The full config surface EulerSolver exposes. A deliberate schema change updates
# this set; an unintended one fails here.
EXPECTED_SCHEMA_KEYS = {
    "name", "eptm", "tissue_type", "parameters", "geom", "effectors",
    "ref_effector", "factory", "auto_reconnect", "bounds", "output_columns",
    "history_columns", "settings", "maps", "backend", "substeps",
    "max_displacement", "record_history", "history_file", "history_save_every",
    "check_intersections", "intersection_options",
}


def _build(history_columns=None, history_file=None):
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    from pbg_superpowers.composite_spec import load_spec, build_composite_from_spec
    from vivarium_tyssue.core import build_core

    spec = copy.deepcopy(load_spec(ANISO))
    spec["emitters"] = []
    cfg = spec["state"]["Tyssue"]["config"]
    cfg["record_history"] = True
    if history_columns is not None:
        cfg["history_columns"] = history_columns
    if history_file is not None:
        cfg["history_file"] = str(history_file)
    comp = build_composite_from_spec(spec, overrides={"interval": 0.1}, core=build_core())
    return comp.state["Tyssue"]["instance"]


def test_config_schema_contract():
    from vivarium_tyssue.processes.eulersolver import EulerSolver

    assert set(EulerSolver.config_schema) == EXPECTED_SCHEMA_KEYS
    assert EulerSolver.config_schema["history_columns"] == "map[list[string]]"


def test_default_records_all_columns():
    proc = _build()
    for el in ("vert", "edge", "face"):
        recorded = set(proc.history.columns[el])
        assert recorded == set(proc.eptm.datasets[el].columns), el


def test_listed_df_keeps_minimum_plus_listed():
    proc0 = _build()
    coords = list(proc0.eptm.coords)
    vert_cols = list(proc0.eptm.vert_df.columns)
    extras = [c for c in vert_cols if c not in coords]
    picked, dropped = extras[0], extras[-1]
    assert picked != dropped

    proc = _build(history_columns={"vert_df": [picked]})
    # vert: exactly coords + the one listed column, nothing else.
    assert set(proc.history.columns["vert"]) == set(coords) | {picked}
    assert dropped not in proc.history.columns["vert"]
    # unlisted dataframes keep their default (all columns) and topology minimum.
    assert set(proc.history.columns["edge"]) == set(proc.eptm.edge_df.columns)
    assert {"srce", "trgt", "face"}.issubset(proc.history.columns["edge"])


def test_roundtrip_reconstructs_eptm():
    proc = _build()
    picked = next(c for c in proc.eptm.vert_df.columns if c not in proc.eptm.coords)
    proc = _build(history_columns={"vert_df": [picked], "face_df": []})
    # record a few frames so the trimmed history has something to rebuild from.
    for t in (0.1, 0.2, 0.3):
        proc.record(t)
    frame = proc.history.retrieve(0.2)
    assert (frame.Nv, frame.Ne, frame.Nf) == (proc.eptm.Nv, proc.eptm.Ne, proc.eptm.Nf)


def test_hdf5_selection_excludes_object_columns(tmp_path):
    from tyssue.core.history import HistoryHdf5

    proc0 = _build()
    picked = next(c for c in proc0.eptm.vert_df.columns if c not in proc0.eptm.coords)
    proc = _build(history_columns={"vert_df": [picked]}, history_file=tmp_path / "h.hf5")
    assert isinstance(proc.history, HistoryHdf5)
    assert set(proc.history.columns["vert"]) == set(proc0.eptm.coords) | {picked}
    # object-dtype columns are never in an HDF5 selection (unserializable).
    for el in proc.history.columns:
        for col in proc.history.columns[el]:
            assert proc.eptm.datasets[el][col].dtype != object, (el, col)
