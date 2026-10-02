"""Per-cell SBML engine (``vivarium_tyssue.sbml``) and the ``SBMLCellCycle`` process.

The parser is pinned against a hand-written copy of the Gérard & Goldbeter 2011
skeleton model (BIOMD0000000941), so a mis-translated kinetic law cannot hide
behind plausible-looking oscillations. The process tests run on synthetic
``face_df`` frames: cells grow in G1/G2, hold in S/M and divide on entering M.
One real tyssue division checks that a daughter inherits its mother's
subcellular state.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("libsbml")

from vivarium_tyssue.sbml import VectorizedSBML  # noqa: E402
from vivarium_tyssue.processes.cell_cycle import SBMLCellCycle  # noqa: E402

MODEL = Path(__file__).resolve().parents[1] / "workspace" / "datasets" / "BIOMD0000000941.xml"
SPECIES = ["cyclin_D_Cdk4_6", "transcription_factor_E2F_active", "cyclin_E_Cdk2",
           "cyclin_A_Cdk2", "cyclin_B_Cdk1", "Cdc20_active"]


def gerard_goldbeter_rhs(y, p):
    """The skeleton model written out by hand (see the experiment README)."""
    Md, E2F, Me, Ma, Mb, C = y
    return np.array([
        p["vsd"] * p["GF"] / (p["Kgf"] + p["GF"]) - p["Vdd"] * Md / (p["Kdd"] + Md),
        p["V1e2f"] * (p["E2F_total"] - E2F) / (p["K1e2f"] + p["E2F_total"] - E2F) * (Md + Me)
        - p["V2e2f"] * E2F / (p["K2e2f"] + E2F) * Ma,
        p["vse"] * E2F - p["Vde"] * Ma * Me / (p["Kde"] + Me),
        p["vsa"] * E2F - p["Vda"] * C * Ma / (p["Kda"] + Ma),
        p["vsb"] * Ma - p["Vdb"] * C * Mb / (p["Kdb"] + Mb),
        p["V1cdc20"] * Mb * (p["Cdc20_total"] - C) / (p["K1cdc20"] + p["Cdc20_total"] - C)
        - p["V2cdc20"] * C / (p["K2cdc20"] + C),
    ])


@pytest.fixture(scope="module")
def model():
    return VectorizedSBML(MODEL)


def _crossings_up(x, level):
    return np.flatnonzero((x[:-1] < level) & (x[1:] >= level))


# --------------------------------------------------------------------- parser
def test_state_and_constants(model):
    assert model.state_ids == SPECIES
    p = model.default_params()
    # Conserved totals are species only ever used as modifiers -> constants.
    assert p["E2F_total"] == 3.0 and p["Cdc20_total"] == 5.0
    assert p["GF"] == 1.0
    np.testing.assert_allclose(model.initial_state(), [0.7205, 2.4855, 2.0, 2.6, 1.0, 0.5])


def test_rhs_matches_hand_written_model(model):
    rng = np.random.default_rng(0)
    Y = np.vstack([rng.uniform(0, 2.9, 200) for _ in SPECIES])
    P = model.params({"GF": 0.37})
    np.testing.assert_allclose(model.rhs(Y, P), gerard_goldbeter_rhs(Y, P), rtol=1e-12)


def test_per_cell_parameter_arrays_broadcast(model):
    Y = np.tile(model.initial_state()[:, None], (1, 3))
    gf = np.array([0.0, 0.5, 2.0])
    dY = model.rhs(Y, model.params({"GF": gf}))
    for i, g in enumerate(gf):
        np.testing.assert_allclose(dY[:, i], gerard_goldbeter_rhs(Y[:, i], model.params({"GF": g})))


# ------------------------------------------------------------------- dynamics
def test_period_at_gf_one(model):
    y = model.integrate(model.initial_state()[:, None], model.params(), 200.0)[:, 0]
    t, states = model.trajectory(y, model.params(), 150.0)
    ups = t[_crossings_up(states[:, model.index("cyclin_B_Cdk1")], 1.0)]
    assert len(ups) >= 5
    assert np.mean(np.diff(ups)) == pytest.approx(23.6, abs=0.3)


def test_quiescent_without_growth_factor(model):
    # The SBML initial state is in G2/M (cyclin B = 1, cyclin A high). Without GF
    # the cell finishes that mitosis, never starts another cycle, and settles in G0.
    t, states = model.trajectory(model.initial_state(), model.params({"GF": 0.0}), 300.0)
    mb = states[:, model.index("cyclin_B_Cdk1")]
    assert mb.max() > 2.0
    assert len(_crossings_up(mb, 1.0)) == 0
    assert mb[t > 60.0].max() < 0.2


def test_limit_cycle_samples_cover_phases(model):
    S = model.limit_cycle_samples(model.params(), 400, rng=np.random.default_rng(1))
    mb = S[model.index("cyclin_B_Cdk1")]
    assert S.shape == (6, 400)
    assert (mb > 1.0).any() and (mb < 0.1).any()


def test_unsupported_sbml_raises():
    sbml = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
  <model id="with_event">
    <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
    <listOfParameters><parameter id="k" value="1" constant="false"/></listOfParameters>
    <listOfEvents>
      <event id="bump" useValuesFromTriggerTime="true">
        <trigger initialValue="false" persistent="true">
          <math xmlns="http://www.w3.org/1998/Math/MathML">
            <apply><gt/><csymbol encoding="text" definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol><cn>1</cn></apply>
          </math>
        </trigger>
        <listOfEventAssignments>
          <eventAssignment variable="k"><math xmlns="http://www.w3.org/1998/Math/MathML"><cn>2</cn></math></eventAssignment>
        </listOfEventAssignments>
      </event>
    </listOfEvents>
  </model>
</sbml>"""
    with pytest.raises(NotImplementedError, match="events"):
        VectorizedSBML(sbml)


# -------------------------------------------------------------------- process
# The Gérard–Goldbeter phases, as the demo defines them (first match wins).
PHASES = [
    "S: cyclin_E_Cdk2 > 1",
    "M: cyclin_B_Cdk1 > 1 & cyclin_E_Cdk2 <= 1 & cyclin_A_Cdk2 < 2",
    "G2: cyclin_B_Cdk1 > 1 & cyclin_E_Cdk2 <= 1",
]


def _process(**overrides):
    config = {
        "model_source": str(MODEL),
        "phases": PHASES,
        "time_scale": 2.36,
        "initial_state": "sbml",
    }
    config.update(overrides)
    from vivarium_tyssue.core import build_core
    return SBMLCellCycle(config, core=build_core())


def _faces(n, cell_type="normal", area=1.0):
    return pd.DataFrame({
        "unique_id": np.arange(n), "area": area, "cell_type": cell_type, "is_alive": 1.0,
    })


def _run(proc, faces, steps, dt=0.05, t0=0.0, record=None):
    """Splits emitted over ``steps`` updates, as (step, uid) pairs. ``record``
    collects (phase names, growing uids) per step."""
    splits = []
    for k in range(steps):
        out = proc.update({"datasets": {"face_df": faces}, "global_time": t0 + k * dt}, dt)
        by_func = {b["func"]: b for b in out["behaviors"]}
        if "split_cells" in by_func:
            splits += [(k, uid) for uid in by_func["split_cells"]["cell_uids"]]
        if record is not None:
            grown = by_func.get("grow_cells", {}).get("cell_uids", [])
            record.append(([proc.phase_names[c] for c in proc.phase], set(grown)))
    return splits


def test_phase_names_and_rule_order():
    proc = _process()
    assert proc.phase_names == ["G1", "S", "M", "G2"]
    # Columns: G1, S (cyclin E high), M (cyclin A destroyed), G2 (cyclin A high).
    Y = np.zeros((6, 4))
    Y[2] = [0.5, 2.0, 0.5, 0.5]       # cyclin E
    Y[3] = [0.0, 0.0, 1.0, 3.0]       # cyclin A
    Y[4] = [0.0, 0.0, 2.0, 2.0]       # cyclin B
    assert [proc.phase_names[c] for c in proc.classify(Y)] == ["G1", "S", "M", "G2"]


def test_bad_phase_rules_raise():
    with pytest.raises(KeyError, match="cyclin_Z"):
        _process(phases=["S: cyclin_Z > 1"])
    with pytest.raises(ValueError, match="NAME: condition"):
        _process(phases=["cyclin_E_Cdk2 > 1"])
    with pytest.raises(ValueError, match="not one of"):
        _process(phases=["S: cyclin_E_Cdk2 > 1"])      # no M phase to divide in


def test_one_split_per_cycle_on_entering_m():
    proc = _process()
    record = []
    # 30 sim time units at time_scale 2.36 ~ 70.8 model units ~ 3 periods.
    splits = _run(proc, _faces(1), 600, record=record)
    assert [uid for _, uid in splits] == [0, 0, 0]
    for k, _ in splits:
        assert record[k][0][0] == "M" and record[k - 1][0][0] != "M"


def test_cells_grow_in_g1_and_g2_only():
    proc = _process()
    record = []
    _run(proc, _faces(1), 600, record=record)
    phases = np.array([r[0][0] for r in record])
    grown = np.array([0 in r[1] for r in record])
    assert set(phases[grown]) == {"G1", "G2"}
    assert not grown[np.isin(phases, ["S", "M"])].any()
    # G1 + G2 take ~44% of the Gérard–Goldbeter cycle.
    assert grown.mean() == pytest.approx(0.44, abs=0.05)


def test_split_behavior_carries_its_settings():
    proc = _process(birth_area=1.5, growth_rate=0.2, geom="SheetGeometry")
    out = None
    for k in range(600):
        out = proc.update({"datasets": {"face_df": _faces(1)}, "global_time": 0.05 * k}, 0.05)
        funcs = {b["func"]: b for b in out["behaviors"]}
        if "split_cells" in funcs:
            break
    assert funcs["split_cells"] == {"func": "split_cells", "geom": "SheetGeometry",
                                    "cell_uids": [0], "prefered_area": 1.5}
    assert proc.last_division[0] == pytest.approx(0.05 * (k + 1))
    proc2 = _process(growth_rate=0.2, scale_perimeter=False)
    for k in range(600):
        out = proc2.update({"datasets": {"face_df": _faces(1)}, "global_time": 0.05 * k}, 0.05)
        grow = [b for b in out["behaviors"] if b["func"] == "grow_cells"]
        if grow:
            break
    assert grow[0] == {"func": "grow_cells", "cell_uids": [0], "growth_rate": 0.2,
                       "dt": 0.05, "scale_perimeter": False}


def test_dividing_cells_are_skipped():
    proc = _process()
    record = []
    assert _run(proc, _faces(1, cell_type="dividing"), 600, record=record) == []
    assert proc.n_skipped == 3
    assert not any(r[1] for r in record)          # nor grown


def test_no_division_types_never_divide():
    proc = _process(no_division_types=["border"])
    faces = _faces(2).assign(cell_type=["border", "normal"])
    assert {uid for _, uid in _run(proc, faces, 600)} == {1}
    assert proc.n_skipped == 3


def test_no_growth_factor_arrests_after_committed_cycle():
    proc = _process(input_columns={"GF": "gf"}, parameters={"GF": 0.0})
    faces = _faces(4).assign(gf=0.0)
    # Each cell finishes the cycle it started in (one division), then arrests.
    assert len(_run(proc, faces, 300)) == 4
    assert _run(proc, faces, 600, t0=15.0) == []


def test_daughter_born_in_m_is_not_divided_again():
    proc = _process()
    faces = _faces(1)
    for k in range(600):
        out = proc.update({"datasets": {"face_df": faces}, "global_time": 0.05 * k}, 0.05)
        if any(b["func"] == "split_cells" for b in out["behaviors"]):
            break
    mother = proc.Y[:, 0].copy()
    daughter = faces.copy().assign(unique_id=7)
    for j, sp in enumerate(SPECIES):
        daughter[f"cc_{sp}"] = mother[j]
    faces2 = pd.concat([faces, daughter], ignore_index=True)
    out = proc.update({"datasets": {"face_df": faces2}, "global_time": 0.05 * (k + 1)}, 0.05)
    assert not any(b["func"] == "split_cells" for b in out["behaviors"])


def test_split_cells_daughter_takes_mothers_live_state():
    # The EventManager can run split_cells before the step's apply_gradient, so
    # the daughter's copied cc_* row may still hold the mother's pre-M state.
    proc = _process()
    faces = _faces(1)
    stale = None
    for k in range(600):
        stale = proc.Y[:, 0].copy() if proc.Y.size else None
        out = proc.update({"datasets": {"face_df": faces}, "global_time": 0.05 * k}, 0.05)
        if any(b["func"] == "split_cells" for b in out["behaviors"]):
            break
    assert proc.phase_names[proc.classify(stale)[0]] != "M"
    mother = proc.Y[:, 0].copy()
    daughter = faces.copy().assign(unique_id=7, mother_uid=0)
    for j, sp in enumerate(SPECIES):
        daughter[f"cc_{sp}"] = stale[j]
    faces2 = pd.concat([faces.assign(mother_uid=-1), daughter], ignore_index=True)
    proc._sync(faces2, 0.05 * (k + 1))
    np.testing.assert_allclose(proc.Y[:, 1], mother)
    assert proc.phase_names[proc.phase[1]] == "M"
    assert proc.last_division[1] == proc.last_division[0]
    out = proc.update({"datasets": {"face_df": faces2}, "global_time": 0.05 * (k + 1)}, 0.05)
    assert not any(b["func"] == "split_cells" for b in out["behaviors"])


def test_seeded_cells_start_at_the_size_their_phase_implies():
    np.random.seed(3)
    proc = _process(initial_state="limit_cycle", time_scale=1.18,
                    growth_rate=np.log(2.0) / (0.44 * 20.0))
    faces = _faces(300).assign(prefered_area=1.0, prefered_perimeter=3.5)
    out = proc.update({"datasets": {"face_df": faces}, "global_time": 0.0}, 0.002)
    upd = [b for b in out["behaviors"] if b["func"] == "apply_gradient"][0]["parameter_updates"]
    area = pd.Series(upd["prefered_area"]["update"]).reindex(faces["unique_id"]).fillna(1.0)
    perim = pd.Series(upd["prefered_perimeter"]["update"]).reindex(faces["unique_id"]).fillna(3.5)
    phase = np.array([proc.phase_names[c] for c in proc.phase])
    np.testing.assert_allclose(perim, 3.5 * np.sqrt(area), rtol=1e-12)
    # A cell in M has already split on entering it, so it is a newborn half. G1
    # grows part-way, S holds G1's growth (~2^(5.8/10.3)), G2 grows on toward 2.
    assert area[phase == "M"].to_numpy() == pytest.approx(1.0)
    assert area[phase == "S"].to_numpy() == pytest.approx(1.48, abs=0.05)
    assert area[phase == "G1"].between(0.99, 1.5).all()
    assert area[phase == "G2"].between(1.45, 2.01).all()
    # Applied once, at seeding.
    out = proc.update({"datasets": {"face_df": faces}, "global_time": 0.002}, 0.002)
    upd = [b for b in out["behaviors"] if b["func"] == "apply_gradient"][0]["parameter_updates"]
    assert "prefered_area" not in upd


def test_ratio_input_column():
    # GF from strain = area / prefered_area: a growing cell (inflated prefered_area)
    # reads low strain, a stretched one reads high.
    proc = _process(input_columns={"GF": "area/prefered_area"})
    faces = _faces(3).assign(area=[2.0, 2.0, 0.5], prefered_area=[1.0, 2.0, 1.0])
    proc._sync(faces, 0.0)
    np.testing.assert_allclose(proc._params(faces, proc.factors)["GF"], [2.0, 1.0, 0.5])


def test_state_columns_are_written():
    proc = _process()
    out = proc.update({"datasets": {"face_df": _faces(3)}, "global_time": 0.0}, 0.05)
    grad = [b for b in out["behaviors"] if b["func"] == "apply_gradient"][0]
    cols = grad["parameter_updates"]
    assert set(cols) == {f"cc_{s}" for s in SPECIES} | {"cc_phase", "cc_last_division"}
    assert set(cols["cc_cyclin_B_Cdk1"]["update"]) == {0, 1, 2}


def test_new_uid_inherits_and_vanished_uid_is_dropped():
    proc = _process()
    faces = _faces(2)
    _run(proc, faces, 10)
    mother = proc.Y[:, 0].copy()
    # A daughter (uid 7) carrying the mother's state columns; uid 1 has gone.
    daughter = faces.iloc[[0]].copy()
    daughter["unique_id"] = 7
    for j, s in enumerate(SPECIES):
        daughter[f"cc_{s}"] = mother[j]
    faces2 = pd.concat([faces.iloc[[0]], daughter], ignore_index=True)
    proc._sync(faces2, 0.5)
    assert proc.uids.tolist() == [0, 7]
    np.testing.assert_allclose(proc.Y[:, 1], mother)


def test_daughter_inherits_through_tyssue_division():
    tyssue = pytest.importorskip("tyssue")
    from tyssue.generation import three_faces_sheet
    from tyssue.topology.sheet_topology import cell_division
    from vivarium_tyssue.behaviors.behaviors import apply_gradient

    sheet = tyssue.Sheet("t", *three_faces_sheet())
    tyssue.SheetGeometry.update_all(sheet)
    for df in sheet.datasets.values():
        df["unique_id"] = np.arange(len(df))
    sheet.face_df["cell_type"] = "normal"

    proc = _process()
    for k in range(5):
        out = proc.update({"datasets": {"face_df": sheet.face_df}, "global_time": 0.05 * k}, 0.05)
        for b in out["behaviors"]:
            if b["func"] == "apply_gradient":
                apply_gradient(sheet, None, b["parameter_updates"])
    mother = proc.Y[:, 0].copy()

    daughter = cell_division(sheet, 0, tyssue.SheetGeometry)
    sheet.face_df.loc[daughter, "unique_id"] = 99
    proc._sync(sheet.face_df, 0.3)
    i = proc.uids.tolist().index(99)
    np.testing.assert_allclose(proc.Y[:, i], mother)
