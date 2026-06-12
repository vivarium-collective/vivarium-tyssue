import numpy as np
from vivarium_tyssue.processes.tumor_coupling import fractional_events, select_uids, TumorCoupling
from vivarium_tyssue.core import build_core


def test_accumulator_fires_when_crossing_one():
    # flux 4.0, scale 0.1, dt 1.0 -> 0.4 per step; needs 3 steps to fire once.
    acc = {"x": 0.0}
    n0, acc = fractional_events({"x": 4.0}, {"x": 0.1}, acc, dt=1.0)
    assert n0["x"] == 0 and abs(acc["x"] - 0.4) < 1e-9
    n1, acc = fractional_events({"x": 4.0}, {"x": 0.1}, acc, dt=1.0)
    assert n1["x"] == 0 and abs(acc["x"] - 0.8) < 1e-9
    n2, acc = fractional_events({"x": 4.0}, {"x": 0.1}, acc, dt=1.0)
    assert n2["x"] == 1 and abs(acc["x"] - 0.2) < 1e-9  # 1.2 -> fire 1, carry 0.2


def test_no_event_lost_to_rounding():
    acc = {"x": 0.0}
    total = 0
    for _ in range(100):
        n, acc = fractional_events({"x": 10.0}, {"x": 0.1}, acc, dt=1.0)
        total += n["x"]
    # 100 steps * 1.0 expected events = 100, give or take the carried remainder (<1).
    assert total in (99, 100)


def test_negative_and_missing_flux_clamped_to_zero():
    n, acc = fractional_events({"x": -5.0}, {"x": 0.1}, {"x": 0.0}, dt=1.0)
    assert n["x"] == 0 and acc["x"] == 0.0
    n, acc = fractional_events({}, {"x": 0.1}, {"x": 0.0}, dt=1.0)
    assert n["x"] == 0


def _face_df(uids_types):
    # Minimal face_df-as-dict-of-lists, like the emitted tyssue datasets.
    return {
        "unique_id": [u for u, _ in uids_types],
        "cell_type": [t for _, t in uids_types],
    }


def test_select_uids_picks_matching_type():
    fdf = _face_df([(1, "tumor"), (2, "healthy"), (3, "tumor")])
    picked = select_uids(fdf, "tumor", 2, exclude=set(), rng_pick=lambda items, k: items[:k])
    assert set(picked) == {1, 3}


def test_select_uids_caps_to_available():
    fdf = _face_df([(1, "tumor")])
    picked = select_uids(fdf, "tumor", 5, exclude=set(), rng_pick=lambda items, k: items[:k])
    assert picked == [1]  # capped


def test_select_uids_excludes_already_chosen():
    fdf = _face_df([(1, "healthy"), (2, "healthy")])
    picked = select_uids(fdf, "healthy", 1, exclude={1}, rng_pick=lambda items, k: items[:k])
    assert picked == [2]


def _make_proc(seed_tumor=0, seed_stem=0, scales=None):
    config = {
        "birth_fluxes": {"tumor": "T_birth", "healthy": "H_birth", "stem": "C_birth"},
        "death_fluxes": {"tumor": "T_death", "healthy": "H_death", "stem": "C_death"},
        "scales": scales or {
            "tumor_births": 1.0, "tumor_deaths": 1.0,
            "healthy_births": 1.0, "healthy_deaths": 1.0,
            "stem_births": 1.0, "stem_deaths": 1.0,
        },
        "geom": "SheetGeometry",
        "dt": 1.0, "growth_rate": 0.1, "shrink_rate": 0.1,
        "division_crit": 0.5, "apoptosis_crit": 2.0,
        "seed": {"tumor": seed_tumor, "stem": seed_stem},
    }
    return TumorCoupling(config=config, core=build_core())


def _datasets(types):
    return {"face_df": {"unique_id": list(range(len(types))), "cell_type": list(types)}}


def test_seed_step_differentiates_focus():
    proc = _make_proc(seed_tumor=2, seed_stem=1)
    ds = _datasets(["healthy"] * 10)
    out = proc.update({"fluxes": {}, "datasets": ds, "global_time": 0.0}, 1.0)
    funcs = [b["func"] for b in out["behaviors"]]
    new_types = [b["new_type"] for b in out["behaviors"]]
    assert funcs == ["differentiation"] * 3
    assert new_types.count("tumor") == 2 and new_types.count("stem") == 1


def test_flux_step_relabel_default():
    # Default (fixed-topology fate) mode: tumor birth relabels a stem cell -> tumor,
    # healthy death relabels a healthy cell -> dead. All behaviors are differentiations.
    np.random.seed(0)
    proc = _make_proc()
    proc._seeded = True  # skip seeding
    ds = _datasets(["tumor", "tumor", "healthy", "healthy", "stem"])
    fluxes = {"T_birth": 1.0, "H_death": 1.0, "T_death": 0.0,
              "H_birth": 0.0, "C_birth": 0.0, "C_death": 0.0}
    out = proc.update({"fluxes": fluxes, "datasets": ds, "global_time": 5.0}, 1.0)
    funcs = [b["func"] for b in out["behaviors"]]
    new_types = sorted(b["new_type"] for b in out["behaviors"])
    assert all(f == "differentiation" for f in funcs)
    assert new_types == ["dead", "tumor"]  # healthy->dead, stem->tumor (C->T)
    assert out["healthy_deaths"] == 1.0 and out["tumor_births"] == 1.0
    assert out["tumor_count"] == 2.0 and out["healthy_count"] == 2.0 and out["dead_count"] == 0.0


def test_flux_step_topology_mode_fires_real_ops():
    # topology_ops=True uses the real tyssue cell_division / remove_face behaviors.
    np.random.seed(0)
    proc = _make_proc()
    proc.topology_ops = True
    proc._seeded = True
    ds = _datasets(["tumor", "tumor", "healthy", "healthy", "stem"])
    fluxes = {"T_birth": 1.0, "H_death": 1.0, "T_death": 0.0,
              "H_birth": 0.0, "C_birth": 0.0, "C_death": 0.0}
    out = proc.update({"fluxes": fluxes, "datasets": ds, "global_time": 5.0}, 1.0)
    funcs = sorted(b["func"] for b in out["behaviors"])
    assert "apoptosis_extrusion" in funcs  # real extrusion (remove_face)
    assert out["healthy_deaths"] == 1.0 and out["tumor_births"] == 1.0
    assert out["tumor_count"] == 2.0 and out["healthy_count"] == 2.0


def test_counts_reported_each_step():
    proc = _make_proc()
    proc._seeded = True
    ds = _datasets(["tumor", "healthy", "healthy", "stem"])
    out = proc.update({"fluxes": {}, "datasets": ds, "global_time": 3.0}, 1.0)
    assert out["tumor_count"] == 1.0 and out["healthy_count"] == 2.0 and out["stem_count"] == 1.0
