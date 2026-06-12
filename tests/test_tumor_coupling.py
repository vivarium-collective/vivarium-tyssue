from vivarium_tyssue.processes.tumor_coupling import fractional_events


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
