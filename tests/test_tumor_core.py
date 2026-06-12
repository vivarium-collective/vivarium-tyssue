from vivarium_tyssue.core import build_core


def test_copasi_process_registered():
    core = build_core()
    assert "CopasiUTCProcess" in core.link_registry


def test_timeseries_from_observables_registered():
    # Shipped by pbg-superpowers; build_core should surface it for studies.
    core = build_core()
    assert "TimeSeriesFromObservables" in core.link_registry


def test_tumor_coupling_registered():
    core = build_core()
    assert "TumorCoupling" in core.link_registry
