from vivarium_tyssue.visualizations.tumor_metrics import (
    _frame_series, TumorCloneGrowth, CellAreaOverTime,
)


def _frame(t, types, xs, ys, areas):
    return {"t": t, "datasets": {"face_df": {
        "cell_type": types, "x": xs, "y": ys, "area": areas,
        "unique_id": list(range(len(types))),
    }}}


def _frames():
    # t=0: a single tumor cell at the origin amid healthy cells.
    # t=1: the tumor clone has grown to 3 contiguous cells (spread radius > 0).
    return [
        _frame(0.0, ["tumor", "healthy", "healthy"], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0],
               [1.0, 1.0, 1.0]),
        _frame(1.0, ["tumor", "tumor", "tumor", "healthy"],
               [0.0, 0.5, -0.5, 2.0], [0.0, 0.3, 0.3, 0.0], [0.9, 0.8, 0.8, 1.0]),
    ]


def test_frame_series_tracks_count_and_spread():
    m = _frame_series(_frames())
    assert m["n_tumor"] == [1, 3]
    assert m["n_total"] == [3, 4]
    # one tumor cell -> zero spread; three spread cells -> positive radius
    assert m["clone_radius"][0] == 0.0
    assert m["clone_radius"][1] > 0.0
    # median areas stay near the prefered area (no collapse toward zero)
    assert 0.5 < m["tumor_med_area"][1] < 1.1
    assert m["p05_area"][1] > 0.0


def test_frame_series_empty_is_safe():
    assert _frame_series([])["t"] == []
    assert _frame_series([{"t": 0.0, "datasets": {}}])["t"] == []


def test_clone_growth_renders_figure():
    v = TumorCloneGrowth.__new__(TumorCloneGrowth)
    v.config = {}  # no runs.db -> empty-state html, must not crash
    assert "No clone-growth data" in v._render_html()


def test_cell_area_renders_figure_from_frames(monkeypatch):
    import vivarium_tyssue.visualizations.tumor_metrics as tm
    monkeypatch.setattr(tm, "_load_frames", lambda *a, **k: _frames())
    v = CellAreaOverTime.__new__(CellAreaOverTime)
    v.config = {"_runs_db_path": "x", "sources": ["tumor"]}
    assert "data:image/png;base64," in v._render_html()
