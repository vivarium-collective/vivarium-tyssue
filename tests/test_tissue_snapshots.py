from vivarium_tyssue.visualizations.tissue_snapshots import _panels_html, _cell_type_color


def test_color_lookup_falls_back():
    assert _cell_type_color("tumor").startswith("#")
    assert _cell_type_color("unknown-type").startswith("#")  # default, no crash


def test_panels_html_handles_empty_frames():
    html = _panels_html([], coords=["x", "y"], n_panels=4, title="t")
    assert "No snapshots" in html


def test_panels_html_emits_figure_for_frames():
    # one frame with two triangular faces via edge source/target coords
    frame = {"t": 0.0, "datasets": {"face_df": {"unique_id": [0], "cell_type": ["tumor"]},
             "edge_df": {"face": [0, 0, 0], "sx": [0.0, 1.0, 0.5], "sy": [0.0, 0.0, 1.0],
                         "tx": [1.0, 0.5, 0.0], "ty": [0.0, 1.0, 0.0]}}}
    html = _panels_html([frame], coords=["x", "y"], n_panels=1, title="snap")
    assert "data:image/png;base64," in html
