"""Render each study's declared visualizations to studies/<name>/viz/<viz>.html.

The dashboard's render_visualizations() injects run data via input-wiring
(gather_emitter_outputs), but our visualizations are Path-C (they read runs.db
directly via the `_runs_db_path` config key), so that path produces "No run
data" placeholders. This renderer sets `_runs_db_path` (+ `_study_yaml_path`)
and calls each viz's render directly, writing the HTML the report embeds.

Usage: python scripts/render_study_viz.py [study-slug ...]   (default: all)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml


def _slug(name: str) -> str:
    """Filename slug — the dashboard's static server 404s filenames with spaces,
    so the report can't fetch+inline them. Keep it ASCII/space-free."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return s.strip("_") or "viz"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STUDIES = ROOT / "workspace" / "studies"


def _with_caption(html: str, caption: str) -> str:
    """Append a reviewer-facing explanatory note below a rendered viz.

    The dashboard embeds these viz/*.html files verbatim into the investigation
    report (server.py Source 1), so an explanatory <div> here travels with the
    chart — the place to answer 'explain better what is going on'."""
    if not caption:
        return html
    return (
        html
        + '<div style="font-family:system-ui;font-size:0.85rem;line-height:1.4;'
        'color:#374151;background:#f9fafb;border-left:3px solid #6366f1;'
        'padding:0.6rem 0.8rem;margin-top:0.5rem;border-radius:0 4px 4px 0">'
        f"{caption}</div>"
    )


def _render_one(address: str, config: dict, runs_db: str, study_yaml: str) -> str:
    cfg = dict(config or {})
    cfg["_runs_db_path"] = runs_db
    cfg["_study_yaml_path"] = study_yaml
    caption = cfg.get("caption") or ""
    # A study's runs.db may hold several simulations; sources[0] selects which one
    # the mesh viz (snapshots/gif) should read. Timeseries handles sources itself.
    source = (cfg.get("sources") or [None])[0]
    name = address.split(":", 1)[-1]
    if name == "TissueSheetSnapshots":
        from vivarium_tyssue.visualizations.tissue_snapshots import _panels_html, _load_frames
        frames = _load_frames(runs_db, source)
        html = _panels_html(frames, list(cfg.get("coords") or ["x", "y"]),
                            int(cfg.get("n_panels") or 6), cfg.get("title") or "snapshots")
        return _with_caption(html, caption)
    if name == "TissueSheetGif":
        from vivarium_tyssue.visualizations.tissue_gif import TissueSheetGif
        v = TissueSheetGif.__new__(TissueSheetGif)
        v.config = cfg
        return _with_caption(v._render_html(), caption)
    if name == "TimeSeriesFromObservables":
        from pbg_superpowers.visualizations.timeseries_from_observables import _render_html
        return _with_caption(_render_html(cfg), caption)
    if name in ("TumorCloneGrowth", "CellAreaOverTime"):
        import vivarium_tyssue.visualizations.tumor_metrics as tm
        v = getattr(tm, name).__new__(getattr(tm, name))
        v.config = cfg
        return _with_caption(v._render_html(), caption)
    return f'<div>unsupported viz address: {address}</div>'


def render_study(slug: str) -> None:
    sdir = STUDIES / slug
    spec = yaml.safe_load((sdir / "study.yaml").read_text())
    runs_db = str(sdir / "runs.db")
    study_yaml = str(sdir / "study.yaml")
    viz_dir = sdir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for old in viz_dir.glob("*.html"):  # clear stale (incl. space-named) files
        old.unlink()
    n = 0
    for v in (spec.get("visualizations") or []):
        if not isinstance(v, dict) or not v.get("name"):
            continue
        html = _render_one(v.get("address", ""), v.get("config", {}), runs_db, study_yaml)
        (viz_dir / f"{_slug(v['name'])}.html").write_text(html, encoding="utf-8")
        ok = ("data:image" in html) or ("Plotly.newPlot" in html)
        print(f"  {v['name']}: {'OK' if ok else 'placeholder'} ({len(html)} bytes)")
        n += 1
    print(f"{slug}: rendered {n} viz -> {viz_dir}")


if __name__ == "__main__":
    slugs = sys.argv[1:] or [p.name for p in sorted(STUDIES.iterdir()) if (p / "study.yaml").is_file()]
    for s in slugs:
        render_study(s)
