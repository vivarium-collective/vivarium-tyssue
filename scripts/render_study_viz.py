"""Render each study's declared visualizations to studies/<name>/viz/<viz>.html.

The dashboard's render_visualizations() injects run data via input-wiring
(gather_emitter_outputs), but our visualizations are Path-C (they read runs.db
directly via the `_runs_db_path` config key), so that path produces "No run
data" placeholders. This renderer sets `_runs_db_path` (+ `_study_yaml_path`)
and calls each viz's render directly, writing the HTML the report embeds.

Usage: python scripts/render_study_viz.py [study-slug ...]   (default: all)
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STUDIES = ROOT / "workspace" / "studies"


def _render_one(address: str, config: dict, runs_db: str, study_yaml: str) -> str:
    cfg = dict(config or {})
    cfg["_runs_db_path"] = runs_db
    cfg["_study_yaml_path"] = study_yaml
    name = address.split(":", 1)[-1]
    if name == "TissueSheetSnapshots":
        from vivarium_tyssue.visualizations.tissue_snapshots import _panels_html, _load_frames
        frames = _load_frames(runs_db)
        return _panels_html(frames, list(cfg.get("coords") or ["x", "y"]),
                            int(cfg.get("n_panels") or 6), cfg.get("title") or "snapshots")
    if name == "TissueSheetGif":
        from vivarium_tyssue.visualizations.tissue_gif import TissueSheetGif
        v = TissueSheetGif.__new__(TissueSheetGif)
        v.config = cfg
        return v._render_html()
    if name == "TimeSeriesFromObservables":
        from pbg_superpowers.visualizations.timeseries_from_observables import _render_html
        return _render_html(cfg)
    return f'<div>unsupported viz address: {address}</div>'


def render_study(slug: str) -> None:
    sdir = STUDIES / slug
    spec = yaml.safe_load((sdir / "study.yaml").read_text())
    runs_db = str(sdir / "runs.db")
    study_yaml = str(sdir / "study.yaml")
    viz_dir = sdir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for v in (spec.get("visualizations") or []):
        if not isinstance(v, dict) or not v.get("name"):
            continue
        html = _render_one(v.get("address", ""), v.get("config", {}), runs_db, study_yaml)
        (viz_dir / f"{v['name']}.html").write_text(html, encoding="utf-8")
        ok = ("data:image" in html) or ("Plotly.newPlot" in html)
        print(f"  {v['name']}: {'OK' if ok else 'placeholder'} ({len(html)} bytes)")
        n += 1
    print(f"{slug}: rendered {n} viz -> {viz_dir}")


if __name__ == "__main__":
    slugs = sys.argv[1:] or [p.name for p in sorted(STUDIES.iterdir()) if (p / "study.yaml").is_file()]
    for s in slugs:
        render_study(s)
