"""One command: run sims -> render viz -> emit a self-contained study report.

Chains the three steps that turn a study spec into a reviewer-ready, offline HTML
report with the visualizations inlined:

  1. run_study_sims  — step each baseline composite, write studies/<slug>/runs.db
  2. render_study_viz — render each declared viz to studies/<slug>/viz/*.html
  3. render_single_study_report — assemble the self-contained HTML (iframe srcdoc)

The report lands wherever --out points (default: ~/Downloads). This is the
"pbg-report, but it drops a self-contained file in Downloads" flow.

Usage:
  python scripts/build_report.py                       # all studies, 60 steps -> ~/Downloads
  python scripts/build_report.py tumor-composite       # one study
  python scripts/build_report.py --steps 100 --out .   # more steps, custom dir
  python scripts/build_report.py --no-sim              # reuse existing runs.db
"""
from __future__ import annotations

import os
import sys

# Re-exec once under Python UTF-8 mode so `python scripts/build_report.py` works
# without a PYTHONUTF8=1 prefix (COPASI/basico resets the locale; YAML reads need
# UTF-8). Guarded by a sentinel env var to avoid an exec loop.
if not sys.flags.utf8_mode and os.environ.get("_BUILD_REPORT_UTF8") != "1":
    os.environ["PYTHONUTF8"] = "1"
    os.environ["_BUILD_REPORT_UTF8"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_study_sims  # noqa: E402
import render_study_viz  # noqa: E402
from vivarium_dashboard.lib.single_study_report import render_single_study_report  # noqa: E402

STUDIES = ROOT / "workspace" / "studies"


def _composite_yaml(spec_id: str) -> Path:
    """Dotted composite id -> source YAML, e.g.
    'vivarium_tyssue.composites.tumor' -> vivarium_tyssue/composites/tumor.composite.yaml."""
    pkg, stem = spec_id.rsplit(".", 1)
    return ROOT / (pkg.replace(".", "/") + f"/{stem}.composite.yaml")


def build(slug: str, *, steps: int, interval: float, out_dir: Path,
          investigation: str, do_sim: bool) -> Path:
    sdir = STUDIES / slug
    spec = yaml.safe_load((sdir / "study.yaml").read_text())
    if do_sim:
        for b in (spec.get("baseline") or []):
            cy = _composite_yaml(b["composite"])
            if not cy.is_file():
                print(f"  ! skipping {b.get('name')}: composite YAML not found ({cy})")
                continue
            run_study_sims.run_study(slug, b["name"], str(cy), steps, interval)
    render_study_viz.render_study(slug)
    p = render_single_study_report(ROOT, slug, investigation_slug=investigation, out_dir=out_dir)
    print(f"report: {p}  ({p.stat().st_size // 1024} KB)")
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("studies", nargs="*", help="study slugs (default: all)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--interval", type=float, default=0.1)
    ap.add_argument("--out", default=str(Path.home() / "Downloads"),
                    help="output directory for the report HTML (default: ~/Downloads)")
    ap.add_argument("--investigation", default="tumor-tyssue")
    ap.add_argument("--no-sim", action="store_true",
                    help="skip the simulation step and reuse existing runs.db")
    a = ap.parse_args()

    slugs = a.studies or [p.name for p in sorted(STUDIES.iterdir())
                          if (p / "study.yaml").is_file()]
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for s in slugs:
        print(f"=== {s} ===")
        build(s, steps=a.steps, interval=a.interval, out_dir=out,
              investigation=a.investigation, do_sim=not a.no_sim)


if __name__ == "__main__":
    main()
