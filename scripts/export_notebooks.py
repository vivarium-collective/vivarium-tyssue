"""Regenerate the cached reproduction notebook for each investigation.

The dashboard's notebook_export turns an investigation into a self-contained
``.ipynb`` (+ ``.py``) that re-runs every study via the process-bigraph protocol
and renders its figures. We commit those notebooks under
``workspace/reports/notebooks/`` so a fresh clone has them ready to run (no
sharing from Downloads, no path editing — the notebook locates its own repo root
and ``RERUN=True`` regenerates results from the in-repo composite specs and
datasets). Run this after changing a composite, study, or investigation to keep
the cached notebooks up to date.

Usage: python scripts/export_notebooks.py [investigation-slug ...]   (default: all)
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from vivarium_dashboard.lib.notebook_export import export_investigation_notebook

ROOT = Path(__file__).resolve().parents[1]
INV_DIR = ROOT / "workspace" / "investigations"


def _all_slugs() -> list[str]:
    return sorted(
        p.parent.name
        for p in INV_DIR.glob("*/investigation.yaml")
    )


def main(argv: list[str]) -> int:
    slugs = argv or _all_slugs()
    if not slugs:
        print("no investigations found under", INV_DIR)
        return 1
    for slug in slugs:
        inv_yaml = INV_DIR / slug / "investigation.yaml"
        if not inv_yaml.is_file():
            print(f"skip {slug}: no investigation.yaml")
            continue
        # sanity: the slug should match the file's declared name
        name = (yaml.safe_load(inv_yaml.read_text(encoding="utf-8")) or {}).get("name")
        if name and name != slug:
            print(f"warn {slug}: investigation.yaml name is {name!r}")
        paths = export_investigation_notebook(ROOT, slug)
        print("wrote", paths["ipynb"].relative_to(ROOT))
        print("wrote", paths["py"].relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
