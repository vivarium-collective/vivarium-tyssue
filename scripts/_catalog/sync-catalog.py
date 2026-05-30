"""Regenerate scripts/_catalog/modules.json from GitHub.

Run from pbg-template repo root:

    python3 scripts/_catalog/sync-catalog.py

Pulls every `pbg-*` repo on vivarium-collective (excluding pbg-superpowers
and pbg-template, which aren't simulation modules), plus the explicitly
listed extras (v2ecoli, spatio-flux). Re-emits modules.json sorted by name.

Requires `gh` CLI authenticated against GitHub.
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path


ORG = "vivarium-collective"
# Non-pbg-* repos that are genuine process-bigraph extensions (Processes,
# Steps, Types) workspaces install via pyproject + import in composites.
# Listed here because the auto-pull filter is `name.startswith("pbg-")`;
# extension packages without that prefix would otherwise be invisible to
# the catalog. Keep this list tight — workspace-y repos (vEcoli*,
# multiscale-bioprocess, etc.) and infrastructure (bigraph-*, sms-*)
# do NOT belong here.
EXTRAS = ["v2ecoli", "spatio-flux", "Viva-munk"]
EXCLUDE = {"pbg-superpowers", "pbg-template"}


def _gh_list_org(org: str) -> list[dict]:
    r = subprocess.run(
        ["gh", "repo", "list", org, "--limit", "200",
         "--json", "name,description,url,defaultBranchRef"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(r.stdout)


def _package_name(repo_name: str) -> str:
    """pbg-cellpack → pbg_cellpack; spatio-flux → spatio_flux; v2ecoli → v2ecoli."""
    return repo_name.replace("-", "_")


_TAG_HINTS = (
    ("whole-cell", "whole-cell"),
    ("ecoli",      "ecoli"),
    ("microbial",  "microbial"),
    ("spatial",    "spatial"),
    ("agent-based", "agent-based"),
    ("multicellular", "multicellular"),
    ("morphogenesis", "morphogenesis"),
    ("cytoskeleton", "cytoskeleton"),
    ("membrane",   "membrane"),
    ("vesicle",    "membrane"),
    ("micelle",    "membrane"),
    ("particle",   "particle"),
    ("reaction-diffusion", "reaction-diffusion"),
    ("rule-based", "rule-based"),
    ("metabolism", "metabolism"),
    ("metabolic",  "metabolism"),
    ("fba",        "fba"),
    ("stochastic", "stochastic"),
    ("sbml",       "sbml"),
    ("antimony",   "sbml"),
    ("ode",        "ode"),
    ("kinetics",   "kinetics"),
    ("kinetic",    "kinetics"),
    ("coarse-grained", "coarse-grained"),
    ("mesoscale",  "mesoscale"),
    ("pde",        "pde"),
    ("finite-volume", "pde"),
    ("cfd",        "cfd"),
    ("bioreactor", "bioreactor"),
    ("lammps",     "lammps"),
    ("bonds",      "md"),
    ("md",         "md"),
    ("molecular dynamics", "md"),
    ("composite",  "composite"),
    ("packing",    "structure"),
    ("nfsim",      "rule-based"),
    ("bionetgen",  "rule-based"),
    ("compucell",  "agent-based"),
    ("v2ecoli",    "ecoli"),
    ("vcell",      "pde"),
    ("yalla",      "agent-based"),
    ("smoldyn",    "stochastic"),
    ("medyan",     "cytoskeleton"),
    ("readdy",     "particle"),
    ("flux",       "metabolism"),
)


def _infer_tags(name: str, description: str | None) -> list[str]:
    """Match tag hints against name + description, using word boundaries to
    avoid false positives (e.g. 'mODEl' wrongly matching 'ode')."""
    text = f"{name} {description or ''}".lower()
    tags: list[str] = []
    for hint, tag in _TAG_HINTS:
        if re.search(rf"\b{re.escape(hint)}\b", text) and tag not in tags:
            tags.append(tag)
    return tags


def _entry(repo: dict) -> dict:
    name = repo["name"]
    ref = (repo.get("defaultBranchRef") or {}).get("name") or "main"
    return {
        "name": name,
        "description": (repo.get("description") or "").strip(),
        "source": f"https://github.com/{ORG}/{name}.git",
        "ref": ref,
        "package": _package_name(name),
        "homepage": repo.get("url") or f"https://github.com/{ORG}/{name}",
        "tags": _infer_tags(name, repo.get("description")),
    }


# Fields that this script owns — it overwrites them on every run from the
# GitHub metadata. Any OTHER field on an existing entry is curator-owned
# (e.g. hand-authored `checks:` blocks for runtime dependency validation,
# like the OpenMPI + lammps-importable probes on pbg-lammps) and must be
# preserved across a re-sync — otherwise running this script eats curation.
_AUTO_KEYS = frozenset({
    "name", "description", "source", "ref", "package", "homepage", "tags",
})


def _merge_extras(fresh: dict, existing: dict | None) -> dict:
    """Return `fresh` augmented with any non-auto fields from `existing`.

    Auto fields (the ones derived from GitHub on each run) always come from
    `fresh`. Curator-authored fields like `checks` survive a re-sync. New
    entries (no existing) pass through unchanged.
    """
    if not existing:
        return fresh
    out = dict(fresh)
    for key, value in existing.items():
        if key not in _AUTO_KEYS:
            out[key] = value
    return out


def main() -> int:
    repos = _gh_list_org(ORG)
    by_name = {r["name"]: r for r in repos}
    selected: list[dict] = []
    for r in repos:
        nm = r["name"]
        if nm in EXCLUDE:
            continue
        if nm.startswith("pbg-"):
            selected.append(r)
    for nm in EXTRAS:
        if nm in by_name:
            selected.append(by_name[nm])
        else:
            print(f"warning: '{nm}' not found on {ORG}", file=sys.stderr)
    selected.sort(key=lambda r: r["name"])

    out_path = Path(__file__).parent / "modules.json"
    existing_by_name: dict[str, dict] = {}
    if out_path.is_file():
        try:
            for e in json.loads(out_path.read_text()):
                if isinstance(e, dict) and e.get("name"):
                    existing_by_name[e["name"]] = e
        except (json.JSONDecodeError, OSError):
            pass

    out = [
        _merge_extras(_entry(r), existing_by_name.get(r["name"]))
        for r in selected
    ]

    # Surface any preserved-extras so the operator running the sync can
    # eyeball that nothing meaningful got dropped on a stale-entry removal.
    preserved = [
        e["name"] for e in out
        if any(k not in _AUTO_KEYS for k in e)
    ]
    dropped_stale = sorted(set(existing_by_name) - {r["name"] for r in selected})

    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {len(out)} entries to {out_path}")
    if preserved:
        print(f"  preserved curator-authored extras on: {', '.join(sorted(preserved))}")
    if dropped_stale:
        print(f"  WARNING: dropped entries no longer on {ORG}: {', '.join(dropped_stale)}")
        for nm in dropped_stale:
            e = existing_by_name[nm]
            extras = [k for k in e if k not in _AUTO_KEYS]
            if extras:
                print(f"    {nm}: had curator-authored fields {extras} — verify they're not needed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
