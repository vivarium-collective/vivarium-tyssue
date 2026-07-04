#!/usr/bin/env python
"""Track tyssue-backend speed + cell throughput across geometries over time.

Measures the clean per-``EulerSolver.update()`` cost (driving the process
directly with no behaviors, so it's independent of the run()/interval sub-step
multiplier and of the behavior processes) for each geometry family, on the
python and rust backends, and appends a record to ``benchmarks/results.jsonl``
tagged with the current git commit. Re-run after each optimization to watch the
numbers move; the printed table shows the delta vs the last recorded commit.

    .venv/bin/python scripts/bench_backends.py --label "what changed"

Cell-count scaling *within* a geometry (a cells-vs-time curve) needs
variable-size mesh generation, which is currently blocked by a pandas-3
`include_groups` bug in tyssue's sanitize(); until that's fixed we track the
three fixed fixtures (different cell counts across geometries).
"""
import argparse
import contextlib
import io
import json
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "benchmarks" / "results.jsonl"
SUMMARY = ROOT / "benchmarks" / "SUMMARY.md"

# geometry family -> (representative composite, backends to try)
CONFIGS = [
    ("sheet", "anisotropic", ["python", "rust"]),        # SheetGeometry, 3-effector
    ("vessel", "base_solver", ["python"]),               # VesselGeometry (4-effector: rust N/A)
    ("monolayer", "monolayer_liftoff", ["python"]),      # MonolayerGeometry / bulk (rust N/A)
]


def git_commit():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip()
        dirty = (
            subprocess.call(["git", "diff", "--quiet"], cwd=ROOT) != 0
            or subprocess.call(["git", "diff", "--cached", "--quiet"], cwd=ROOT) != 0
        )
        return h + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def bench_one(composite, backend, interval, warmup, updates):
    from pbg_superpowers.composite_spec import build_composite_from_spec, load_spec
    from vivarium_tyssue.core import build_core

    spec = load_spec(ROOT / "vivarium_tyssue" / "composites" / f"{composite}.composite.yaml")
    spec["emitters"] = []
    spec["state"]["Tyssue"]["config"]["backend"] = backend
    comp = build_composite_from_spec(spec, overrides={"interval": interval}, core=build_core())
    proc = comp.state["Tyssue"]["instance"]
    if backend == "rust" and not getattr(proc, "_rust_gradient", False):
        return None  # kernel not built or model unsupported — skip, don't fake it
    eptm = proc.eptm
    times = []
    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(warmup):
            proc.update({"behaviors": [], "global_time": i * interval}, interval)
        for i in range(updates):
            t = time.perf_counter()
            proc.update({"behaviors": [], "global_time": (warmup + i) * interval}, interval)
            times.append(time.perf_counter() - t)
    ms = statistics.median(times) * 1e3
    return {
        "cells": int(eptm.Nf),
        "verts": int(eptm.Nv),
        "edges": int(eptm.Ne),
        "per_update_ms": round(ms, 4),
        "updates_per_sec": round(1000 / ms, 1),
        "cell_updates_per_sec": round(eptm.Nf * 1000 / ms),
    }


def load_prior():
    if not RESULTS.exists():
        return []
    return [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]


def last_for(prior, commit, geometry, backend):
    """Most recent record for this geometry/backend from a *different* commit."""
    for rec in reversed(prior):
        if rec["geometry"] == geometry and rec["backend"] == backend and rec["commit"] != commit:
            return rec
    return None


def write_summary(prior):
    """Regenerate SUMMARY.md with the latest value per (geometry, backend)."""
    latest = {}
    for rec in prior:
        latest[(rec["geometry"], rec["backend"])] = rec
    lines = [
        "# Backend benchmark tracker",
        "",
        "Per-`EulerSolver.update()` time and cell throughput by geometry/backend.",
        "Higher `cell·updates/s` = more cells feasible per unit time. Appended by",
        "`scripts/bench_backends.py`; full history in `results.jsonl`.",
        "",
        "| geometry | backend | cells | ms/update | updates/s | cell·updates/s | commit |",
        "|---|---|--:|--:|--:|--:|---|",
    ]
    for (geom, backend), r in sorted(latest.items()):
        lines.append(
            f"| {geom} | {backend} | {r['cells']} | {r['per_update_ms']:.2f} | "
            f"{r['updates_per_sec']:.0f} | {r['cell_updates_per_sec']:,} | `{r['commit']}` |"
        )
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="", help="what changed since last run")
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--interval", type=float, default=0.001)
    ap.add_argument("--no-write", action="store_true", help="print only, don't append")
    args = ap.parse_args()

    commit = git_commit()
    prior = load_prior()
    RESULTS.parent.mkdir(exist_ok=True)
    new_records = []

    print(f"commit {commit}  label={args.label!r}\n")
    header = f"{'geometry':10s} {'backend':7s} {'cells':>6s} {'ms/upd':>8s} {'upd/s':>7s} {'cell·upd/s':>12s}  vs last"
    print(header)
    print("-" * len(header))
    for geometry, composite, backends in CONFIGS:
        for backend in backends:
            try:
                res = bench_one(composite, backend, args.interval, args.warmup, args.updates)
            except Exception as e:
                print(f"{geometry:10s} {backend:7s}  ERROR: {type(e).__name__}: {str(e)[:50]}")
                continue
            if res is None:
                print(f"{geometry:10s} {backend:7s}  (rust backend unavailable — skipped)")
                continue
            rec = {"commit": commit, "label": args.label, "geometry": geometry,
                   "backend": backend, "composite": composite, **res}
            prev = last_for(prior, commit, geometry, backend)
            if prev:
                d = (prev["per_update_ms"] - res["per_update_ms"]) / prev["per_update_ms"] * 100
                delta = f"{d:+.0f}% faster" if d >= 0 else f"{-d:+.0f}% slower"
            else:
                delta = "(first record)"
            print(f"{geometry:10s} {backend:7s} {res['cells']:>6d} {res['per_update_ms']:>8.3f} "
                  f"{res['updates_per_sec']:>7.0f} {res['cell_updates_per_sec']:>12,d}  {delta}")
            new_records.append(rec)

    if not args.no_write and new_records:
        with RESULTS.open("a") as f:
            for rec in new_records:
                f.write(json.dumps(rec) + "\n")
        write_summary(prior + new_records)
        print(f"\nappended {len(new_records)} records -> {RESULTS.relative_to(ROOT)}; regenerated {SUMMARY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
