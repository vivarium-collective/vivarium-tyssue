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


def make_flat_sheet(n):
    """Generate a runnable NxN flat 3D sheet with the standard sheet model.

    Depends on the pandas-3 sanitize() fix; promotes the 2D grid to 3D (z=0) so
    SheetGeometry's normals work. Returns (sheet, model)."""
    from tyssue import Sheet, SheetGeometry
    from tyssue.dynamics import effectors, model_factory

    s = Sheet.planar_sheet_2d(f"s{n}", nx=n, ny=n, distx=1, disty=1)
    s.sanitize(trim_borders=True)
    s.vert_df["z"] = 0.0
    s = Sheet(f"s{n}", {"vert": s.vert_df, "edge": s.edge_df, "face": s.face_df}, coords=["x", "y", "z"])
    model = model_factory(
        [effectors.LineTension, effectors.FaceAreaElasticity, effectors.PerimeterElasticity],
        effectors.FaceAreaElasticity,
    )
    s.update_specs(model.specs, reset=True)
    s.vert_df["viscosity"] = 1.0
    SheetGeometry.update_all(s)
    return s, model


def bench_sweep_one(n, backend, warmup, updates):
    """Per-update MATH cost (geom.update_all + gradient) on a generated sheet."""
    import numpy as np

    from tyssue import SheetGeometry
    from vivarium_tyssue.processes.utils import (
        rust_kernels_available,
        rust_sheet_gradient,
        rust_update_geometry,
    )

    s, model = make_flat_sheet(n)
    if backend == "rust":
        if not rust_kernels_available():
            return None
        vmap = {v: i for i, v in enumerate(s.vert_df.index)}
        fmap = {v: i for i, v in enumerate(s.face_df.index)}
        srce = np.ascontiguousarray(s.edge_df["srce"].map(vmap).values, np.uint32)
        trgt = np.ascontiguousarray(s.edge_df["trgt"].map(vmap).values, np.uint32)
        face = np.ascontiguousarray(s.edge_df["face"].map(fmap).values, np.uint32)

        def one():  # what the rust backend actually runs each step
            rust_update_geometry(s, srce, trgt, face)
            rust_sheet_gradient(s, False)
    else:
        def one():
            SheetGeometry.update_all(s)
            model.compute_gradient(s)

    times = []
    for _ in range(warmup):
        one()
    for _ in range(updates):
        t = time.perf_counter()
        one()
        times.append(time.perf_counter() - t)
    ms = statistics.median(times) * 1e3
    return {
        "cells": int(s.Nf),
        "verts": int(s.Nv),
        "edges": int(s.Ne),
        "per_update_ms": round(ms, 4),
        "updates_per_sec": round(1000 / ms, 1),
        "cell_updates_per_sec": round(s.Nf * 1000 / ms),
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


def run_sweep(args, commit, prior):
    """Scaling curve: per-update math cost vs cell count on generated sheets."""
    sizes = [int(x) for x in args.sizes.split(",")]
    records = []
    print(f"commit {commit}  SCALING SWEEP (generated sheets, per-update math)\n")
    header = f"{'grid':>6s} {'cells':>6s} {'edges':>6s} {'py ms':>8s} {'rust ms':>8s} {'speedup':>8s} {'rust cell·upd/s':>16s}"
    print(header)
    print("-" * len(header))
    for n in sizes:
        try:
            py = bench_sweep_one(n, "python", args.warmup, args.updates)
            ru = bench_sweep_one(n, "rust", args.warmup, args.updates)
        except Exception as e:
            print(f"{n}x{n:<3} ERROR: {type(e).__name__}: {str(e)[:50]}")
            continue
        speed = f"{py['per_update_ms'] / ru['per_update_ms']:.2f}x" if ru else "n/a"
        rcu = f"{ru['cell_updates_per_sec']:,}" if ru else "n/a"
        print(f"{n}x{n:<3} {py['cells']:>6d} {py['edges']:>6d} {py['per_update_ms']:>8.3f} "
              f"{ru['per_update_ms'] if ru else float('nan'):>8.3f} {speed:>8s} {rcu:>16s}")
        for backend, res in (("python", py), ("rust", ru)):
            if res:
                records.append({"commit": commit, "label": args.label, "geometry": "sheet-scan",
                                "scope": "math", "backend": backend, "grid": n, **res})
    if not args.no_write and records:
        with RESULTS.open("a") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        print(f"\nappended {len(records)} sweep records -> {RESULTS.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="", help="what changed since last run")
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--interval", type=float, default=0.001)
    ap.add_argument("--no-write", action="store_true", help="print only, don't append")
    ap.add_argument("--sweep", action="store_true",
                    help="cells-vs-time scaling sweep over generated sheet sizes (needs the sanitize fix)")
    ap.add_argument("--sizes", default="8,16,24,32,40", help="NxN grid sizes for --sweep")
    args = ap.parse_args()

    commit = git_commit()
    prior = load_prior()
    RESULTS.parent.mkdir(exist_ok=True)
    new_records = []

    if args.sweep:
        return run_sweep(args, commit, prior)

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
