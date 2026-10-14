"""Parameter sweeps against folding of the growing free sheet.

The free sheet in ``sbml_cell_cycle.py`` folds (faces pass through each other)
once it grows past a few hundred cells. This script varies division or mechanics
parameters around a baseline and measures how large the sheet gets
before its first fold. Results are printed, not saved; ``SWEEPS.md`` describes
what earlier sweeps found.

Every run starts from the same seeded sheet. It is advanced in ``CHUNK`` steps,
and after each chunk the mesh is checked with tyssue's exact
``find_intersections(strict=True)``. A run stops at the first fold, at
``--target`` cells, or at ``tf_max``. Fold onset is chaotic, so compare medians
over seeds rather than single runs.

Sweepable parameters (``--vary name=v1,v2,...``):

  division   cycle (sim time per cell cycle), growth (multiplies the growth rate
             that doubles a cell's target area over its G1 + G2 time)
  mechanics  area_elasticity, prefered_perimeter, perimeter_elasticity,
             line_tension, viscosity, threshold_length (T1 edge length),
             p_4, p_5p (rank-4 / rank-5+ vertex detachment rates, per sim time)
  solver     dt_factor (multiplies the solver step from ``solver_dt``)

``--oat`` varies each parameter on its own around the baseline (one at a time,
plus one baseline row); otherwise the full grid of every ``--vary`` is run. The
solver step shrinks with lower viscosity and stiffer elasticities so explicit
Euler stays stable (see ``solver_dt``).

    cd Experiments/sbml_cell_cycle
    python sweep_folding.py --oat --vary area_elasticity=0.5,2,5 --vary viscosity=0.1,0.4
    python sweep_folding.py --vary cycle=40,60 --vary growth=0.5,1 --seed 1 2 3 4
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
# Parallel workers open the same dataset file.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse  # noqa: E402
import copy  # noqa: E402
import itertools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import sbml_cell_cycle as demo  # noqa: E402

# The mechanics baseline of the sweeps in SWEEPS.md (the demo before it moved to
# viscosity 0.02). Change it with --set, e.g. --set viscosity=0.02 for the demo's.
BASELINE = {
    "cycle": 20.0,
    "growth": 1.0,
    "area_elasticity": 1.0,
    "prefered_perimeter": 3.5,
    "perimeter_elasticity": 0.5,
    "line_tension": 0.0,
    "viscosity": 0.2,
    "threshold_length": 0.03,
    # tyssue's default detachment rates, which sections 1-4 of SWEEPS.md used; the
    # demo now uses a 30 min wait (sbml_cell_cycle.P_4 / P_5P, section 5).
    "p_4": 0.1,
    "p_5p": 0.01,
    "dt_factor": 1.0,
}
# Stable explicit-Euler step at the reference mechanics (K_A 1, K_P 0.5, viscosity 0.2).
REF_DT, REF_PERIMETER_ELASTICITY, REF_VISCOSITY = 0.02, 0.5, 0.2
TARGET_CELLS = 800       # ~2 doublings of the 206-cell sheet
CHUNK = 1.0              # sim time between intersection checks


def solver_dt(p: dict) -> float:
    """REF_DT at the reference mechanics, shortened in proportion to the fastest
    relaxation rate, elasticity / viscosity."""
    stiffness = max(1.0, p["area_elasticity"],
                    p["perimeter_elasticity"] / REF_PERIMETER_ELASTICITY)
    rate = stiffness / p["viscosity"]
    dt = min(REF_DT, REF_DT * (1.0 / REF_VISCOSITY) / rate)
    return dt * p["dt_factor"]


def build_spec(p: dict) -> dict:
    spec = demo.build_spec(demo.ensure_dataset())
    cc = spec["Cell Cycle"]["config"]
    cc["time_scale"] = demo.MODEL_PERIOD / p["cycle"]
    cc["growth_rate"] = p["growth"] * np.log(2.0) / (demo.GROWTH_FRACTION * p["cycle"])
    solver = spec["Tyssue"]["config"]
    params = solver["parameters"]
    for key in ("area_elasticity", "prefered_perimeter", "perimeter_elasticity"):
        params["face_df"][key] = p[key]
    params["edge_df"]["line_tension"] = p["line_tension"]
    params["vert_df"]["viscosity"] = p["viscosity"]
    solver["settings"]["threshold_length"] = p["threshold_length"]
    solver["auto_reconnect"] = {"enabled": True, "p_4": p["p_4"], "p_5p": p["p_5p"]}
    solver["record_history"] = False
    dt = solver_dt(p)
    spec["Tyssue"]["interval"] = dt
    spec["Cell Cycle"]["interval"] = dt
    return spec


def run_one(p: dict, seed: int, target: int, chunk: float = CHUNK) -> dict:
    from process_bigraph import Composite
    from vivarium_tyssue.core import build_core

    tf_max = 3.0 * p["cycle"] * np.log2(target / 206) / 2 + 20.0
    demo.seed_all(seed)
    t, n, outcome, max_clean, wall0 = 0.0, 206, "tf_max", 0, time.time()
    shape = np.nan
    try:
        sim = Composite({"state": copy.deepcopy(build_spec(p))}, core=build_core())
        while t < tf_max - 1e-9:
            sim.run(chunk)
            t += chunk
            eptm = sim.state["Tyssue"]["instance"].eptm
            fd = eptm.face_df
            n = len(fd)
            if not eptm.find_intersections(strict=True).clean:
                outcome = "folded"
                break
            max_clean = n
            shape = float((fd["perimeter"] / np.sqrt(fd["area"])).mean())
            if n >= target:
                outcome = "target"
                break
    except Exception as exc:  # noqa: BLE001 -- an unstable setting is a result
        outcome = f"error: {type(exc).__name__}"
    return {**p, "seed": seed, "outcome": outcome, "t_end": t, "n_end": n,
            "max_clean_cells": max_clean, "shape_index": shape,
            "dt": solver_dt(p), "wall_s": round(time.time() - wall0)}


def _parse_vary(items):
    vary = {}
    for item in items:
        name, _, values = item.partition("=")
        if name not in BASELINE:
            raise SystemExit(f"unknown parameter {name!r}; choose from {list(BASELINE)}")
        vary[name] = [float(v) for v in values.split(",")]
    return vary


def configurations(vary: dict, oat: bool):
    if oat:
        yield dict(BASELINE)
        for name, values in vary.items():
            for v in values:
                if v != BASELINE[name]:
                    yield {**BASELINE, name: v}
        return
    names = list(vary)
    for values in itertools.product(*vary.values()):
        yield {**BASELINE, **dict(zip(names, values))}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--vary", action="append", default=[],
                   help="name=v1,v2,... (repeatable)")
    p.add_argument("--set", action="append", default=[],
                   help="name=value: change the baseline (repeatable)")
    p.add_argument("--oat", action="store_true",
                   help="vary one parameter at a time around the baseline")
    p.add_argument("--seed", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--target", type=int, default=TARGET_CELLS)
    p.add_argument("--chunk", type=float, default=CHUNK,
                   help="sim time between fold checks (the demo checks every 0.25)")
    p.add_argument("--workers", type=int, default=os.cpu_count() - 1)
    args = p.parse_args()

    for name, (value,) in _parse_vary(args.set).items():
        BASELINE[name] = value
    vary = _parse_vary(args.vary)
    configs = list(configurations(vary, args.oat))
    jobs = [(cfg, seed) for cfg in configs for seed in args.seed]
    print(f"{len(configs)} configurations x {len(args.seed)} seeds = {len(jobs)} runs "
          f"on {args.workers} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run_one, cfg, seed, args.target, args.chunk) for cfg, seed in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            changed = {k: r[k] for k in BASELINE if r[k] != BASELINE[k]} or "baseline"
            print(f"{changed} seed {r['seed']}: {r['outcome']} at t={r['t_end']:g}, "
                  f"max clean {r['max_clean_cells']} cells ({r['wall_s']} s)", flush=True)

    df = pd.DataFrame(results)
    keys = list(BASELINE)
    table = (df.groupby(keys)
               .agg(seeds=("seed", "size"),
                    median_clean=("max_clean_cells", "median"),
                    min_clean=("max_clean_cells", "min"),
                    max_clean=("max_clean_cells", "max"),
                    reached=("outcome", lambda s: int((s == "target").sum())),
                    errors=("outcome", lambda s: int(s.str.startswith("error").sum())),
                    shape_index=("shape_index", "median"),
                    dt=("dt", "first"))
               .reset_index()
               .sort_values("median_clean", ascending=False))
    shown = [k for k in keys if table[k].nunique() > 1] or keys[:1]
    print(table[shown + ["seeds", "median_clean", "min_clean", "max_clean", "reached",
                         "errors", "shape_index", "dt"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
