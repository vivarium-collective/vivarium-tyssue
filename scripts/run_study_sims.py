"""Run a study's baseline composite and record a runs.db the dashboard viz read.

The dashboard's own study-run path didn't capture our stores (it recorded only
global_time, and the scalar observable stores accumulate additively), so this
runner steps the composite directly and writes the per-step state the
visualizations expect:

  - scalar observables (tumor/healthy/stem/dead_count, *_births, *_deaths) — for
    TimeSeriesFromObservables (reads history.state[<name>] per step)
  - Datasets/{vert,edge,face}_df — for TissueSheetSnapshots / TissueSheetGif

runs.db schema matches the dashboard (runs_meta / history / simulations), keyed
so RunReader / TimeSeriesFromObservables._load_runs pick it up. Counts are
recomputed from face_df each step (instantaneous); births/deaths are the per-step
delta of the accumulating coupling stores.

Usage: python scripts/run_study_sims.py <study> <sim_name> <composite.yaml> <steps> [interval]
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TYPES = ["tumor", "healthy", "stem", "dead"]
_EVENTS = ["tumor_births", "healthy_births", "stem_births",
           "tumor_deaths", "healthy_deaths", "stem_deaths"]


def _jsonable(df):
    """face/vert/edge DataFrame -> dict[col -> list] with plain Python scalars."""
    out = {}
    for col in df.columns:
        vals = df[col].tolist()
        out[col] = [None if (isinstance(v, float) and np.isnan(v)) else
                    (v.item() if isinstance(v, np.generic) else v) for v in vals]
    return out


def run_study(study: str, sim_name: str, composite, steps: int,
              interval: float = 0.1, seed: int = 0) -> None:
    """Run one simulation and record runs.db.

    ``composite`` is either a path to a composite spec file (str/Path) or an
    already-parsed spec dict. The dict form lets a caller (e.g. the generated
    reproduction notebook) edit the process-bigraph specification — config
    parameters, per-process ``interval`` — before the composite is built and
    run. ``${interval}`` placeholders left in the spec are filled from the
    ``interval`` argument; a process whose ``interval`` was pinned to a literal
    keeps that value.
    """
    from pbg_superpowers.composite_spec import load_spec, build_composite_from_spec
    from vivarium_tyssue.core import build_core

    np.random.seed(seed)
    spec = composite if isinstance(composite, dict) else load_spec(Path(composite))

    # 3D monolayer studies have no cell-fate coupling, so their observables are
    # morphology (apical lift-off), not tumor/healthy counts. The generated
    # reproduction notebook imports a single run_study per workspace, so route
    # monolayer composites to the lumen recorder here rather than recording the
    # tumor stores (which would all be zero for these specs).
    try:
        _tissue = spec.get("state", {}).get("Tyssue", {}).get("config", {}).get("tissue_type")
    except AttributeError:
        _tissue = None
    if _tissue == "Monolayer":
        from scripts.run_lumen_sims import run_study as _run_lumen
        return _run_lumen(study, sim_name, spec, steps, interval, seed)

    core = build_core()
    comp = build_composite_from_spec(spec, overrides={"interval": interval}, core=core)

    db_path = ROOT / "workspace" / "studies" / study / "runs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs_meta (run_id TEXT PRIMARY KEY, spec_id TEXT NOT NULL,
            label TEXT, params_json TEXT, started_at REAL NOT NULL, completed_at REAL,
            n_steps INTEGER, status TEXT NOT NULL, sim_name TEXT);
        CREATE TABLE IF NOT EXISTS history (simulation_id TEXT NOT NULL, step INTEGER NOT NULL,
            global_time REAL, state TEXT NOT NULL, PRIMARY KEY (simulation_id, step));
        CREATE TABLE IF NOT EXISTS simulations (simulation_id TEXT PRIMARY KEY, name TEXT,
            started_at TEXT NOT NULL, completed_at TEXT, elapsed_seconds REAL,
            composite_config TEXT, metadata TEXT);
        """
    )
    spec_id = spec.get("name", study)
    run_id = f"{spec_id}__{sim_name}"
    # Replace any prior run for this sim_name (idempotent).
    conn.execute("DELETE FROM history WHERE simulation_id=?", (run_id,))
    conn.execute("DELETE FROM runs_meta WHERE run_id=?", (run_id,))
    conn.execute("DELETE FROM simulations WHERE simulation_id=?", (run_id,))
    t0 = time.time()
    conn.execute(
        "INSERT INTO runs_meta (run_id, spec_id, label, params_json, started_at, n_steps, status, sim_name)"
        " VALUES (?,?,?,?,?,?,?,?)",
        (run_id, spec_id, sim_name, json.dumps({"interval": interval, "steps": steps}),
         t0, steps, "running", sim_name),
    )
    conn.execute(
        "INSERT INTO simulations (simulation_id, name, started_at, composite_config) VALUES (?,?,?,?)",
        (run_id, sim_name, str(t0), json.dumps({"composite": spec_id, "interval": interval})),
    )

    eptm = comp.state["Tyssue"]["instance"].eptm
    prev = {k: 0.0 for k in _EVENTS}
    # The heavy per-step mesh snapshot (vert/edge/face DataFrames as JSON) dominates
    # runs.db size. The timeseries viz read the scalar state every step, but the
    # snapshot/gif viz only subsample ~6/60 frames — so store the full mesh only every
    # Nth step (keeping ~120 mesh frames regardless of step count), which bounds the db
    # at long step counts (interval 0.01 needs ~10x more steps than the old 0.1 runs).
    snapshot_every = max(1, steps // 120)
    for step in range(steps + 1):
        if step > 0:
            comp.run(1)
        gt = float(comp.state.get("global_time", step * interval))
        counts = Counter(eptm.face_df["cell_type"]) if "cell_type" in eptm.face_df else Counter()
        state = {"global_time": gt}
        for t in _TYPES:
            state[f"{t}_count"] = float(counts.get(t, 0))
        # total_count = tissue size (grows under topology_ops as cells divide);
        # healthy_fraction = displaced-tissue signal robust to division-driven growth.
        total = sum(float(counts.get(t, 0)) for t in _TYPES)
        state["total_count"] = total
        state["healthy_fraction"] = float(counts.get("healthy", 0)) / total if total else 0.0
        # births/deaths: per-step delta of the accumulating coupling stores
        for k in _EVENTS:
            cur = float(comp.state.get(k, 0.0) or 0.0)
            state[k] = max(cur - prev[k], 0.0)
            prev[k] = cur
        if step % snapshot_every == 0 or step == steps:
            state["Datasets"] = {
                "vert_df": _jsonable(eptm.vert_df),
                "edge_df": _jsonable(eptm.edge_df),
                "face_df": _jsonable(eptm.face_df),
            }
        conn.execute("INSERT OR REPLACE INTO history (simulation_id, step, global_time, state)"
                     " VALUES (?,?,?,?)", (run_id, step, gt, json.dumps(state)))

    conn.execute("UPDATE runs_meta SET status='complete', completed_at=?, progress_step=? WHERE run_id=?"
                 if False else
                 "UPDATE runs_meta SET status='complete', completed_at=? WHERE run_id=?",
                 (time.time(), run_id))
    conn.commit()
    final = {t: int(counts.get(t, 0)) for t in _TYPES}
    print(f"{study}/{sim_name}: {steps} steps -> runs.db ({db_path.name}); final {final}")
    conn.close()


if __name__ == "__main__":
    study, sim_name, comp_yaml, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    interval = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    run_study(study, sim_name, comp_yaml, steps, interval)
