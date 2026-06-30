"""Run the 3D monolayer lift-off composite and record a runs.db the viz read.

Sibling of ``run_study_sims.py`` (the tumor runner), but for the pure-mechanics
3D monolayer: there are no cell-fate events, so it records *morphology* scalars
that quantify apical lift-off, plus the per-step mesh (vert/edge/face/cell df)
the 3D / profile viz draw.

Recorded scalar observables (history.state, per step):
  - apical_mean_z      mean z of apical-surface vertices (0 at t0; departs as it buckles)
  - apical_min_z       lowest apical vertex (the deepest point of the invagination)
  - apical_amplitude   max-min apical z (out-of-plane deformation magnitude)
  - boundary_cell_z    mean z of cells on the apical free boundary (the purse-string rim)
  - interior_cell_z    mean z of interior cells
  - dome_height        interior_cell_z - boundary_cell_z (dome/lumen bulge vs the rim)

runs.db schema matches the dashboard / run_study_sims so the same RunReader /
visualization Path-C loaders pick it up.

Usage: python scripts/run_lumen_sims.py <study> <sim_name> <composite> <steps> [interval]
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _jsonable(df):
    """vert/edge/face/cell DataFrame -> dict[col -> list] with plain scalars."""
    out = {}
    for col in df.columns:
        vals = df[col].tolist()
        out[col] = [None if (isinstance(v, float) and np.isnan(v)) else
                    (v.item() if isinstance(v, np.generic) else v) for v in vals]
    return out


def run_study(study: str, sim_name: str, composite, steps: int,
              interval: float = 0.1, seed: int = 0) -> None:
    """Run one monolayer lift-off simulation and record runs.db.

    ``composite`` is a path to a composite spec or an already-parsed spec dict
    (the dict form lets the reproduction notebook edit the spec — e.g. drop the
    apical tension to 0 for a flat control — before the composite is built)."""
    from pbg_superpowers.composite_spec import load_spec, build_composite_from_spec
    from vivarium_tyssue.core import build_core

    np.random.seed(seed)
    core = build_core()
    spec = composite if isinstance(composite, dict) else load_spec(Path(composite))
    comp = build_composite_from_spec(spec, overrides={"interval": interval}, core=core)
    eptm = comp.state["Tyssue"]["instance"].eptm

    # Cells on the apical free boundary: an apical edge with no opposite half-edge.
    edge_df = eptm.edge_df
    bnd = ((edge_df["segment"] == "apical") & (edge_df["opposite"] == -1))
    boundary_cells = set(edge_df.loc[bnd, "cell"].unique()) if "cell" in edge_df else set()

    db_path = ROOT / "workspace" / "studies" / study / "runs.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
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

    # Keep ~40 mesh frames regardless of step count. The monolayer mesh is large
    # (~7k edges), so each snapshot is heavy; the gif/snapshot viz subsample anyway.
    snapshot_every = max(1, steps // 40)
    for step in range(steps + 1):
        if step > 0:
            comp.run(1)
        e = comp.state["Tyssue"]["instance"].eptm
        gt = float(comp.state.get("global_time", step * interval))
        vdf = e.vert_df
        apical_z = vdf.loc[vdf["segment"] == "apical", "z"]
        cdf = e.cell_df
        is_bnd = cdf.index.isin(boundary_cells)
        bnd_z = float(cdf.loc[is_bnd, "z"].mean()) if is_bnd.any() else 0.0
        int_z = float(cdf.loc[~is_bnd, "z"].mean()) if (~is_bnd).any() else 0.0
        state = {
            "global_time": gt,
            "apical_mean_z": float(apical_z.mean()),
            "apical_min_z": float(apical_z.min()),
            "apical_amplitude": float(apical_z.max() - apical_z.min()),
            "boundary_cell_z": bnd_z,
            "interior_cell_z": int_z,
            "dome_height": int_z - bnd_z,
        }
        if step % snapshot_every == 0 or step == steps:
            datasets = {
                "vert_df": _jsonable(e.vert_df),
                "edge_df": _jsonable(e.edge_df),
                "face_df": _jsonable(e.face_df),
            }
            if e.cell_df is not None and len(e.cell_df) > 0:
                datasets["cell_df"] = _jsonable(e.cell_df)
            state["Datasets"] = datasets
        conn.execute("INSERT OR REPLACE INTO history (simulation_id, step, global_time, state)"
                     " VALUES (?,?,?,?)", (run_id, step, gt, json.dumps(state)))

    conn.execute("UPDATE runs_meta SET status='complete', completed_at=? WHERE run_id=?",
                 (time.time(), run_id))
    conn.commit()
    e = comp.state["Tyssue"]["instance"].eptm
    final_apical = float(e.vert_df.loc[e.vert_df["segment"] == "apical", "z"].mean())
    print(f"{study}/{sim_name}: {steps} steps -> runs.db ({db_path.name}); "
          f"final apical_mean_z={final_apical:.3f}")
    conn.close()


if __name__ == "__main__":
    study, sim_name, comp_yaml, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    interval = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
    run_study(study, sim_name, comp_yaml, steps, interval)
