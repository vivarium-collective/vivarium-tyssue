"""Stochastic line-tension experiments — a tau/sigma parameter sweep (SIMULATION).

A flat epithelial square sheet is driven by an Ornstein-Uhlenbeck fluctuation of
per-junction line tension (the ``StochasticLineTension`` process). This script sweeps
the OU relaxation timescale ``tau`` and noise amplitude ``sigma``, runs
``N_REPEATS`` independent repeats (different seeds) of every grid cell, and for each
run **writes summarised raw data, not the full History**:

* ``summary.json``  — the run's scalar metrics (vertex movement, T1 counts, tension
  statistics) plus its seed and status,
* ``timeseries.csv`` — per-frame RMS vertex step and junction-tension spread,
* ``t1_events.csv``  — every completed neighbour exchange (break / join times, cells),
* ``movie.hf5``      — repeat 0 only: ``MOVIE_FRAMES`` evenly spaced frames with just
  the columns needed to draw the sheet (positions, topology, line tension), for the
  GIF and stills.
* ``junction_tension.hf5`` — repeat 0 only: every interior junction's tension at
  every solver step, keyed by the ``unique_id`` of the two cells it separates
  (single-junction traces in the notebook).

The metrics are computed here, from the full-resolution in-memory History, while the
run is still live; the History is then discarded. ``runs.csv`` collects every run's
summary. All visualisation, averaging over repeats and the summary figures live in
``stochastic_tension_analysis.ipynb``.

Each run executes in its own worker process (``maxtasksperchild=1``), so memory is
returned to the OS after every run and ``N_WORKERS`` bounds the peak.

Run (from the repo's ``vivarium-tyssue`` conda env; the .venv/uv are broken):

    conda activate vivarium-tyssue
    cd Experiments/stochastic_tension
    python stochastic_tension.py
"""
from __future__ import annotations

import os

# One BLAS thread per worker: several workers run at once, and multithreaded BLAS
# makes same-seed runs diverge (summation order flips T1s). Must precede numpy.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
# Workers starting together all open the input mesh at once; HDF5's file lock then
# fails with "Resource temporarily unavailable". Every worker only reads it.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import json
import multiprocessing as mp
import random
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / sweep configuration
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
DATASET_NAME = "test_square.hf5"

# 5 x 5 sweep. tau = OU relaxation timescale, sigma = noise amplitude (the
# stationary standard deviation of each junction's tension). Both axes are
# geometric (x2 per step).
TAUS = [0.1, 0.2, 0.5, 1.0, 2.0]
SIGMAS = [0.05, 0.1, 0.2, 0.4, 0.8]
N_REPEATS = 3      # independent seeds per (tau, sigma); the notebook plots their mean

TF = 15.0          # total simulated time
DT = 0.1           # emit / solver interval (~150 steps per run)
SEED = 20260714    # base seed; run k uses SEED + k (see ``run_seed``)

COORDS = ["x", "y"]

# Parallel workers. Each holds one run (~a few hundred MB with the imports); keep
# this well under the core count so the machine stays usable.
N_WORKERS = 4

# Frames kept in repeat 0's ``movie.hf5`` (drawing only — the metrics are computed
# from every solver step before the History is dropped), and the columns kept.
MOVIE_FRAMES = 60
MOVIE_COLUMNS = {
    "vert": ["unique_id", "x", "y", "z"],
    "edge": ["srce", "trgt", "face", "line_tension"],
    "face": ["unique_id"],
}
# Repeat 0 also keeps every interior junction's tension at every solver step
# (~150 x ~600 values), so single-junction OU traces can be drawn at full resolution.
JUNCTION_TRACE_FILE = "junction_tension.hf5"
ARCHIVE_COMPLIB = "blosc:zstd"
ARCHIVE_COMPLEVEL = 5


def tag(tau: float, sigma: float) -> str:
    return f"tau{tau:0.2f}_sigma{sigma:0.2f}"


def run_dir(tau: float, sigma: float, rep: int) -> Path:
    return OUT_DIR / tag(tau, sigma) / f"rep{rep}"


def run_seed(tau: float, sigma: float, rep: int) -> int:
    """Seed of one run, fixed by its position in the sweep — rerunning a single
    cell reproduces it without rerunning the rest."""
    k = (TAUS.index(tau) * len(SIGMAS) + SIGMAS.index(sigma)) * N_REPEATS + rep
    return SEED + k


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def ensure_dataset() -> Path:
    """Copy the flat-sheet mesh into ``data/`` (git-ignored) if not already there,
    sourcing it from the repo's tracked ``workspace/datasets`` copy."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_DIR / DATASET_NAME
    if not dst.exists():
        src = REPO / "workspace" / "datasets" / DATASET_NAME
        if not src.exists():
            raise FileNotFoundError(f"source mesh not found: {src}")
        shutil.copy(src, dst)
        print(f"copied {src} -> {dst}")
    return dst


# ---------------------------------------------------------------------------
# Simulation spec (adapted from tests/tests.py get_test_config_flat +
# get_test_stochastic_spec — self-contained, no emitter; the metrics are taken from
# the solver's in-memory History instead).
# ---------------------------------------------------------------------------
def build_spec(tau: float, sigma: float, dt: float, dataset_path: Path) -> dict:
    tyssue_config = {
        "name": "Stochastic Square",
        "eptm": str(dataset_path),
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {
                "area_elasticity": 1.0,
                "prefered_area": 1.0,
                "perimeter_elasticity": 0.1,
                "prefered_perimeter": 3.6,
                "migration_strength": 0.0,
                "is_alive": 1.0,
                "mx": 1.0,
                "mz": 0.0,
                "my": 1.0,
            },
            "edge_df": {
                "line_tension": 0.0,
                "is_active": 1.0,
            },
            "vert_df": {
                "viscosity": 1.0,
                "is_alive": 1.0,
            },
        },
        "geom": "SheetGeometry",
        "effectors": ["LineTension", "FaceAreaElasticity", "PerimeterElasticity"],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory_bound",
        "settings": {"threshold_length": 0.03},
        "auto_reconnect": {"enabled": True},
        "bounds": None,
        "output_columns": {},
        "history_columns": {},
        "maps": {},
        "backend": "rust",
        "substeps": 1,
        "max_displacement": 0.0,
        "record_history": True,  # build an in-memory History (records all columns)
    }

    return {
        "Tyssue": {
            "_type": "process",
            "address": "local:EulerSolver",
            "config": tyssue_config,
            "inputs": {
                "behaviors": ["Behaviors"],
                "global_time": ["global_time"],
            },
            "outputs": {
                "datasets": ["Tissue State"],
                "network_changed": ["Network Changed"],
                "behaviors_update": ["Behaviors"],
            },
            "interval": dt,
        },
        "Stochastic": {
            "_type": "process",
            "address": "local:StochasticLineTension",
            "config": {"tau": tau, "sigma": sigma},
            "inputs": {"datasets": ["Tissue State"]},
            "outputs": {"behaviors": ["Behaviors"], "datasets": ["Tissue State"]},
            "interval": dt,
        },
        "Network Changed": False,
        "Behaviors": {},
    }


# ---------------------------------------------------------------------------
# Metrics — computed on the full-resolution History (every solver step). Both
# match vertices/faces by unique_id across frames: identity is stable across
# reconnections, the positional index is not.
#
# They read the stacked ``history.datasets``, never ``history.retrieve(t)``:
# tyssue's ``Epithelium.__init__`` overwrites ``unique_id`` with the row index, so
# every retrieved Sheet has positional ids. After the first T1 renumbers the
# vertices, "the same unique_id" in two retrieved frames is two different
# vertices, and the per-step displacement is inflated ~60x.
# ---------------------------------------------------------------------------
def vertex_movement(history):
    """Degree of stochastic vertex movement.

    Returns ``(per_step_rms, mean_path_length)``: the RMS vertex displacement of
    every step (one value per frame after the first, over the vertices present in
    both frames) and the total distance travelled per vertex, averaged over
    vertices.
    """
    vert = history.datasets["vert"]
    pos = vert.pivot(index="time", columns="unique_id", values=COORDS).sort_index()
    sq = sum(pos[c].diff() ** 2 for c in COORDS).iloc[1:]   # NaN: absent in a frame
    step = np.sqrt(sq)
    per_step_rms = np.sqrt(sq.mean(axis=1, skipna=True)).to_numpy(dtype=float)
    moved = step.notna().any(axis=0)
    mean_path_length = float(step.loc[:, moved].sum(axis=0).mean()) if moved.any() else 0.0
    return per_step_rms, mean_path_length


def _frames(history):
    """``(t, vert_df, edge_df, face_df)`` per recorded time, from the stacked
    datasets — each indexed by its per-frame element label, with the true
    ``unique_id`` column (see the note above)."""
    groups = {elem: dict(tuple(history.datasets[elem].groupby("time", sort=True)))
              for elem in ("vert", "edge", "face")}
    for t in sorted(groups["vert"]):
        yield (t, *(groups[elem][t].set_index(elem) for elem in ("vert", "edge", "face")))


def _face_adjacency(vert_df, edge_df, face_df):
    # Set of neighbouring face pairs (share >= 2 vertices), keyed by face unique_id.
    vuid = vert_df["unique_id"]
    fuid = face_df["unique_id"]

    face_vsets = defaultdict(set)
    for f, s in zip(edge_df["face"].to_numpy(), edge_df["srce"].to_numpy()):
        face_vsets[int(f)].add(int(vuid.loc[s]))

    vert_faces = defaultdict(set)
    for fpos, vset in face_vsets.items():
        for v in vset:
            vert_faces[v].add(fpos)

    shared = defaultdict(int)
    for faces in vert_faces.values():
        fl = sorted(faces)
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                shared[(fl[i], fl[j])] += 1

    adj = set()
    for (a, b), n in shared.items():
        if n >= 2:
            adj.add(frozenset((int(fuid.loc[a]), int(fuid.loc[b]))))
    return adj


def _neighbours(adj):
    # cell unique_id -> set of neighbouring cell unique_ids, from an adjacency set.
    nb = defaultdict(set)
    for pair in adj:
        a, b = tuple(pair)
        nb[a].add(b)
        nb[b].add(a)
    return nb


def t1_events(history):
    """Completed neighbour exchanges over the run, matched break -> join.

    A junction between cells A and B vanishes as soon as it shrinks past
    `threshold_length` and tyssue merges its two vertices — but that four-fold vertex
    can split either way: back into the original A|B junction (nothing was exchanged),
    or across it, handing the flanking cells C and D a junction of their own (a real
    T1). Losing an adjacency is therefore not yet an event; the pair that flanked it
    has to join up. Each lost pair opens a pending break, and

      * regaining {A, B} closes it as **reverted** — the vertex split back the way it
        came, so no exchange happened;
      * gaining the flanking pair {C, D} closes it as a completed **T1**, matched
        either against the flank recorded when the junction broke or against the cells
        flanking the new junction now (the neighbourhood can shift in between);
      * any other new adjacency is an unmatched new contact (cells touching as the
        sheet deforms), and breaks still open at the end of the run stay unresolved
        (``unresolved_boundary`` counts those on the tissue boundary, which have a
        single flanking cell and so can never complete).

    Returns ``(events, diagnostics)``; ``len(events)`` is the T1 count.
    """
    events, pending = [], {}          # pending: broken pair -> (t_break, flanking pair)
    reverted = unmatched = 0
    prev = None
    for t, vert_df, edge_df, face_df in _frames(history):
        adj = _face_adjacency(vert_df, edge_df, face_df)
        if prev is not None:
            lost, gained = prev - adj, adj - prev
            nb_before, nb_now = _neighbours(prev), _neighbours(adj)
            for pair in lost:                     # open a break (flank: while intact)
                a, b = tuple(pair)
                pending[pair] = (t, frozenset(nb_before[a] & nb_before[b]))
            for pair in gained:
                if pair in pending:               # split back the way it came
                    del pending[pair]
                    reverted += 1
                    continue
                c, d = tuple(pair)
                flank_now = nb_now[c] & nb_now[d]     # cells flanking the NEW junction
                matches = [broken for broken, (_, flank) in pending.items()
                           if (len(flank) >= 2 and pair <= flank) or broken <= flank_now]
                if matches:
                    # most recent break first: the junction that just merged is the one
                    # splitting back open (ties do not arise in practice).
                    broken = max(matches, key=lambda b: pending[b][0])
                    t_break, _ = pending.pop(broken)
                    events.append({"t_break": float(t_break), "t_join": float(t),
                                   "lost": tuple(sorted(broken)),
                                   "gained": tuple(sorted(pair))})
                else:
                    unmatched += 1
        prev = adj
    boundary = sum(1 for _, flank in pending.values() if len(flank) < 2)
    return events, {"reverted": reverted, "unmatched": unmatched,
                    "unresolved": len(pending), "unresolved_boundary": boundary}


def junction_tension(edge_df: pd.DataFrame) -> np.ndarray:
    """Tension each interior junction exerts: ``LineTension`` charges ½ Λ ℓ per
    half-edge, so a junction feels the mean of its two halves (equal by
    construction now that the noise is drawn per junction)."""
    from vivarium_tyssue.processes.regulations import junction_index

    junction = junction_index(edge_df)
    counts = np.bincount(junction)
    lam = np.bincount(junction, weights=edge_df["line_tension"].to_numpy(dtype=float))
    interior = counts == 2
    return lam[interior] / 2.0


def junction_traces(history) -> pd.DataFrame:
    """Every interior junction's tension at every solver step, long format:
    ``time, f_lo, f_hi, tension``. A junction is keyed by the ``unique_id`` of the
    two cells it separates (smaller first). Vertex ids are no use as a key: each
    vertex merge/split at a T1 attempt (reverted ones included) re-creates the
    vertices, so neighbouring junctions would change key while physically
    persisting. The cell pair only changes when the cells really stop being
    neighbours. The tension is the mean of the two half-edges (equal by
    construction under per-junction noise).
    """
    parts = []
    for t, vert_df, edge_df, face_df in _frames(history):
        vuid = vert_df["unique_id"]
        a = vuid.loc[edge_df["srce"].to_numpy()].to_numpy(dtype=np.int64)
        b = vuid.loc[edge_df["trgt"].to_numpy()].to_numpy(dtype=np.int64)
        fuid = face_df["unique_id"].loc[edge_df["face"].to_numpy()].to_numpy(dtype=np.int64)
        half = pd.DataFrame({"v_lo": np.minimum(a, b), "v_hi": np.maximum(a, b),
                             "face": fuid,
                             "tension": edge_df["line_tension"].to_numpy(dtype=float)})
        # Pair the two half-edges of each junction, then name it by their two cells.
        j = half.groupby(["v_lo", "v_hi"]).agg(f_lo=("face", "min"), f_hi=("face", "max"),
                                               n=("face", "size"), tension=("tension", "mean"))
        j = j[j["n"] == 2]                                   # interior junctions only
        # Two cells sharing more than one junction would collide on the key: drop them.
        j = j[~j.duplicated(["f_lo", "f_hi"], keep=False)]
        parts.append(pd.DataFrame({"time": t, "f_lo": j["f_lo"].to_numpy(),
                                   "f_hi": j["f_hi"].to_numpy(),
                                   "tension": j["tension"].to_numpy()}))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Saving the summarised data
# ---------------------------------------------------------------------------
def save_movie(history, path: Path, n_frames: int = MOVIE_FRAMES):
    """``n_frames`` evenly spaced frames, ``MOVIE_COLUMNS`` only, compressed. Enough
    to redraw the sheet with edges coloured by tension; nothing else."""
    times = np.asarray(list(history.time_stamps), dtype=float)
    idx = np.unique(np.round(np.linspace(0, times.size - 1, n_frames)).astype(int))
    keep = times[idx]
    if path.exists():
        path.unlink()
    with pd.HDFStore(path, "w", complevel=ARCHIVE_COMPLEVEL, complib=ARCHIVE_COMPLIB) as store:
        for elem, cols in MOVIE_COLUMNS.items():
            df = history.datasets[elem]
            df = df.loc[df["time"].isin(keep), ["time", elem] + cols].reset_index(drop=True)
            store.put(elem, df, format="table", data_columns=["time"])


def simulate_one(job: tuple) -> dict:
    """Worker: run one (tau, sigma, rep), write its summarised data, return the
    summary row. Never raises — a failed run is reported with ``status`` and
    ``error`` so the rest of the sweep continues."""
    tau, sigma, rep = job
    out = run_dir(tau, sigma, rep)
    out.mkdir(parents=True, exist_ok=True)
    seed = run_seed(tau, sigma, rep)
    row = {"tau": tau, "sigma": sigma, "rep": rep, "seed": seed}
    t0 = time.time()
    try:
        sys.path.insert(0, str(REPO))
        from process_bigraph import Composite
        from vivarium_tyssue.core import build_core

        np.random.seed(seed)
        random.seed(seed)       # tyssue's EventManager shuffles with `random`
        core = build_core()
        spec = build_spec(tau, sigma, DT, DATA_DIR / DATASET_NAME)
        sim = Composite({"state": spec}, core=core)
        sim.run(TF)
        history = sim.state["Tyssue"]["instance"].history
        history.update_datasets()
        del sim

        times = np.asarray(list(history.time_stamps), dtype=float)
        step_rms, mean_path_length = vertex_movement(history)
        events, diag = t1_events(history)

        # Junction-tension spread per frame (the realised OU amplitude).
        edges = history.datasets["edge"]
        tension_sd = [float(np.std(junction_tension(edges[edges["time"] == t])))
                      for t in times]
        # Stationary estimate over the second half of the run (past the OU warm-up).
        late = edges[edges["time"] >= times[len(times) // 2]]
        late_lam = np.concatenate([junction_tension(late[late["time"] == t])
                                   for t in np.unique(late["time"])])

        pd.DataFrame({
            "time": times,
            "step_rms": np.concatenate([[np.nan], step_rms]),
            "tension_sd": tension_sd,
        }).to_csv(out / "timeseries.csv", index=False)
        pd.DataFrame(events, columns=["t_break", "t_join", "lost", "gained"]).to_csv(
            out / "t1_events.csv", index=False)
        if rep == 0:
            save_movie(history, out / "movie.hf5")
            junction_traces(history).to_hdf(
                out / JUNCTION_TRACE_FILE, key="junctions", mode="w",
                complevel=ARCHIVE_COMPLEVEL, complib=ARCHIVE_COMPLIB)

        row.update({
            "status": "ok",
            "n_frames": int(times.size),
            "mean_step_rms": float(np.nanmean(step_rms)) if step_rms.size else 0.0,
            "mean_path_length": mean_path_length,
            "t1_count": len(events),
            "t1_reverted": diag["reverted"],
            "t1_unresolved": diag["unresolved"],
            "t1_unresolved_boundary": diag["unresolved_boundary"],
            "t1_unmatched": diag["unmatched"],
            "tension_absmax": float(np.nanmax(np.abs(edges["line_tension"].to_numpy(float)))),
            "junction_tension_sd": float(np.std(late_lam)),
        })
    except Exception as exc:  # noqa: BLE001 — record and keep the sweep going
        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        (out / "error.txt").write_text(traceback.format_exc())
    row["runtime_s"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(row, indent=1))
    return row


def collect_runs() -> pd.DataFrame:
    """Every run's ``summary.json`` as one table (also written to ``runs.csv``)."""
    rows = [json.loads(p.read_text()) for p in sorted(OUT_DIR.glob("tau*_sigma*/rep*/summary.json"))]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["tau", "sigma", "rep"]).reset_index(drop=True)
        df.to_csv(OUT_DIR / "runs.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main(jobs=None, workers: int = N_WORKERS):
    ensure_dataset()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if jobs is None:
        jobs = [(tau, sigma, rep) for tau in TAUS for sigma in SIGMAS
                for rep in range(N_REPEATS)]
    print(f"{len(jobs)} runs on {workers} workers", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        for i, row in enumerate(pool.imap_unordered(simulate_one, jobs), 1):
            label = f"{tag(row['tau'], row['sigma'])}/rep{row['rep']}"
            if row["status"] == "ok":
                msg = (f"rms={row['mean_step_rms']:.4g} T1={row['t1_count']} "
                       f"junction sd={row['junction_tension_sd']:.3g}")
            else:
                msg = f"FAILED {row['error']}"
            print(f"[{i:3d}/{len(jobs)}] {label:24s} {row['runtime_s']:6.1f}s  {msg}", flush=True)

    df = collect_runs()
    n_ok = int((df["status"] == "ok").sum()) if not df.empty else 0
    print(f"\ndone in {time.time() - t0:.0f}s — {n_ok}/{len(df)} runs ok; "
          f"summaries in {OUT_DIR / 'runs.csv'}")
    print("Run stochastic_tension_analysis.ipynb to visualise and analyse the sweep.")


if __name__ == "__main__":
    main()
