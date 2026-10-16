"""Tension composition: two tension processes driving the same junctions at once.

The parameter-setting processes ship *deltas*: each one moves only its own share of
``line_tension`` (mirrored in its own ``edge_df`` column) and the ``update_parameter``
behavior adds the change in place. So a stochastic line tension now adds to a
directional or an adhesion tension instead of overwriting it:

    line_tension = directional_tension + stochastic_tension        (convergent extension)
    line_tension = adhesion_tension    + stochastic_tension        (cell sorting)

Two scenarios, each the parent experiment's composite plus ``StochasticLineTension``
(an Ornstein-Uhlenbeck share with persistence ``TAU`` and amplitude ``sigma``):

  * **ce** — ``Experiments/directional_tension``: the flat patch, polarity along x,
    Lambda(theta) from 0 (perpendicular) to 0.15 (parallel), cos^2 law. Runs: no
    noise (the parent's ``px`` run), ``SIGMAS`` of noise on top, and a noise-only
    control (no directional tension) at the largest sigma.
  * **sort** — ``Experiments/differential_adhesion``: the 200-cell sphere, 50:50
    random labels, Lambda_homo 0.05, Lambda_het 0.45. Runs: no noise (the parent's
    sorting run), ``SIGMAS`` of noise on top, and a noise-only control (uniform
    Lambda_homo, i.e. the parent's control) at the largest sigma.

Mechanics, meshes, T1 settings and seeds are copied from the parents, so each
scenario's no-noise run reproduces the parent's main run (up to the solver's
unseeded event-order shuffle, which is seeded here).

Each run's metrics are computed from its full history (every frame) and written to
``outputs/metrics_<scenario>.csv``; the history itself is archived thinned to
``ARCHIVE_FRAMES`` frames as ``outputs/<scenario>/<run>/history.hf5`` for the
figures in ``tension_composition_analysis.ipynb``. Run from the repo's
``vivarium-tyssue`` conda env:

    cd Experiments/tension_composition
    python tension_composition.py               # both scenarios, runs in parallel
    python tension_composition.py --only ce     # one scenario
"""
from __future__ import annotations

import os

# Pin BLAS threads before numpy loads: same-seed runs otherwise diverge, and the
# runs go in parallel processes anyway.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")
# The parallel workers all open the same mesh file; HDF5's file lock refuses that.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import json
import random
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
sys.path.insert(0, str(REPO))

# ---------------------------------------------------------------------------
# The stochastic share (shared by both scenarios)
# ---------------------------------------------------------------------------
TAU = 1.0                         # OU persistence time (~ the K_A / eta relaxation time)
SIGMAS = [0.025, 0.05, 0.1]       # OU amplitudes on top of the deterministic tension
CONTROL_SIGMA = SIGMAS[-1]        # noise-only control
ARCHIVE_FRAMES = 241              # frames kept in each history.hf5 (metrics use all)

# ---------------------------------------------------------------------------
# Scenario ce: Experiments/directional_tension (values copied from its notebook)
# ---------------------------------------------------------------------------
CE_SEED = 20260901
CE_COORDS = ["x", "y"]
CE_SOURCE_MESH = REPO / "workspace" / "datasets" / "test_square.hf5"
CE_MESH = DATA_DIR / "flat_sheet_2d.hf5"
CE_POLARITY = (1.0, 0.0)
CE_LAMBDA_MIN, CE_LAMBDA_MAX = 0.0, 0.15
CE_PROFILE, CE_SHARPNESS = "cos2", 1.0
CE_THRESHOLD_LENGTH, CE_P_4, CE_P_5P = 0.12, 5.0, 0.5
CE_DT, CE_TF = 0.1, 240.0
CE_METRIC_STRIDE = 6              # geometry sampled every 6th frame; T1s on every frame

# ---------------------------------------------------------------------------
# Scenario sort: Experiments/differential_adhesion (values copied from its notebook)
# ---------------------------------------------------------------------------
SORT_SEED = 20260831
SORT_COORDS = ["x", "y", "z"]
SORT_MESH = DATA_DIR / "sphere_200.hf5"
SORT_N_CELLS = 200
SORT_LAMBDA_HOMO, SORT_LAMBDA_HET = 0.05, 0.45
SORT_PREFERED_PERIMETER = 3.72
SORT_LUMEN_VOL_ELASTICITY = 0.1
SORT_SURFACE_ELASTICITY = 0.2
SORT_THRESHOLD_LENGTH, SORT_P_4, SORT_P_5P = 0.08, 5.0, 0.5
SORT_DT, SORT_TF = 0.05, 200.0
SORT_METRIC_STRIDE = 4

ARCHIVE_COMPLIB, ARCHIVE_COMPLEVEL = "blosc:zstd", 5


def runs(scenario: str) -> dict:
    """``{run name: {"sigma": .., "deterministic": bool, "label": ..}}``."""
    out = {"none": {"sigma": 0.0, "deterministic": True, "label": "no noise"}}
    for s in SIGMAS:
        out[f"s{s:g}"] = {"sigma": s, "deterministic": True, "label": rf"$\sigma = {s:g}$"}
    out[f"noise_only_s{CONTROL_SIGMA:g}"] = {
        "sigma": CONTROL_SIGMA, "deterministic": False,
        "label": rf"noise only, $\sigma = {CONTROL_SIGMA:g}$"}
    return out


# ---------------------------------------------------------------------------
# Meshes
# ---------------------------------------------------------------------------
def ce_mesh() -> dict:
    """The directional_tension patch: the shared flat square with z dropped.
    Returns its mean cell area and perimeter (the preferred values)."""
    from tyssue import Sheet
    from tyssue.config.geometry import planar_spec
    from tyssue.geometry.planar_geometry import PlanarGeometry
    from tyssue.io.hdf5 import load_datasets, save_datasets

    datasets = {key: df.drop(columns=[c for c in df.columns if c == "z"], errors="ignore")
                for key, df in load_datasets(str(CE_SOURCE_MESH)).items()}
    sheet = Sheet("flat_sheet", datasets, specs=planar_spec(), coords=CE_COORDS)
    for df in sheet.datasets.values():
        df.drop(columns=[c for c in df.columns if c == "z"], inplace=True, errors="ignore")
    PlanarGeometry.update_all(sheet)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CE_MESH.exists():
        CE_MESH.unlink()
    save_datasets(str(CE_MESH), sheet)
    return {"A_0": float(sheet.face_df["area"].mean()),
            "P_0": float(sheet.face_df["perimeter"].mean())}


def sort_mesh() -> dict:
    """The differential_adhesion sphere, labelled 50:50 at random. Returns the
    lumen and surface-smoother rest values measured on it."""
    from tyssue.generation.shapes import spherical_sheet
    from tyssue.io.hdf5 import save_datasets
    from vivarium_tyssue.geometry import ClosedSheetVertDeviationGeometry

    np.random.seed(SORT_SEED)
    radius = float(np.sqrt(SORT_N_CELLS / (4 * np.pi)))
    sheet = spherical_sheet(radius=radius, Nf=SORT_N_CELLS, Lloyd_relax=True)
    ClosedSheetVertDeviationGeometry.update_all(sheet)
    rng = np.random.default_rng(SORT_SEED)
    n = len(sheet.face_df)
    labels = np.array(["A"] * (n // 2) + ["B"] * (n - n // 2), dtype=object)
    rng.shuffle(labels)
    sheet.face_df["cell_type"] = labels
    sheet.face_df["is_alive"] = 1
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SORT_MESH.exists():
        SORT_MESH.unlink()
    save_datasets(str(SORT_MESH), sheet)
    return {"lumen_vol_0": float(sheet.settings["lumen_vol"]),
            "deviation_0": float(sheet.vert_df["dev_length"].mean())}


# ---------------------------------------------------------------------------
# Composite specs
# ---------------------------------------------------------------------------
def _solver(config: dict, dt: float) -> dict:
    return {
        "_type": "process",
        "address": "local:EulerSolver",
        "config": config,
        "inputs": {"behaviors": ["Behaviors"], "global_time": ["global_time"]},
        "outputs": {"datasets": ["Tissue State"], "network_changed": ["Network Changed"],
                    "behaviors_update": ["Behaviors"]},
        "interval": dt,
    }


def _regulator(address: str, config: dict, dt: float) -> dict:
    outputs = {"behaviors": ["Behaviors"]}
    if address in ("DirectionalLineTension", "StochasticLineTension"):
        # Their parameter deltas also go to the store, so readers see them.
        outputs["datasets"] = ["Tissue State"]
    return {"_type": "process", "address": f"local:{address}", "config": config,
            "inputs": {"datasets": ["Tissue State"]},
            "outputs": outputs, "interval": dt}


def ce_spec(run: dict, mesh: dict) -> dict:
    config = {
        "name": "Directional Tension Patch",
        "eptm": str(CE_MESH),
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {"area_elasticity": 1.0, "prefered_area": mesh["A_0"],
                        "perimeter_elasticity": 0.1, "prefered_perimeter": mesh["P_0"],
                        "is_alive": 1.0},
            # line_tension starts at 0 and is the sum of the two shares from then on;
            # the share columns must exist now so History records them.
            "edge_df": {"line_tension": 0.0, "is_active": 1.0, "polar_alignment": 0.0,
                        "directional_tension": 0.0, "stochastic_tension": 0.0},
            "vert_df": {"viscosity": 1.0, "is_active": 1.0},
        },
        "geom": "PlanarGeometry",
        "effectors": ["LineTension", "FaceAreaElasticity", "PerimeterElasticity"],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory",
        "settings": {"threshold_length": CE_THRESHOLD_LENGTH, "p_4": CE_P_4, "p_5p": CE_P_5P},
        "auto_reconnect": {"enabled": True},   # a bare True becomes {}: reconnect off
        "check_intersections": False,
        "intersection_options": {},
        "bounds": {},
        "output_columns": {},
        "history_columns": {
            "vert_df": [],
            "edge_df": ["unique_id", "line_tension", "directional_tension",
                        "stochastic_tension", "polar_alignment", "length"],
            "face_df": ["unique_id", "area", "perimeter", "num_sides"],
        },
        "maps": {},
        "backend": "python",
        "substeps": 1,
        "max_displacement": 0.0,
        "record_history": True,
    }
    spec = {"Tyssue": _solver(config, CE_DT), "Network Changed": False, "Behaviors": {}}
    if run["deterministic"]:
        spec["Polarity"] = _regulator("DirectionalLineTension", {
            "polarity": list(CE_POLARITY), "tension_min": CE_LAMBDA_MIN,
            "tension_max": CE_LAMBDA_MAX, "profile": CE_PROFILE,
            "sharpness": CE_SHARPNESS, "coords": [], "record_column": "polar_alignment",
            "component": "directional_tension"}, CE_DT)
    if run["sigma"] > 0:
        spec["Stochastic"] = _regulator("StochasticLineTension", {
            "tau": TAU, "sigma": run["sigma"], "component": "stochastic_tension"}, CE_DT)
    return spec


def sort_spec(run: dict, mesh: dict) -> dict:
    config = {
        "name": "Sorting Sphere",
        "eptm": str(SORT_MESH),
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {"area_elasticity": 1.0, "prefered_area": 1.0,
                        "perimeter_elasticity": 0.1,
                        "prefered_perimeter": SORT_PREFERED_PERIMETER, "is_alive": 1.0},
            # Adhesion owns the seeded homotypic tension (its share is seeded at the
            # same value); the stochastic share starts at 0 on top of it.
            "edge_df": {"line_tension": SORT_LAMBDA_HOMO, "is_active": 1.0,
                        "heterotypic": 0.0, "adhesion_tension": SORT_LAMBDA_HOMO,
                        "stochastic_tension": 0.0},
            "vert_df": {"viscosity": 1.0, "is_active": 1.0,
                        "surface_elasticity": SORT_SURFACE_ELASTICITY,
                        "prefered_deviation": mesh["deviation_0"]},
        },
        "geom": "ClosedSheetVertDeviationGeometry",
        "effectors": ["FaceAreaElasticity", "PerimeterElasticity", "LineTension",
                      "LumenVolumeElasticity", "SurfaceElasticity"],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory",
        "settings": {"threshold_length": SORT_THRESHOLD_LENGTH, "p_4": SORT_P_4,
                     "p_5p": SORT_P_5P,
                     "lumen_vol_elasticity": SORT_LUMEN_VOL_ELASTICITY,
                     "lumen_prefered_vol": mesh["lumen_vol_0"]},
        "auto_reconnect": {"enabled": True},   # a bare True becomes {}: reconnect off
        "check_intersections": False,
        "intersection_options": {},
        "bounds": {},
        "output_columns": {},
        "history_columns": {
            "vert_df": ["dev_length"],
            "edge_df": ["line_tension", "adhesion_tension", "stochastic_tension",
                        "heterotypic", "length"],
            "face_df": ["cell_type", "area", "perimeter"],
        },
        "maps": {},
        "backend": "python",
        "substeps": 1,
        "max_displacement": 0.0,
        "record_history": True,
    }
    het = SORT_LAMBDA_HET if run["deterministic"] else SORT_LAMBDA_HOMO
    spec = {
        "Tyssue": _solver(config, SORT_DT),
        "Adhesion": _regulator("DifferentialAdhesion", {
            "homotypic_tension": SORT_LAMBDA_HOMO, "heterotypic_tension": het,
            "type_column": "cell_type", "boundary_tension": 0.0,
            "apply_boundary_tension": False, "record_column": "heterotypic",
            "component": "adhesion_tension"}, SORT_DT),
        "Network Changed": False,
        "Behaviors": {},
    }
    if run["sigma"] > 0:
        spec["Stochastic"] = _regulator("StochasticLineTension", {
            "tau": TAU, "sigma": run["sigma"], "component": "stochastic_tension"}, SORT_DT)
    return spec


SCENARIOS = {
    "ce": {"mesh": ce_mesh, "spec": ce_spec, "dt": CE_DT, "tf": CE_TF,
           "seed": CE_SEED, "share": "directional_tension"},
    "sort": {"mesh": sort_mesh, "spec": sort_spec, "dt": SORT_DT, "tf": SORT_TF,
             "seed": SORT_SEED, "share": "adhesion_tension"},
}


# ---------------------------------------------------------------------------
# Metrics (from the recorded mesh, every frame)
# ---------------------------------------------------------------------------
def opposite_positional(srce, trgt):
    """Positional index of each half-edge's opposite, -1 where there is none."""
    forward = pd.Series(np.arange(len(srce)), index=pd.MultiIndex.from_arrays([srce, trgt]))
    forward = forward[~forward.index.duplicated(keep="first")]
    opposite = forward.reindex(pd.MultiIndex.from_arrays([trgt, srce])).to_numpy()
    return np.where(np.isnan(opposite), -1, np.nan_to_num(opposite)).astype(int)


def tension_columns(e, share):
    """Per-frame tension read-outs: each share's mean and spread, the fraction of
    junctions under compression (negative tension), and the bookkeeping residual
    ``max |line_tension - share - stochastic_tension|`` (0 up to rounding)."""
    lt = e["line_tension"].to_numpy(float)
    det = e[share].to_numpy(float) if share in e else np.zeros_like(lt)
    sto = e["stochastic_tension"].to_numpy(float)
    return {
        "tension_mean": float(lt.mean()), "tension_std": float(lt.std()),
        "share_mean": float(det.mean()), "share_std": float(det.std()),
        "stochastic_mean": float(sto.mean()), "stochastic_std": float(sto.std()),
        "negative_fraction": float((lt < 0).mean()),
        "residual": float(np.abs(lt - det - sto).max()),
    }


def _second_moment_axes(cxx, cxy, cyy):
    tr, det = cxx + cyy, cxx * cyy - cxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4.0 - det, 0.0))
    l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc
    return l1, l2, 0.5 * np.arctan2(2.0 * cxy, cxx - cyy)


def ce_metrics(history, stride=CE_METRIC_STRIDE) -> pd.DataFrame:
    """directional_tension's ``run_metrics``: tissue and cell shape in the polarity
    frame, junction nematic order and completed T1s, plus the tension read-outs."""
    p = np.asarray(CE_POLARITY, float)
    p = p / np.linalg.norm(p)
    q = np.array([-p[1], p[0]])
    phi_p = np.arctan2(p[1], p[0])
    vert_by_t = dict(tuple(history.datasets["vert"].groupby("time")))
    face_by_t = dict(tuple(history.datasets["face"].groupby("time")))

    seen, n_t1, rows = None, 0, []
    for i, (t, e) in enumerate(history.datasets["edge"].groupby("time")):
        v = vert_by_t[t].sort_values("vert")
        pos = np.empty((int(v["vert"].max()) + 1, 2))
        pos[v["vert"].to_numpy()] = v[CE_COORDS].to_numpy(float)
        srce, trgt = e["srce"].to_numpy(), e["trgt"].to_numpy()

        # completed T1s: pairs of cells newly adjacent (every frame)
        face_uid = face_by_t[t].set_index("face")["unique_id"]
        opp = opposite_positional(srce, trgt)
        interior = opp >= 0
        own = face_uid.reindex(e["face"].to_numpy()).to_numpy()
        other = own[np.where(interior, opp, 0)]
        pairs = {frozenset(pr) for pr in
                 zip(own[interior].astype(int), other[interior].astype(int))}
        if seen is None:
            seen = set(pairs)
        else:
            n_t1 += len(pairs - seen)
            seen |= pairs
        if i % stride:
            continue

        used = np.unique(np.r_[srce, trgt])
        xy = pos[used] - pos[used].mean(axis=0)
        L_par, L_perp = 2 * (xy @ p).std(), 2 * (xy @ q).std()

        d = pd.DataFrame({"face": e["face"].to_numpy(), "x": pos[srce, 0], "y": pos[srce, 1]})
        d["xx"], d["xy"], d["yy"] = d.x * d.x, d.x * d.y, d.y * d.y
        m = d.groupby("face").mean()
        l1, l2, ang = _second_moment_axes(m.xx - m.x ** 2, m.xy - m.x * m.y, m.yy - m.y ** 2)
        ok = l2 > 1e-12
        nem = (l1 - l2) / (l1 + l2)

        dvec = pos[trgt] - pos[srce]
        length = np.linalg.norm(dvec, axis=1)
        phi = np.arctan2(dvec[:, 1], dvec[:, 0])
        w = length / length.sum()
        rows.append({
            "time": float(t),
            "L_parallel": float(L_par), "L_perp": float(L_perp),
            "aspect": float(L_perp / L_par),
            "n_cells": int(face_by_t[t]["face"].nunique()),
            "n_t1": int(n_t1),
            "cell_elongation": float(np.mean(np.sqrt(l1[ok] / l2[ok]))),
            "S_cell": float(np.mean(nem[ok] * np.cos(2 * (ang[ok] - phi_p)))),
            "S_junction": float(np.sum(w * np.cos(2 * (phi - phi_p)))),
            **tension_columns(e, "directional_tension"),
        })
    out = pd.DataFrame(rows)
    out["ce_index"] = out["aspect"] / out["aspect"].iloc[0]
    return out


def sort_metrics(history, stride=SORT_METRIC_STRIDE) -> pd.DataFrame:
    """differential_adhesion's ``interface_metrics`` (sorting index, heterotypic
    interface length), surface roughness, plus the tension read-outs."""
    face_by_t = dict(tuple(history.datasets["face"].groupby("time")))
    rough = history.datasets["vert"].groupby("time")["dev_length"].mean()
    rows = []
    for i, (t, e) in enumerate(history.datasets["edge"].groupby("time")):
        if i % stride:
            continue
        types = face_by_t[t].set_index("face")["cell_type"]
        own = types.reindex(e["face"].to_numpy()).to_numpy()
        opp = opposite_positional(e["srce"].to_numpy(), e["trgt"].to_numpy())
        has_opp = opp >= 0
        hetero = has_opp & (own != own[np.where(has_opp, opp, 0)])
        lengths = e["length"].to_numpy()
        rows.append({
            "time": float(t),
            "n_junctions": len(e) // 2,
            "n_heterotypic": int(hetero.sum()) // 2,
            "sorting_index": 1.0 - hetero.sum() / len(e),
            "heterotypic_length": float(lengths[hetero].sum()) / 2,
            "roughness": float(rough.loc[t]),
            **tension_columns(e, "adhesion_tension"),
        })
    return pd.DataFrame(rows)


METRICS = {"ce": ce_metrics, "sort": sort_metrics}


# ---------------------------------------------------------------------------
# Archiving
# ---------------------------------------------------------------------------
def save_history(history, path: Path, keep_frames: int = ARCHIVE_FRAMES):
    """``history.datasets`` thinned to ``keep_frames`` frames, as a compressed HDF5
    with tyssue's ``History.to_archive`` keys (``HistoryHdf5.from_archive`` reopens
    it). Object columns (``cell_type``) are written as strings."""
    datasets = history.datasets
    times = np.array(list(history.time_stamps))
    if times.size > keep_frames:
        idx = np.unique(np.round(np.linspace(0, times.size - 1, keep_frames)).astype(int))
        keep = set(times[idx].tolist())
        datasets = {k: df[df["time"].isin(keep)] for k, df in datasets.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with pd.HDFStore(path, "a", complevel=ARCHIVE_COMPLEVEL, complib=ARCHIVE_COMPLIB) as store:
        for key, df in datasets.items():
            df = df.copy()
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].astype(str)
            store.append(key=key, value=df, data_columns=["time"])


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def run_case(scenario: str, name: str, mesh: dict) -> dict:
    """One composite run: simulate, compute metrics, archive. Runs in a worker."""
    from process_bigraph import Composite
    from vivarium_tyssue.core import build_core

    sc = SCENARIOS[scenario]
    run = runs(scenario)[name]
    np.random.seed(sc["seed"])
    random.seed(sc["seed"])          # tyssue's EventManager shuffles with `random`
    sim = Composite({"state": sc["spec"](run, mesh)}, core=build_core())
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sim.run(sc["tf"])
    eptm = sim.state["Tyssue"]["instance"].eptm
    history = sim.state["Tyssue"]["instance"].history
    history.update_datasets()
    elapsed = time.time() - t0

    metrics = METRICS[scenario](history)
    metrics.insert(0, "run", name)
    save_history(history, OUT_DIR / scenario / name / "history.hf5")
    return {
        "scenario": scenario, "run": name, **run,
        "seconds": round(elapsed, 1),
        "frames": len(list(history.time_stamps)),
        "final_valid": bool(eptm.validate()),
        "final_cells": int(eptm.Nf),
        "max_residual": float(metrics["residual"].max()),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=sorted(SCENARIOS), action="append")
    parser.add_argument("--jobs", type=int, default=min(10, os.cpu_count() or 1))
    args = parser.parse_args()
    scenarios = args.only or list(SCENARIOS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meshes = {s: SCENARIOS[s]["mesh"]() for s in scenarios}
    for s in scenarios:
        shutil.rmtree(OUT_DIR / s, ignore_errors=True)
        (OUT_DIR / f"mesh_{s}.json").write_text(json.dumps(meshes[s], indent=2))

    jobs = [(s, name) for s in scenarios for name in runs(s)]
    print(f"{len(jobs)} runs on {args.jobs} workers (tau = {TAU}, sigmas = {SIGMAS})",
          flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_case, s, name, meshes[s]): (s, name) for s, name in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            m = r["metrics"].iloc[-1]
            key = "ce_index" if r["scenario"] == "ce" else "sorting_index"
            print(f"[{r['scenario']:>4} {r['run']:>17}] {r['seconds']:6.0f}s  "
                  f"{key} {m[key]:.3f}  valid {r['final_valid']}  "
                  f"residual {r['max_residual']:.1e}", flush=True)
            results.append(r)

    for s in scenarios:
        mine = [r for r in results if r["scenario"] == s]
        order = list(runs(s))
        mine.sort(key=lambda r: order.index(r["run"]))
        pd.concat([r["metrics"] for r in mine], ignore_index=True).to_csv(
            OUT_DIR / f"metrics_{s}.csv", index=False)
        summary = [{k: v for k, v in r.items() if k != "metrics"} for r in mine]
        (OUT_DIR / f"runs_{s}.json").write_text(json.dumps(summary, indent=2))
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
