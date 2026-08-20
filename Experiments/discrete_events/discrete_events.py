"""Discrete-event experiments — random divisions, random deaths, Gillespie (SIMULATION).

Three scenarios exercising the discrete-event processes:

  * **divisions** — a ``CellDivisions`` process fires cell divisions as a Poisson
    process (rate-based random times) on a plain **flat square sheet**
    (``test_square.hf5``, ``SheetGeometry``). Every cell starts ``"normal"``; a cell
    being actively grown toward division is flagged ``"dividing"``.
  * **deaths** — a ``CellDeaths`` process fires apoptotic extrusions as a Poisson
    process on the same flat square; a dying cell is flagged ``"extruding"``.
  * **gillespie** — the full Gillespie biochemistry (``Gillespie`` process) on the
    3-D crypt cylinder (``crypt_cylinder.hf5``) as in ``tests/tests.py`` /
    ``Notebooks/simulation_walkthrough.ipynb``, run to ``tf=72`` with a mechanics
    step of ``GILL_SOLVER_DT`` = 0.001 (see the constant for why).
  * **gillespie_restart** — the *same* model and the *same* parameters, restarted
    from the settled crypt the ``gillespie`` run ends on (checkpointed to
    ``outputs/gillespie/stable_eptm.hf5``), so the second run begins where the
    cell-type populations have already plateaued instead of from the uniform
    initial mesh.

This script **only runs the simulations and archives their data**: each scenario's
full-resolution (crypt: capped, see ``GILL_ARCHIVE_FRAMES``) ``History`` to a
compressed HDF5 file (``outputs/<scenario>/history.hf5``), plus — for the crypt
scenarios — the emitted discrete events to ``outputs/<scenario>/events.csv``
(events come from the process emitter, not the History, so they are saved
separately) and, for ``gillespie``, the restart mesh ``stable_eptm.hf5``.

All visualisation (2-D / 3-D colour-coded GIFs and stills) and analysis (cell-type
distributions over time and along z, event-type distribution along z) live in the
companion notebook ``discrete_events_analysis.ipynb``, which reopens the archives
with ``tyssue``'s ``HistoryHdf5.from_archive``.

Everything lands under ``outputs/`` (input meshes under ``data/``), both git-ignored.
Run from the repo's ``vivarium-tyssue`` conda env:

    conda activate vivarium-tyssue
    cd Experiments/discrete_events
    python discrete_events.py            # runs all four, in order
    python discrete_events.py divisions  # or a single scenario:
                                         #   divisions|deaths|gillespie|gillespie_restart
"""
from __future__ import annotations

import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"
FLAT_DATASET = "test_square.hf5"        # divisions / deaths (flat sheet)
CRYPT_DATASET = "crypt_cylinder.hf5"    # gillespie (3-D crypt)

COORDS_3D = ["x", "y", "z"]
COORDS_2D = ["x", "y"]
SEED = 20260715

# Discrete-event scenarios on the flat square (Poisson processes).
DIV_TF, DIV_DT = 50.0, 0.05
DIV_RATE, DIV_CRIT, DIV_GROWTH = 0.4, 2.0, 0.3           # CellDivisions
DEATH_TF, DEATH_DT = 50.0, 0.05
DEATH_RATE, DEATH_CRIT, DEATH_SHRINK = 0.4, 0.2, 0.3     # CellDeaths
# Two settings are needed for DEATH_CRIT to be the threshold that actually removes a
# cell here. (1) A dying cell is also removed once its shrinking prefered_area passes
# a floor (behaviors.DEATH_FLOOR = 0.5 by default) — on this relaxed sheet that fired
# while the cell was still near full size (measured: removal at area ~0.83), so
# DEATH_CRIT never governed; the floor is dropped well below it. (2) A cell holding
# its full prefered_perimeter (3.6) cannot shrink past area ~0.35 however small its
# area target gets, so DEATH_CONTRACT constricts the dying cell's target perimeter at
# the same rate, letting it reach DEATH_CRIT and be removed there.
DEATH_FLOOR = 0.01
DEATH_CONTRACT = True

# Gillespie scenario. The mechanics step and the event step are *separate clocks*:
#
#   * GILL_SOLVER_DT — the EulerSolver's interval, i.e. the explicit-Euler dt.
#     Lowered 0.005 -> 0.001. At 0.005 the transient gradient that follows a
#     division / extrusion could displace a single vertex far enough in one step
#     to push it off the tube and let neighbouring faces overlap (visible as a
#     spike at the crypt mouth in the t=72 still). Explicit Euler's displacement
#     per step is linear in dt, so a 5x smaller step shrinks those transients 5x.
#   * The Gillespie clock is NOT this one and does not need changing: the process
#     overrides ``calculate_timestep``, which process_bigraph consults before every
#     one of its updates, and returns a true SSA waiting time -ln(u)/sum(rate_max)
#     (~0.004 here). Its spec ``interval`` below is only a placeholder the
#     scheduler immediately replaces, so the event statistics are set by the rates
#     alone and are unaffected by the solver step.
#   * The one real coupling is the Gillespie's ``global_interval``: it is handed to
#     the division / extrusion behaviours as their per-step ``dt``, and the
#     committed-cell grower runs once per *solver* step. So it has to track
#     GILL_SOLVER_DT, which keeps a committed cell's growth / shrinkage per unit
#     time identical to the dt=0.005 run.
GILL_TF = 72.0
GILL_SOLVER_DT = 0.001
GILL_EVENT_DT = 0.005          # placeholder only — see calculate_timestep above

# --- division mechanics -----------------------------------------------------
# A committed cell's prefered_area ramps at GILL_GROWTH_RATE until its ACTUAL area
# crosses GILL_DIVISION_CRIT, at which point tyssue's cell_division halves it.
#
# GILL_DIVISION_CRIT was 1.2. With prefered_area = 1.0 that left each daughter at
# ~0.6 against a target of 1.0, i.e. an area-elastic energy of
# ½·K_A·(0.6-1.0)² = 0.08 per daughter appearing in a single step, which the pair
# then relieves by expanding fast against their neighbours. Splitting at 2.0
# instead lands both daughters at ~1.0 = prefered_area, so the division is
# energy-neutral and nothing has to re-inflate.
GILL_DIVISION_CRIT = 2.0
# Raised from 0.02 to hold the division rate. A committed cell realises only ~0.78
# of its prefered_area here (measured median; its neighbours resist), so A_0 has to
# reach crit/0.78, and the ramp A_0(t) = exp(rate·t) needs ~47 time units at 0.02 to
# clear 2.0 — most of the run. Measured over tf=45: at 0.02 the crit=2.0 tissue
# completes *zero* divisions, while 0.045 gives 87 against the old
# crit=1.2/rate=0.02 baseline's 79, i.e. the original cadence is preserved.
GILL_GROWTH_RATE = 0.045
GILL_SHRINK_RATE = 0.02        # extrusion is unaffected by the division threshold
# Area-elastic modulus, per face. Sets how hard a cell holds its target area
# against the perimeter (0.5) and vessel (1.0) terms. Left at 1.0: raising it to 2.0
# was measured to be strictly worse — it does not land daughters any closer to their
# target (the residual is the uneven bisection, not compliance), so at fixed area
# error the energy jump just scales with the modulus (mean 0.022 -> 0.040, max
# 0.17 -> 0.39) and folding rose with it.
GILL_AREA_ELASTICITY = 1.0

# Vertex drag. dot_r = -grad(E)/viscosity, so this is the mechanical relaxation
# timescale: raising it slows the tissue's response to a force without changing the
# equilibrium it relaxes towards. The Gillespie's event rates are fixed in absolute
# time, so this knob also sets the ratio of mechanical to biological timescales.
#
# Left at 0.05 rather than the composite's 5.0: measured over tf=45, raising it
# makes folding WORSE at matched vessel_elasticity (integrated folded cell-frames
# 12.5k at visc=0.05/vess=5 vs 18.9k at visc=5/vess=5). The composite raised it to
# damp an everting rim, but that failure mode does not occur here (face radius stays
# <=2.6 at 0.05); all it does in this configuration is slow the tissue's relief of
# post-division crowding by 100x while the event rates stay fixed in absolute time,
# so the crowding folds the sheet instead of relaxing out of it.
GILL_VISCOSITY = 0.05
# Modulus of VesselSurfaceElasticity, which pins each vertex radially to
# prefered_radius (2.5). Read from the `vessel_elasticity` column, NOT
# `surface_elasticity`.
#
# Raised 1.0 -> 10.0. This is the single most effective knob found against the
# self-intersecting ("bow-tie") cells: integrated folded cell-frames over tf=45 fall
# 28.5k (1.0) -> 13.5k (5.0) -> 7.2k (10.0), with worst-case folded cells per frame
# 61 -> 25 -> 14. It saturates there: 20.0 gives 6.9k / 15, no better. Divisions
# (93) and daughter relaxation to prefered_area (1.001) are unaffected, so this is
# genuine stiffening of the radial constraint, not the tissue being frozen.
#
# NOTE this reduces folding ~4x, it does not eliminate it: folds still appear from
# t~24.7 and persist. See the crypt-cell-overlaps notes.
GILL_VESSEL_ELASTICITY = 10.0

# The tyssue History snapshots the whole mesh on every solver step and holds it in
# RAM, so the 5x finer step would cost 5x the memory for no extra information.
# Record every GILL_RECORD_EVERY-th step instead, which keeps the recording cadence
# (and the memory footprint) exactly what it was at dt=0.005.
GILL_RECORD_EVERY = int(round(GILL_EVENT_DT / GILL_SOLVER_DT))

# Archiving. The flat-sheet scenarios (~1000 frames) are archived in full. The
# gillespie crypt records ~14400 frames, so its History is capped to a still-dense
# subsample (the GIF only needs ~120 frames and the cell-type distributions stay
# smooth). The discrete *events* are always saved in full (from the emitter). Set a
# scenario's cap to ``None`` to keep every frame.
FLAT_ARCHIVE_FRAMES: int | None = None
GILL_ARCHIVE_FRAMES: int | None = 1500
ARCHIVE_COMPLIB = "blosc:zstd"
ARCHIVE_COMPLEVEL = 5

# ``gillespie_restart`` — same model, same parameters, same steps; only the initial
# tissue differs. It starts from the *relaxed, compositionally settled* crypt that
# ``gillespie`` ends on (cell-type counts have plateaued by then), checkpointed to
# ``outputs/gillespie/stable_eptm.hf5``. Set GILL_STABLE_TIME to a float to
# checkpoint that recorded timepoint instead of the final state.
GILL_STABLE_MESH = "stable_eptm.hf5"
GILL_STABLE_TIME: float | None = None
GILL_RESTART_TF = GILL_TF


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def ensure_dataset(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_DIR / name
    if not dst.exists():
        src = REPO / "workspace" / "datasets" / name
        if not src.exists():
            raise FileNotFoundError(f"source mesh not found: {src}")
        shutil.copy(src, dst)
        print(f"copied {src} -> {dst}")
    return dst


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------
def flat_config(dataset_path: Path) -> dict:
    """Plain flat square sheet (SheetGeometry / model_factory_bound). Every cell
    starts ``cell_type="normal"`` so the division / extrusion behaviours can flag the
    active cell for colour-coding."""
    return {
        "name": "Flat Square",
        "eptm": str(dataset_path),
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {
                "area_elasticity": 1.0,
                "prefered_area": 1.0,
                "perimeter_elasticity": 0.1,
                "prefered_perimeter": 3.6,
                "is_alive": 1.0,
                "cell_type": "normal",
            },
            "edge_df": {"line_tension": 0.0, "is_active": 1.0},
            "vert_df": {"viscosity": 1.0, "is_alive": 1.0},
        },
        "geom": "SheetGeometry",
        "effectors": ["LineTension", "FaceAreaElasticity", "PerimeterElasticity"],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory_bound",
        "settings": {"threshold_length": 0.03},
        "auto_reconnect": True,
        "bounds": None,
        "output_columns": {},
        "history_columns": {},
        "maps": {},
        # topology-mutating behaviours (division / extrusion) run safest on python.
        "backend": "python",
        "substeps": 1,
        "max_displacement": 0.0,
        "record_history": True,
    }


def crypt_config(dataset_path: Path) -> dict:
    """The crypt-cylinder EulerSolver config used by the Gillespie model
    (VesselGeometry / model_factory_vessel)."""
    return {
        "name": "Crypt Cylinder",
        "eptm": str(dataset_path),
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {
                "area_elasticity": GILL_AREA_ELASTICITY,
                "prefered_area": 1.0,
                "perimeter_elasticity": 0.5,
                "prefered_perimeter": 3.5,
            },
            "edge_df": {"line_tension": 0.0, "is_active": 1.0},
            "vert_df": {
                "viscosity": GILL_VISCOSITY,
                "vessel_elasticity": GILL_VESSEL_ELASTICITY,
                "prefered_radius": 2.5,
                "is_alive": 1.0,
                "surface_elasticity": 0.1,
            },
        },
        "geom": "VesselGeometry",
        "effectors": ["FaceAreaElasticity", "PerimeterElasticity", "LineTension", "VesselSurfaceElasticity"],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory_vessel",
        "settings": {"threshold_length": 0.03, "radius": 2.5, "axis": "z"},
        "auto_reconnect": True,
        "bounds": None,
        "output_columns": {},
        "history_columns": {},
        "maps": {},
        # VesselGeometry + model_factory_vessel run on the rust compositional
        # gradient path (kernels.rust_model_gradient) with rust geometry — ~2x
        # faster than python here. Topology-change staleness (division / extrusion
        # / rosette detach) is handled, so long crypt runs stay stable.
        "backend": "rust",
        "substeps": 1,
        "max_displacement": 0.0,
        "record_history": True,
    }


def _solver_node(config: dict, dt: float) -> dict:
    return {
        "Tyssue": {
            "_type": "process",
            "address": "local:EulerSolver",
            "config": config,
            "inputs": {"behaviors": ["Behaviors"], "global_time": ["global_time"]},
            "outputs": {
                "datasets": ["Tissue State"],
                "network_changed": ["Network Changed"],
                "behaviors_update": ["Behaviors"],
            },
            "interval": dt,
        },
        "Network Changed": False,
        "Behaviors": {},
    }


def build_divisions_spec(dataset_path: Path) -> dict:
    spec = _solver_node(flat_config(dataset_path), DIV_DT)
    spec["Divisions"] = {
        "_type": "process",
        "address": "local:CellDivisions",
        "config": {
            "rate": DIV_RATE,
            "geom": "SheetGeometry",
            "crit_area": DIV_CRIT,
            "growth_rate": DIV_GROWTH,
        },
        "inputs": {"global_time": ["global_time"], "datasets": ["Tissue State"]},
        "outputs": {"behaviors": ["Behaviors"]},
        "interval": DIV_DT,
    }
    return spec


def build_deaths_spec(dataset_path: Path) -> dict:
    spec = _solver_node(flat_config(dataset_path), DEATH_DT)
    spec["Deaths"] = {
        "_type": "process",
        "address": "local:CellDeaths",
        "config": {
            "rate": DEATH_RATE,
            "geom": "SheetGeometry",
            "crit_area": DEATH_CRIT,
            "shrink_rate": DEATH_SHRINK,
            "death_floor": DEATH_FLOOR,
            "contract_perimeter": DEATH_CONTRACT,
        },
        "inputs": {"global_time": ["global_time"], "datasets": ["Tissue State"]},
        "outputs": {"behaviors": ["Behaviors"]},
        "interval": DEATH_DT,
    }
    return spec


def build_gillespie_spec(dataset_path: Path) -> dict:
    from vivarium_tyssue.models.crypt_gillespie.crypt_params import cell_types
    from vivarium_tyssue.models.crypt_gillespie.jump_rates import (
        rates_max, K, k, regulations, regulation_loc,
    )

    spec = _solver_node(crypt_config(dataset_path), GILL_SOLVER_DT)
    spec["Gillespie"] = {
        "_type": "process",
        "address": "local:Gillespie",
        "config": {
            "cell_types": cell_types,
            "rates_max": rates_max,
            "michaelis_constants": K,
            "transition_lengths": k,
            "geom": "VesselGeometry",
            # per-solver-step dt for the committed-cell grower (see GILL_SOLVER_DT)
            "global_interval": GILL_SOLVER_DT,
            "growth_rate": GILL_GROWTH_RATE,
            "shrink_rate": GILL_SHRINK_RATE,
            "division_crit": GILL_DIVISION_CRIT,
            "apoptosis_crit": 0.1,
            "regulations": regulations,
            "regulation_loc": regulation_loc,
        },
        "inputs": {
            "datasets": ["Tissue State"],
            "behaviors": ["Behaviors"],
            "global_time": ["global_time"],
        },
        "outputs": {
            "behaviors": ["Behaviors"],
            "gillespie_trigger": ["Gillespie Trigger"],
        },
        # superseded every step by Gillespie.calculate_timestep (SSA waiting time)
        "interval": GILL_EVENT_DT,
    }
    return spec


def run(core, spec: dict, tf: float, capture_behaviors: bool = False,
        record_every: int = 1):
    """Run a spec; return (history, events, eptm).

    ``events`` is a list of emitted behaviour dicts (only when
    ``capture_behaviors``); ``eptm`` is the solver's live epithelium at ``tf``.
    ``record_every`` > 1 snapshots the History only every N-th solver step —
    tyssue's ``History`` supports this through ``save_every``/``dt``, but
    ``EulerSolver`` builds it with the defaults, so set them on the instance
    before the run (the solver never touches either attribute afterwards)."""
    from process_bigraph import Composite
    from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

    spec = copy.deepcopy(spec)
    if capture_behaviors:
        spec["emitter"] = emitter_from_wires({
            "global_time": ["global_time"],
            "behaviors": ["Behaviors"],
        })
    sim = Composite({"state": spec}, core=core)
    solver = sim.state["Tyssue"]["instance"]
    if record_every > 1 and solver.history is not None:
        solver.history.dt = spec["Tyssue"]["interval"]
        solver.history.save_every = record_every * solver.history.dt
    sim.run(tf)
    history = solver.history
    history.update_datasets()

    events = []
    if capture_behaviors:
        results = gather_emitter_results(sim)[("emitter",)]
        events = _collect_events(results)
    return history, events, solver.eptm


def _collect_events(results) -> list:
    """Flatten emitter frames into distinct (time, func, cell_uid) event records."""
    seen = set()
    events = []
    for frame in results:
        t = float(frame.get("global_time", 0.0))
        behaviors = frame.get("behaviors", [])
        if isinstance(behaviors, dict):
            behaviors = list(behaviors.values())
        for b in behaviors or []:
            if not isinstance(b, dict) or "func" not in b:
                continue
            key = (round(t, 5), b.get("func"), b.get("cell_uid"))
            if key in seen:
                continue
            seen.add(key)
            events.append({"time": t, "func": b["func"], "cell_uid": b.get("cell_uid")})
    return events


# ---------------------------------------------------------------------------
# History archiving (compressed; optionally thinned)
# ---------------------------------------------------------------------------
def _sanitize_for_hdf(df: pd.DataFrame) -> pd.DataFrame:
    """Make an element dataframe serialisable by PyTables' table format.

    The crypt's ``cell`` element carries object-dtype columns whose contents are
    plain integers (e.g. the ``cell`` index column); PyTables refuses to serialise
    an object column that is neither all-string nor a real numeric dtype. Coerce each
    object column to numeric where every non-null value converts, otherwise to str.
    Non-object columns are left untouched, so already-clean archives are unchanged.
    """
    df = df.copy()
    for c in df.columns:
        if df[c].dtype != object:
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        notnull = df[c].notna()
        if notnull.any() and (conv.notna() | ~notnull).all():
            df[c] = conv            # genuinely numeric (ints/floats stored as object)
        else:
            df[c] = df[c].astype(str)   # strings (e.g. cell_type)
    return df


def settle_pending_commitments(eptm) -> int:
    """Un-commit every cell that is mid-division / mid-extrusion, in place.

    A commitment lives in two places: the ``commit_*`` face columns (persisted with
    the mesh) and the grower queued on the solver's ``EventManager`` (not
    persisted). A checkpoint therefore can't carry a commitment across runs — a
    cell left flagged ``"dividing"`` / ``"extruding"`` would keep that label
    forever, and the Gillespie would never pick it again (neither label is in
    ``cell_types``). Restore each such cell's real type from ``commit_type`` and
    clear the flags; the shrunk / grown ``prefered_area`` needs no fixing, since
    ``EulerSolver`` re-applies the configured face parameters on load.

    Returns the number of cells settled."""
    fd = eptm.face_df
    if "commit_state" not in fd.columns:
        return 0
    pending = fd.index[fd["commit_state"].to_numpy(dtype=float) != 0.0]
    if not len(pending):
        return 0
    restore = fd.loc[pending, "commit_type"].astype(str)
    known = restore[(restore != "") & (restore != "nan")]
    fd.loc[known.index, "cell_type"] = known
    unknown = pending.difference(known.index)
    if len(unknown):
        print(f"  warning: {len(unknown)} committed cell(s) had no commit_type to "
              f"restore; leaving their cell_type as-is")
    for col, default in (("commit_state", 0.0), ("commit_rate", 0.0),
                         ("commit_crit", 0.0), ("commit_dt", 0.0),
                         ("commit_contract", 0.0)):
        if col in fd.columns:
            fd.loc[pending, col] = default
    fd.loc[pending, "commit_type"] = ""
    return len(pending)


def save_checkpoint(eptm, path: Path):
    """Write the epithelium out as a plain tyssue mesh (the same
    ``vert``/``edge``/``face`` layout ``load_datasets`` reads), so another run can
    take it as its ``eptm`` starting point. Pending commitments are settled first."""
    from tyssue.io.hdf5 import save_datasets

    settled = settle_pending_commitments(eptm)
    if settled:
        print(f"  settled {settled} pending division/extrusion commitment(s)")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    save_datasets(str(path), eptm)


def save_history(history, path: Path, keep_frames: int | None):
    """Archive ``history.datasets`` to a compressed HDF5 file (same per-element keys
    as tyssue's ``History.to_archive`` so the notebook can reopen it with
    ``HistoryHdf5.from_archive``). ``keep_frames`` optionally thins to that many
    subsampled timepoints; ``None`` keeps the full history. The live history is not
    mutated. Returns the number of frames actually written, which is what the
    callers report -- ``history.time_stamps`` is the *live* count and overstates
    the archive whenever ``keep_frames`` thinned it."""
    datasets = history.datasets
    times = np.array(list(history.time_stamps))
    if keep_frames is not None and times.size > keep_frames:
        idx = np.unique(np.round(np.linspace(0, times.size - 1, keep_frames)).astype(int))
        keep = set(times[idx].tolist())
        datasets = {k: df[df["time"].isin(keep)] for k, df in datasets.items()}

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with pd.HDFStore(path, "a", complevel=ARCHIVE_COMPLEVEL, complib=ARCHIVE_COMPLIB) as store:
        for key, df in datasets.items():
            df = _sanitize_for_hdf(df)
            kwargs = {"data_columns": ["time"]}
            if "segment" in df.columns:
                kwargs["min_itemsize"] = {"segment": 7}
            store.append(key=key, value=df, **kwargs)
    return len(next(iter(datasets.values()))["time"].unique())


# ---------------------------------------------------------------------------
# Scenario drivers
# ---------------------------------------------------------------------------
def run_divisions(core):
    dataset = ensure_dataset(FLAT_DATASET)
    out = OUT_DIR / "divisions"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[divisions] flat square, tf={DIV_TF} dt={DIV_DT} rate={DIV_RATE} ...", flush=True)
    history, _, _ = run(core, build_divisions_spec(dataset), DIV_TF)
    n_frames = save_history(history, out / "history.hf5", FLAT_ARCHIVE_FRAMES)
    print(f"[divisions] archived {n_frames} frames -> {out / 'history.hf5'}", flush=True)


def run_deaths(core):
    dataset = ensure_dataset(FLAT_DATASET)
    out = OUT_DIR / "deaths"
    out.mkdir(parents=True, exist_ok=True)
    print(f"[deaths] flat square, tf={DEATH_TF} dt={DEATH_DT} rate={DEATH_RATE} ...", flush=True)
    history, _, _ = run(core, build_deaths_spec(dataset), DEATH_TF)
    n_frames = save_history(history, out / "history.hf5", FLAT_ARCHIVE_FRAMES)
    print(f"[deaths] archived {n_frames} frames -> {out / 'history.hf5'}", flush=True)


def _run_crypt(core, tag: str, dataset: Path, tf: float, checkpoint: bool):
    """Run the Gillespie crypt from ``dataset`` and archive it under
    ``outputs/<tag>/``. Shared by the ``gillespie`` and ``gillespie_restart``
    scenarios, which differ only in their starting mesh."""
    out = OUT_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"[{tag}] crypt from {dataset.name}, tf={tf} solver dt={GILL_SOLVER_DT} "
          f"(recording every {GILL_RECORD_EVERY} steps) ...", flush=True)
    history, events, eptm = run(
        core, build_gillespie_spec(dataset), tf,
        capture_behaviors=True, record_every=GILL_RECORD_EVERY,
    )
    print(f"[{tag}] captured {len(events)} events", flush=True)
    n_frames = save_history(history, out / "history.hf5", GILL_ARCHIVE_FRAMES)
    pd.DataFrame(events).to_csv(out / "events.csv", index=False)
    print(f"[{tag}] archived {n_frames} of {len(list(history.time_stamps))} recorded "
          f"frames -> {out / 'history.hf5'}", flush=True)
    print(f"[{tag}] wrote {out / 'events.csv'}", flush=True)

    if checkpoint:
        # The settled tissue this run ends on, as the next run's starting mesh.
        if GILL_STABLE_TIME is not None:
            eptm = history.retrieve(GILL_STABLE_TIME)
            print(f"[{tag}] checkpointing recorded frame nearest t={GILL_STABLE_TIME}", flush=True)
        path = out / GILL_STABLE_MESH
        save_checkpoint(eptm, path)
        print(f"[{tag}] checkpointed {len(eptm.face_df)} cells -> {path}", flush=True)


def run_gillespie(core):
    _run_crypt(core, "gillespie", ensure_dataset(CRYPT_DATASET), GILL_TF, checkpoint=True)


def run_gillespie_restart(core):
    """Same model and parameters as ``gillespie``; only the initial tissue differs
    — it starts from the checkpoint that run ends on.

    It writes its own checkpoint too (``outputs/gillespie_restart/stable_eptm.hf5``),
    so a still-drifting population can be settled further by pointing another run at
    it — the first run's tf=72 leaves ``pc`` still climbing, so its checkpoint is
    close to but not exactly on the stationary composition."""
    mesh = OUT_DIR / "gillespie" / GILL_STABLE_MESH
    if not mesh.exists():
        raise SystemExit(
            f"no checkpoint at {mesh} — run `python discrete_events.py gillespie` first"
        )
    _run_crypt(core, "gillespie_restart", mesh, GILL_RESTART_TF, checkpoint=True)


SCENARIOS = {
    "divisions": run_divisions,
    "deaths": run_deaths,
    "gillespie": run_gillespie,
    "gillespie_restart": run_gillespie_restart,
}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in SCENARIOS and which != "all":
        raise SystemExit(f"unknown scenario '{which}'; choose from {list(SCENARIOS)} or 'all'")

    np.random.seed(SEED)
    sys.path.insert(0, str(REPO))
    from vivarium_tyssue.core import build_core

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    core = build_core()

    todo = list(SCENARIOS) if which == "all" else [which]
    for name in todo:
        SCENARIOS[name](core)

    print(f"\ndone — archives under {OUT_DIR}")
    print("Run discrete_events_analysis.ipynb to visualise and analyse the scenarios.")


if __name__ == "__main__":
    main()
