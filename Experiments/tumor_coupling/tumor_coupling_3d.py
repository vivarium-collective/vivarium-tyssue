"""Tumor-coupling experiment (3D monolayer) — a COPASI population ODE drives a
volumetric vertex model (SIMULATION).

The 2D companion (``tumor_coupling.py``) couples the breast-cancer population ODE
(BioModels ``BIOMD0000000903``, integrated in COPASI) to a flat tyssue *Sheet*:
each step the ``TumorCoupling`` process reads the SBML per-reaction birth/death
fluxes and fires ``floor(flux * scale * dt)`` discrete vertex-model events on the
mesh. This script runs the SAME coupling on a **3D monolayer** (``monolayer_box.hf5``,
``MonolayerGeometry``) so the tumor grows as a genuine 3D mass:

  * **births** -> real 3D ``cell_division`` (``division_3d``): the cell grows its
    prefered VOLUME (the ``CellVolumeElasticity`` reference) until its measured
    ``vol`` crosses ``DIVISION_CRIT`` — the volumetric analogue of the 2D critical
    area — then splits with tyssue's monolayer ``cell_division``.
  * **orientation** is RANDOMIZED per division (``vertical`` in-plane vs
    ``horizontal`` stacking a daughter in z), so the clone is not confined to one
    plane and builds up in 3D.
  * **deaths** -> ``apoptosis_3d``: the cell shrinks and is marked dead / necrotic
    (kept in the mesh; volumetric extrusion is numerically fragile in 3D).
  * **tumor induction** -> ``differentiation`` of a cancer stem cell into a tumor cell.

A compact central **cancer-stem-cell** focus is seeded (matching the SBML model's
nonzero initial stem population); it self-renews and commits its first cell to
tumor, which then grows by real volumetric divisions. Necrotic death is switched
off, so the clone grows and then HOLDS instead of declining: a confined confluent
monolayer is volumetrically jammed (vertex-model rigidity, with a solid<->fluid
transition near preferred shape index s0 ~= 5.4; Azote & Manning 2025), so only the
divisions that find room complete and the 3D mass saturates at a stable, mechanics-
limited size — unlike the non-spatial ODE, which grows the tumor population without
bound. A small T1-reconnection bump nudges the basal layer toward the fluid regime
so a few more divisions succeed (see monolayer_config).

This script **only runs the simulation and archives its data**. Two consumers read
the archive without re-simulating:

  * ``tumor_coupling_3d_analysis.ipynb`` — the matplotlib analyses (population over
    time, a 3D apical-surface still, and the internal-SBML-model-vs-tissue comparison),
  * ``tumor_coupling_3d_viz.ipynb`` — the interactive ipyvolume 3D tissue viewer.

Run from the repo's ``vivarium-tyssue`` conda env:

    conda activate vivarium-tyssue
    cd Experiments/tumor_coupling
    python tumor_coupling_3d.py

Run length is env-overridable for quick smoke checks:
    TUMOR3D_TF=2 TUMOR3D_DT=0.1 python tumor_coupling_3d.py

Outputs (all under ``outputs/``, git-ignored):
  * ``monolayer_tumor_history.hf5`` — archived 3D apical tissue History (both
    notebooks load this)
  * ``sbml_population_3d.csv`` — the internal SBML model trajectory (species + fluxes),
    captured live during the run (the analysis notebook compares it to the tissue)
"""
from __future__ import annotations

import contextlib
import copy
import io
import os
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
# The 3D monolayer the user pointed at: Notebooks/monolayer_box.hf5 (extruded from
# the flat test_square sheet in monolayer_liftoff.ipynb).
MONO_SRC = REPO / "Notebooks" / "monolayer_box.hf5"
SBML_MODEL = "BIOMD0000000903.xml"          # COPASI breast-cancer population ODE
HISTORY_FILE = OUT_DIR / "monolayer_tumor_history.hf5"

COORDS_3D = ["x", "y", "z"]
FIG_DPI = 300
SEED = 20260722

# Run length (env-overridable for smoke tests). The 3D volume gradient is python
# (rust does geometry only), so steps are heavier than the 2D sheet — keep tf/dt
# modest. Divisions add cells, so the clone reaches a clear 3D mass well before tf.
TUMOR_TF = float(os.environ.get("TUMOR3D_TF", "160.0"))
TUMOR_DT = float(os.environ.get("TUMOR3D_DT", "0.1"))
# Mechanics step, independent of the coupling cadence above. A division transient
# briefly produces a large local gradient, and an explicit-Euler displacement is
# linear in dt: at 0.1 that one step throws a vertex far enough to blow the mesh to
# NaN (measured: first non-finite vertex at t=6.0; 0.05 fails at t=5.3; 0.02 runs
# clean to tf=60 with |pos|max 17.2, matching the old clamped run's 17.2). This is
# what replaces the former max_displacement clamp — the clamp masked the bad step by
# rescaling it, which silently makes the recorded trajectory not the model's.
# TumorCoupling keeps TUMOR_DT, so COPASI and the behaviours fire exactly as before.
TUMOR_SOLVER_DT = float(os.environ.get("TUMOR3D_SOLVER_DT", "0.02"))
# The tyssue History snapshots the whole monolayer every solver step and holds it in
# RAM; the 5x finer step would cost 5x the memory for no extra information (the full
# tf=160 run already OOM-killed a 19 GB machine when recording every step). Record
# every N-th step instead, keeping the recorded cadence exactly what it was at 0.1.
TUMOR_RECORD_EVERY = int(round(TUMOR_DT / TUMOR_SOLVER_DT))
COPASI_TIME = 1.0
SEED_STEM = 6           # initial cancer-stem-cell focus (matches the spec's seed below)

# --- mechanics (from monolayer_liftoff.ipynb — a stable monolayer energy model) --
VOL_ELASTICITY = 1.0    # K_V  : cell volume stiffness
PREFERRED_VOL = 1.0     # V_0  : target cell volume
AREA_ELASTICITY = 1.0   # K_A  : face area stiffness (all faces)
PREFERRED_AREA = 1.0    # A_0  : target face area
CONTRACTILITY = 0.05    # K_P  : face (perimeter) contractility
VISCOSITY = 1.0         # vertex drag (sets the relaxation timescale)
LINE_TENSION = 0.0

# Face palette by cell_type (shared with the 2D experiment / notebook).
CELL_TYPE_COLORS = {
    "healthy": "#4a90d9",    # blue — normal epithelium
    "tumor": "#c0392b",      # red — tumor cells
    "stem": "#8e44ad",       # purple — cancer stem cells
    "dead": "#2b2b2b",       # near-black — necrotic (SBML-driven death)
    "dividing": "#feeda3",   # yellow — mid-division (transient)
    "extruding": "#000000",  # black — mid-death (transient)
}
POP_TYPES = ["healthy", "tumor", "stem", "dead"]

# Exact COPASI reaction (flux) keys in BIOMD0000000903, per target cell type.
BIRTH_FLUXES = {
    "tumor": "Induction of tumor cell",
    "healthy": "Increase in the healthy cell in the system",
    "stem": "Formation of Stem cell",
}
DEATH_FLUXES = {
    "tumor": "Removal of tumor cell y immune cell and other immune response",
    "healthy": "Decrase of healthy cell due to cancer",
    "stem": "Removal of stem cell from the system",
}
# Convert SBML fluxes into mesh events (an event fires when flux*scale*dt passes 1).
# Same balance as the 2D run; tumor births lead so the seeded clone expands in 3D.
# tumor_births 3e-6 (vs 1e-6 in 2D): ~0.9 birth events/unit-time; each is a real
# volumetric division that adds a cell.
#
# tumor_deaths and stem_deaths are OFF (0.0): apoptosis_3d marks a cell necrotic
# but leaves it in the mesh, so a nonzero death flux made the clone *decline* into
# accumulating dead cells over a long run. With death off the clone grows and then
# HOLDS at the size the monolayer mechanics allow — see the rigidity note on
# threshold_length below: a confined 3D monolayer is volumetrically jammed, so only
# the divisions that find room complete and the mass saturates at a stable ceiling
# rather than growing without bound (as the non-spatial ODE does).
# stem_births keeps the seeded cancer-stem-cell focus self-renewing into a
# persistent core.
SCALES = {
    "tumor_births": 3.0e-6, "tumor_deaths": 0.0,
    "healthy_births": 3.0e-8, "healthy_deaths": 6.0e-8,
    "stem_births": 3.0e-6, "stem_deaths": 0.0,
}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def ensure_dataset(name: str, src: Path) -> Path:
    """Copy a source dataset into the git-ignored local data/ (once)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dst = DATA_DIR / name
    if not dst.exists():
        if not src.exists():
            raise FileNotFoundError(f"source dataset not found: {src}")
        shutil.copy(src, dst)
        print(f"copied {src} -> {dst}")
    return dst


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
def monolayer_config(dataset_path: Path) -> dict:
    """EulerSolver config for the 3D monolayer. Every cell starts
    cell_type='healthy'; the energy model (cell volume + face area + contractility
    + line tension) is the stable one from monolayer_liftoff.ipynb."""
    return {
        "name": "Tumor Monolayer 3D",
        "eptm": str(dataset_path),
        "tissue_type": "Monolayer",
        "parameters": {
            "cell_df": {
                "vol_elasticity": VOL_ELASTICITY,
                "prefered_vol": PREFERRED_VOL,
                "prefered_area": PREFERRED_AREA,
                "cell_type": "healthy",
                "is_alive": 1,
            },
            "face_df": {
                "area_elasticity": AREA_ELASTICITY,
                "prefered_area": PREFERRED_AREA,
                "contractility": CONTRACTILITY,
                "is_alive": 1,
            },
            "edge_df": {"line_tension": LINE_TENSION, "is_active": 1},
            "vert_df": {"viscosity": VISCOSITY, "is_alive": 1},
        },
        "geom": "MonolayerGeometry",
        "effectors": ["CellVolumeElasticity", "FaceAreaElasticity",
                      "FaceContractility", "LineTension"],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory",
        # T1 neighbour swaps fluidize the confluent basal layer so cells can
        # rearrange around a growing/dividing cell. In vertex-model rigidity theory
        # the 3D tissue is solid below a preferred shape index s0 ~= 5.4 and fluid
        # above it (Azote & Manning, 3D vertex models of stratified epithelia,
        # 2025; cancer tissue is the fluid side, Grosser++ PRX 2020); a higher T1
        # threshold pushes the layer toward that fluid regime so more divisions find
        # room to complete. 0.05 is a deliberate, safe bump from the old 0.03: tyssue's
        # monolayer reconnect corrupts the mesh above ~0.1, so this is as fluid as the
        # topology stays numerically stable. Even so the layer is only marginally
        # unjammed, which is why the 3D clone saturates at a modest stable size.
        "settings": {"threshold_length": 0.05},
        "auto_reconnect": True,
        "bounds": None,
        "output_columns": {},
        "history_columns": {},
        "maps": {},
        # MonolayerGeometry bulk kernel is rust-eligible; the volume-elastic
        # gradient falls back to python (rust geometry + python gradient).
        "backend": "rust",
        "substeps": 1,
        # Off (0 = disabled). The clamp is a numerical safety net, not physics: it
        # rescales the one step after a division transient, so the trajectory it
        # produces is not the model's. Kept off in every experiment — if a division
        # transient diverges, the fix belongs in the step size or the behaviour, not
        # in a bound on how far a vertex may move.
        "max_displacement": 0.0,
        "record_history": True,    # in-memory History -> saved via to_archive()
    }


def build_tumor_spec(mesh_path: Path, model_path: Path) -> dict:
    """EulerSolver (3D monolayer) + a TumorCoupling process wired over Behaviors."""
    spec = {
        "Tyssue": {
            "_type": "process",
            "address": "local:EulerSolver",
            "config": monolayer_config(mesh_path),
            "inputs": {"behaviors": ["Behaviors"], "global_time": ["global_time"]},
            "outputs": {
                "datasets": ["Tissue State"],
                "network_changed": ["Network Changed"],
                "behaviors_update": ["Behaviors"],
            },
            "interval": TUMOR_SOLVER_DT,
        },
        "Network Changed": False,
        "Behaviors": {},
    }
    spec["TumorCoupling"] = {
        "_type": "process",
        "address": "local:TumorCoupling",
        "config": {
            "model_source": str(model_path),
            "copasi_time": COPASI_TIME,
            "copasi_intervals": 10,
            "birth_fluxes": dict(BIRTH_FLUXES),
            "death_fluxes": dict(DEATH_FLUXES),
            "scales": dict(SCALES),
            "geom": "MonolayerGeometry",
            "dt": 1.0,
            # growth_rate 1.0: a committed cell grows past crit_vol (2.0) in ~2
            # tyssue-time units (measured), fast enough that divisions keep up with
            # the birth rate but slow enough that the mesh relaxes around each split.
            "growth_rate": 1.0,
            "shrink_rate": 1.0,
            # Critical VOLUME, set like the 2D critical area: a cell grows its
            # prefered_vol from 1.0 and divides once its actual vol passes 2.0.
            "division_crit": 2.0,
            "apoptosis_crit": 0.5,
            # Seed a compact central cancer-stem-cell focus (like the SBML model's
            # nonzero initial stem population); it commits its first cell to tumor and
            # self-renews, so the 3D clone grows while a stem core persists.
            "seed": {"tumor": 0, "stem": SEED_STEM},
            # Real 3D vertex-model topology (monolayer cell_division).
            "topology_ops": True,
            # Randomize the division plane so the tumor grows in 3D, not one plane.
            "orientations": ["vertical", "horizontal"],
        },
        "inputs": {"datasets": ["Tissue State"], "global_time": ["global_time"]},
        "outputs": {"behaviors": ["Behaviors"]},
        "interval": TUMOR_DT,
    }
    return spec


def run(core, spec: dict, tf: float):
    """Run a spec to elapsed global time ``tf``; return ``(history, sbml_records)``.
    ``sbml_records`` is the internal COPASI trajectory captured during the run.
    Division prints a line per new cell; those are silenced here."""
    from process_bigraph import Composite

    spec = copy.deepcopy(spec)
    sim = Composite({"state": spec}, core=core)
    solver = sim.state["Tyssue"]["instance"]
    # EulerSolver builds its History with the defaults, so set the thinning on the
    # instance before the run (it never touches either attribute afterwards).
    if TUMOR_RECORD_EVERY > 1 and solver.history is not None:
        solver.history.dt = spec["Tyssue"]["interval"]
        solver.history.save_every = TUMOR_RECORD_EVERY * solver.history.dt
    sbml_records = _attach_sbml_recorder(sim)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run(tf)
    history = sim.state["Tyssue"]["instance"].history
    history.update_datasets()
    return history, sbml_records


# ---------------------------------------------------------------------------
# Persist history for the notebook
# ---------------------------------------------------------------------------
# The archived history is the monolayer's **apical surface** — a single-layer
# sheet (one polygon per cell) so tyssue's `browse_history` draws a clean tissue
# where the coloured faces and the wireframe edges coincide, instead of the full
# solid monolayer (apical + basal + lateral walls) whose overlapping faces read as
# a jagged blob. Per element we keep the minimal columns the ipyvolume renderer
# needs: vertex + face-centroid positions for the polygons, edge srce/trgt/face for
# the wireframe, edge sub-coordinates sx..tz for the face triangles, and the
# per-face `cell_type` (mapped down from the cell) the notebook colours by.
_ARCHIVE_COLS = {
    "vert": ["x", "y", "z"],
    "edge": ["srce", "trgt", "face", "sx", "sy", "sz", "tx", "ty", "tz"],
    "face": ["x", "y", "z", "cell_type"],
    "cell": ["cell_type", "unique_id", "vol", "is_alive", "x", "y", "z"],
}
_ARCHIVE_INT_COLS = {"srce", "trgt", "face", "unique_id"}
_ARCHIVE_STR_COLS = {"cell_type"}
HISTORY_FRAMES = 60


def apical_sheet(frame):
    """The apical surface of a monolayer frame as a standalone sub-epithelium
    (one face per cell), with each apical face stamped with its cell's ``cell_type``
    for colouring. ``frame`` must be an in-memory History frame (contiguous
    indices, so tyssue's positional ``srce``/``face``/``cell`` line up); geometry is
    refreshed so the face centroids / edge sub-coordinates match the vertices."""
    from tyssue.geometry.bulk_geometry import MonolayerGeometry
    from tyssue.utils.utils import get_sub_eptm

    MonolayerGeometry.update_all(frame)
    seg = frame.face_df["segment"].astype(str).str.strip().to_numpy()
    apical_face_pos = np.where(seg == "apical")[0]
    apical_edges = frame.edge_df.index[frame.edge_df["face"].isin(apical_face_pos)]
    sub = get_sub_eptm(frame, apical_edges, copy=True)
    if sub is None:
        return None
    sub.reset_index()
    ctype = sub.cell_df["cell_type"].astype(str).to_numpy()          # positional by cell
    face_cell = sub.edge_df.groupby("face")["cell"].first()          # face pos -> cell pos
    ftype = np.array(["healthy"] * sub.Nf, dtype=object)
    for fpos, cpos in face_cell.items():
        cpos = int(cpos)
        if 0 <= cpos < len(ctype):
            ftype[int(fpos)] = ctype[cpos]
    sub.face_df["cell_type"] = ftype
    return sub


def save_history(history, out_path: Path):
    """Archive the apical tissue surface to HDF5 so tyssue's
    ``HistoryHdf5.from_archive`` can reopen it and ``browse_history`` can redraw the
    coloured surface frame by frame.

    For ``HISTORY_FRAMES`` evenly-spaced timesteps, extract the apical sheet
    (:func:`apical_sheet`) and append one long-format table per element (a ``time``
    column per row) restricted to the render columns. Topology columns are coerced
    to int (divisions can leave them object-dtype, which an HDF *table* rejects) and
    ``cell_type`` kept as a fixed-width string."""
    if out_path.exists():
        out_path.unlink()
    all_times = np.asarray(history.time_stamps)
    keep_idx = np.unique(np.round(np.linspace(0, len(all_times) - 1,
                                              min(HISTORY_FRAMES, len(all_times)))).astype(int))
    keep_times = all_times[keep_idx]
    with pd.HDFStore(str(out_path), "a") as store:
        for t in keep_times:
            sub = apical_sheet(history.retrieve(float(t)))
            if sub is None:
                continue
            for el, cols in _ARCHIVE_COLS.items():
                df = getattr(sub, f"{el}_df")
                keep = [c for c in cols if c in df.columns]
                d = df[keep].reset_index(drop=True)
                d.insert(0, el, np.arange(len(d)))   # positional index as a column
                d["time"] = float(t)
                for c in d.columns:
                    if c in _ARCHIVE_INT_COLS:
                        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(-1).astype("int64")
                    elif c in _ARCHIVE_STR_COLS:
                        d[c] = d[c].astype(str)
                store.append(key=el, value=d, data_columns=["time"],
                             min_itemsize={c: 12 for c in _ARCHIVE_STR_COLS if c in d.columns} or None)
    print(f"  wrote {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB, "
          f"{len(keep_times)} frames)")


# ---------------------------------------------------------------------------
# Internal SBML model — record its trajectory (the analysis notebook compares it
# to the tissue)
# ---------------------------------------------------------------------------
# The tumor ODE (BIOMD0000000903) runs autonomously inside TumorCoupling (the
# process feeds nothing back into it), so wrapping the COPASI update captures the
# exact trajectory that drove the tissue. Species: T tumor, H healthy, C stem,
# I immune, E estrogen.
SBML_SPECIES = {"T": "tumor", "H": "healthy", "C": "stem", "I": "immune", "E": "estrogen"}


def _attach_sbml_recorder(sim) -> list:
    """Wrap the TumorCoupling's internal COPASI update so every SBML step it takes
    is recorded (species amounts + fluxes). Returns the list that fills during the
    run; empty if the process/COPASI isn't present. Coupling behaviour is unchanged."""
    records: list = []
    try:
        tc = sim.state["TumorCoupling"]["instance"]
    except (KeyError, TypeError):
        return records
    copasi = getattr(tc, "_copasi", None)
    if copasi is None:
        return records
    original = copasi.update

    def recording_update(inputs, interval):
        out = original(inputs, interval)
        records.append({
            "interval": float(interval),
            "species": dict(out.get("species_concentrations", {})),
            "fluxes": dict(out.get("fluxes", {})),
        })
        return out

    copasi.update = recording_update
    return records


def sbml_trajectory_df(records: list) -> pd.DataFrame:
    """Internal SBML model over (model) time: one row per recorded step with each
    species amount and each birth/death flux. ``time`` is cumulative model-time."""
    rows = []
    t = 0.0
    for rec in records:
        t += rec["interval"]
        row = {"time": t}
        for sid, val in rec["species"].items():
            row[f"species_{sid}"] = float(val)
        for cell_type, key in BIRTH_FLUXES.items():
            row[f"flux_{cell_type}_birth"] = float(rec["fluxes"].get(key, 0.0))
        for cell_type, key in DEATH_FLUXES.items():
            row[f"flux_{cell_type}_death"] = float(rec["fluxes"].get(key, 0.0))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _composition(history, t) -> dict:
    """{cell_type: count} of cells at recorded time ``t`` (for the log)."""
    cell = history.datasets["cell"]
    return cell[cell["time"] == t]["cell_type"].value_counts().to_dict()


def main():
    np.random.seed(SEED)
    sys.path.insert(0, str(REPO))
    from vivarium_tyssue.core import build_core

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = ensure_dataset("monolayer_box.hf5", MONO_SRC)
    model = ensure_dataset(SBML_MODEL, REPO / "workspace" / "datasets" / SBML_MODEL)
    core = build_core()

    print(f"[tumor3d] monolayer + COPASI BIOMD0000000903, tf={TUMOR_TF} dt={TUMOR_DT} "
          f"(~{int(TUMOR_TF / TUMOR_DT)} coupled updates) ...", flush=True)
    history, sbml_records = run(core, build_tumor_spec(mesh, model), TUMOR_TF)

    save_history(history, HISTORY_FILE)

    # The internal SBML model trajectory is captured live during the run (it wraps
    # the process's own COPASI update), so it is written here; the analysis notebook
    # reads it back and compares it to the archived tissue counts.
    if sbml_records:
        sbml_df = sbml_trajectory_df(sbml_records)
        sbml_df.to_csv(OUT_DIR / "sbml_population_3d.csv", index=False)
        print(f"[tumor3d] wrote sbml_population_3d.csv ({len(sbml_df)} steps)", flush=True)
    else:
        print("[tumor3d] no SBML records captured (coupling ran without COPASI)", flush=True)

    times = list(history.time_stamps)
    if times:
        print(f"[tumor3d] composition start {_composition(history, times[0])} "
              f"-> end {_composition(history, times[-1])}", flush=True)
    print("\ndone — analyse with tumor_coupling_3d_analysis.ipynb "
          "(or view in 3D with tumor_coupling_3d_viz.ipynb)")


if __name__ == "__main__":
    main()
