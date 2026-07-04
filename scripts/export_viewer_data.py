#!/usr/bin/env python
"""Export tyssue composite runs as JSON time-series for the three.js viewer.

Runs each showcase composite for a number of emit frames, snapshotting the
epithelium mesh (vertices + half-edge faces + per-face fields) at every frame,
and writes ``viewer/data/<slug>.json`` plus a ``viewer/data/index.json`` manifest
— the same static-data contract the pbg-cpm / parsimony viewers use.

    .venv/bin/python scripts/export_viewer_data.py            # default showcase set
    .venv/bin/python scripts/export_viewer_data.py --frames 80

Each model file is ``{name, is3d, n_cells, n_verts, bounds, face_fields, frames}``
where every frame is a self-contained mesh (topology is stored per frame so cell
division / T1 transitions are handled):

    frame = {t, verts:[x,y,z,...], centroids:[x,y,z,...],
             tris:[i,j,k,...], edges:[a,b,...], face_of_tri:[f,...],
             fields:{area:[...], perimeter:[...], num_sides:[...]}}

`tris` index into the concatenated [verts ; centroids] position block, so each
half-edge (srce, trgt, face-centroid) is one triangle — a clean fan-filled
polygon per cell. `edges` are vertex-index pairs for the wireframe.
"""
import argparse
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "viewer" / "data"

# slug -> (composite, human name, blurb, dt, frame_span, n frames).
# `dt` is the solver step (interval override — small enough that explicit Euler
# stays stable for stiff 3D meshes); `frame_span` is how much sim-time each emitted
# frame advances (frame_span/dt solver steps per frame). Models that error or blow
# up to NaN are truncated/omitted (see run_model).
SHOWCASE = [
    ("anisotropic", "Anisotropic elongation",
     "Flat sheet biased by axis-dependent line tension — oriented (anisotropic) "
     "tissue elongation. Colour by cell area to watch cells stretch.", 0.1, 0.4, 60),
    ("stochastic", "Stochastic line tension",
     "Flat sheet with Ornstein–Uhlenbeck line-tension noise on every edge — "
     "jittery, fluctuating cell shapes.", 0.1, 0.4, 60),
    ("gradient", "Preferred-perimeter gradient",
     "Flat sheet with a linear spatial gradient of preferred perimeter along x, "
     "so cell size varies smoothly across the sheet.", 0.1, 0.4, 60),
    ("jamming", "Stochastic + jamming",
     "Flat sheet with stochastic line tension plus a CellJamming event that ramps "
     "preferred perimeter toward a rigid, jammed regime.", 0.1, 0.4, 60),
    ("epithelium_2d", "Baseline 2D relaxation",
     "Plain flat (square) sheet relaxing to mechanical equilibrium — the baseline "
     "2D epithelium, no behaviours.", 0.05, 0.2, 50),
    ("base_solver", "Vessel relaxation (3D)",
     "Cylindrical (vessel) sheet relaxing to mechanical equilibrium — a true 3D "
     "surface. Orbit to inspect the tube.", 0.02, 0.1, 50),
    ("monolayer_liftoff", "Monolayer lift-off (3D)",
     "A 3D monolayer with apical/basal surfaces — volumetric cells lifting off a "
     "substrate. Orbit to see the layered geometry.", 0.01, 0.05, 40),
    ("gillespie", "Intestinal crypt (3D)",
     "Crypt model on a cylinder: mechanics plus a Gillespie process firing "
     "stochastic cell-type transitions under Wnt/density regulation.", 0.005, 0.02, 40),
    ("tumor", "Tumor growth (COPASI-coupled)",
     "Flat sheet coupled to a breast-cancer population ODE in COPASI; cells divide "
     "as the tumour grows (needs the COPASI process).", 0.01, 0.05, 40),
]


def build(composite, backend="rust", interval=None):
    from pbg_superpowers.composite_spec import build_composite_from_spec, load_spec
    from vivarium_tyssue.core import build_core

    spec = load_spec(ROOT / "vivarium_tyssue" / "composites" / f"{composite}.composite.yaml")
    spec["emitters"] = []  # we snapshot directly; no parquet sink needed
    spec["state"]["Tyssue"]["config"]["backend"] = backend
    overrides = {"interval": interval} if interval is not None else {}
    with contextlib.redirect_stdout(io.StringIO()):
        comp = build_composite_from_spec(spec, overrides=overrides, core=build_core())
    return comp


def snapshot(eptm, t, r=4):
    """Extract a render-ready mesh from the epithelium at time t."""
    coords = eptm.coords
    V = np.asarray(eptm.vert_df[coords].values, dtype=float)
    if V.shape[1] == 2:  # promote flat 2D sheets to z=0
        V = np.column_stack([V, np.zeros(len(V))])
    C = np.asarray(eptm.face_df[coords].values, dtype=float)
    if C.shape[1] == 2:
        C = np.column_stack([C, np.zeros(len(C))])
    Nv = len(V)

    vmap = {v: i for i, v in enumerate(eptm.vert_df.index)}
    fmap = {f: i for i, f in enumerate(eptm.face_df.index)}
    s = eptm.edge_df["srce"].map(vmap).values.astype(int)
    tt = eptm.edge_df["trgt"].map(vmap).values.astype(int)
    f = eptm.edge_df["face"].map(fmap).values.astype(int)

    tris = np.column_stack([s, tt, Nv + f]).ravel().tolist()
    edges = np.column_stack([s, tt]).ravel().tolist()

    fields = {}
    for col in ("area", "perimeter"):
        if col in eptm.face_df.columns:
            fields[col] = np.round(eptm.face_df[col].values.astype(float), 6).tolist()
    fields["num_sides"] = eptm.edge_df.groupby("face").size().reindex(eptm.face_df.index).fillna(0).astype(int).tolist()

    # categorical cell type (or monolayer face segment), if the mesh carries one —
    # kept as raw strings here; run_model maps them to stable integer ids.
    tcol = next((c for c in ("cell_type", "segment") if c in eptm.face_df.columns), None)
    cell_type_raw = eptm.face_df[tcol].astype(str).tolist() if tcol else None

    return {
        "t": round(float(t), 4),
        "verts": np.round(V, r).ravel().tolist(),
        "centroids": np.round(C, r).ravel().tolist(),
        "fields": fields,
        "cell_type_raw": cell_type_raw,
        # topology — hoisted to model level by run_model when constant
        "tris": tris,
        "edges": edges,
        "face_of_tri": f.tolist(),
    }


def _finite(fr):
    return np.isfinite(np.asarray(fr["verts"])).all() and np.isfinite(np.asarray(fr["centroids"])).all()


def run_model(slug, composite, name, blurb, dt, frame_span, nframes):
    comp = build(composite, interval=dt)  # solver dt (small enough to stay stable)
    proc = comp.state["Tyssue"]["instance"]
    eptm = proc.eptm
    frames = [snapshot(eptm, 0.0)]
    t = 0.0
    truncated = False
    for _ in range(nframes - 1):
        t += frame_span
        with contextlib.redirect_stdout(io.StringIO()):
            comp.run(frame_span)
        fr = snapshot(proc.eptm, t)
        if not _finite(fr):  # explicit-Euler blow-up — keep the stable frames only
            truncated = True
            break
        frames.append(fr)
    if truncated:
        print(f"  {slug:14s} (blew up after {len(frames)} frames — truncated to the stable portion)")
    if len(frames) < 2:
        raise RuntimeError("model unstable from the first step; nothing stable to show")

    verts = np.asarray(frames[0]["verts"]).reshape(-1, 3)
    zspan = float(verts[:, 2].max() - verts[:, 2].min())
    xyspan = float(max(np.ptp(verts[:, 0]), np.ptp(verts[:, 1])))
    is3d = zspan > 0.05 * (xyspan or 1.0)  # a flat sheet has ~no z extent
    allv = np.concatenate([np.asarray(f["verts"]).reshape(-1, 3) for f in frames])
    bounds = [allv.min(0).round(4).tolist(), allv.max(0).round(4).tolist()]

    # Categorical cell type: gather the categories across all frames into a stable
    # id map (a cell's type can change over time, e.g. gillespie differentiation),
    # then store an integer cell_type field per frame + the index->name list.
    type_names = None
    if frames[0].get("cell_type_raw") is not None:
        cats = sorted({v for f in frames for v in f["cell_type_raw"]})
        tmap = {c: i for i, c in enumerate(cats)}
        type_names = cats
        for f in frames:
            f["fields"]["cell_type"] = [tmap[v] for v in f["cell_type_raw"]]
    for f in frames:
        f.pop("cell_type_raw", None)

    n_cells = max(len(f["fields"]["num_sides"]) for f in frames)
    model = {
        "name": name, "blurb": blurb, "is3d": is3d,
        "n_cells": int(n_cells), "n_verts": int(len(verts)),
        "bounds": bounds,
        "face_fields": [k for k in ("area", "perimeter", "num_sides")],
        "type_names": type_names,
        "frames": frames,
    }
    # Hoist topology to model level when it never changes (no division/T1) — the
    # tris/edges/face_of_tri arrays dominate file size, so storing them once
    # instead of per frame roughly halves it. The viewer falls back to per-frame
    # topology when a frame carries its own (dividing meshes).
    keys = ("tris", "edges", "face_of_tri")
    constant = all(frames[i][k] == frames[0][k] for k in keys for i in range(1, len(frames)))
    if constant:
        for k in keys:
            model[k] = frames[0][k]
        for fr in frames:
            for k in keys:
                del fr[k]
        model["static_topology"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{slug}.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    kb = path.stat().st_size / 1024
    print(f"  {slug:14s} {'3D' if is3d else '2D'}  {n_cells:>4d} cells  {len(frames):>3d} frames  {kb:7.0f} KB")
    return {"file": f"{slug}.json", "slug": slug, "name": name,
            "is3d": is3d, "n_cells": int(n_cells), "n_frames": len(frames)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0, help="override frame count for all models")
    ap.add_argument("--only", default="", help="comma-separated slugs to export (default: all)")
    args = ap.parse_args()

    only = set(s.strip() for s in args.only.split(",") if s.strip())
    manifest = []
    print("exporting tyssue viewer data ->", OUT.relative_to(ROOT))
    for slug, name, blurb, dt, frame_span, nframes in SHOWCASE:
        if only and slug not in only:
            continue
        if args.frames:
            nframes = args.frames
        try:
            manifest.append(run_model(slug, slug, name, blurb, dt, frame_span, nframes))
        except Exception as e:
            print(f"  {slug:14s} ERROR {type(e).__name__}: {str(e)[:70]}")

    if manifest:
        # preserve any models we didn't regenerate this run
        idx_path = OUT / "index.json"
        existing = {}
        if idx_path.exists():
            for m in json.loads(idx_path.read_text()).get("models", []):
                existing[m["slug"]] = m
        for m in manifest:
            existing[m["slug"]] = m
        order = [s[0] for s in SHOWCASE]
        models = sorted(existing.values(), key=lambda m: order.index(m["slug"]) if m["slug"] in order else 99)
        idx_path.write_text(json.dumps({"models": models}, indent=2), encoding="utf-8")
        print(f"wrote {len(manifest)} model(s); manifest has {len(models)} total -> {idx_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
