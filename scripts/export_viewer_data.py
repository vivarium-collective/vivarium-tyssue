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

# slug -> (composite, human name, blurb, emit interval, n frames)
SHOWCASE = [
    ("anisotropic", "Anisotropic elongation",
     "Flat sheet biased by axis-dependent line tension — oriented (anisotropic) "
     "tissue elongation. Colour by cell area to watch cells stretch.", 0.4, 60),
    ("stochastic", "Stochastic line tension",
     "Flat sheet with Ornstein–Uhlenbeck line-tension noise on every edge — "
     "jittery, fluctuating cell shapes.", 0.4, 60),
    ("gradient", "Preferred-perimeter gradient",
     "Flat sheet with a linear spatial gradient of preferred perimeter along x, "
     "so cell size varies smoothly across the sheet.", 0.4, 60),
    ("base_solver", "Vessel relaxation (3D)",
     "Cylindrical (vessel) sheet relaxing to mechanical equilibrium — a true 3D "
     "surface. Orbit to inspect the tube.", 0.5, 50),
    # NOTE: "regulation" (cylinder + cell division) is omitted — it hits a
    # pre-existing empty-cell-pick bug in behaviors/regulations.py, unrelated to
    # the viewer. Re-add it here once that's fixed to showcase changing topology.
]


def build(composite, backend="rust"):
    from pbg_superpowers.composite_spec import build_composite_from_spec, load_spec
    from vivarium_tyssue.core import build_core

    spec = load_spec(ROOT / "vivarium_tyssue" / "composites" / f"{composite}.composite.yaml")
    spec["emitters"] = []  # we snapshot directly; no parquet sink needed
    spec["state"]["Tyssue"]["config"]["backend"] = backend
    with contextlib.redirect_stdout(io.StringIO()):
        comp = build_composite_from_spec(spec, core=build_core())
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

    return {
        "t": round(float(t), 4),
        "verts": np.round(V, r).ravel().tolist(),
        "centroids": np.round(C, r).ravel().tolist(),
        "fields": fields,
        # topology — hoisted to model level by run_model when constant
        "tris": tris,
        "edges": edges,
        "face_of_tri": f.tolist(),
    }


def run_model(slug, composite, name, blurb, interval, nframes):
    comp = build(composite)
    proc = comp.state["Tyssue"]["instance"]
    eptm = proc.eptm
    frames = [snapshot(eptm, 0.0)]
    t = 0.0
    for _ in range(nframes - 1):
        t += interval
        with contextlib.redirect_stdout(io.StringIO()):
            comp.run(interval)
        frames.append(snapshot(proc.eptm, t))

    verts = np.asarray(frames[0]["verts"]).reshape(-1, 3)
    zspan = float(verts[:, 2].max() - verts[:, 2].min())
    xyspan = float(max(np.ptp(verts[:, 0]), np.ptp(verts[:, 1])))
    is3d = zspan > 0.05 * (xyspan or 1.0)  # a flat sheet has ~no z extent
    allv = np.concatenate([np.asarray(f["verts"]).reshape(-1, 3) for f in frames])
    bounds = [allv.min(0).round(4).tolist(), allv.max(0).round(4).tolist()]

    n_cells = max(len(f["fields"]["num_sides"]) for f in frames)
    model = {
        "name": name, "blurb": blurb, "is3d": is3d,
        "n_cells": int(n_cells), "n_verts": int(len(verts)),
        "bounds": bounds,
        "face_fields": [k for k in ("area", "perimeter", "num_sides")],
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
    for slug, name, blurb, interval, nframes in SHOWCASE:
        if only and slug not in only:
            continue
        if args.frames:
            nframes = args.frames
        try:
            manifest.append(run_model(slug, slug, name, blurb, interval, nframes))
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
