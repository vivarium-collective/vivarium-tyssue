#!/usr/bin/env python
"""Generate the big "how many cells?" showcase demos straight to viewer JSON.

These two demos exist to show *scale* — tens of thousands of individually
simulated vertex-model cells — with genuinely dynamic behaviour, not a static
render. They drive the tyssue meshes with the Rust EulerSolver kernels directly
(no composite / behaviour machinery) so we get full control over the dynamics
and the frame cadence, then write ``viewer/data/<slug>.json`` in the same
contract as ``export_viewer_data.py``.

    .venv/bin/python scripts/gen_scale_showcase.py

- ``scale_growth_wave``  — a ~40k-cell flat epithelium with a travelling wave of
  cell growth sweeping across it: a band of cells swells then relaxes as the
  front passes (a morphogen / peristaltic wave through 40,000 cells).
- ``scale_organoid``     — a ~20k-cell hollow "organoid": a sphere tiled by a
  spherical-Voronoï epithelium where scattered growth centres bulge outward into
  buds, so the ball goes from smooth to lumpy. Fully orbitable.

Both keep a fixed topology (no division / T1), so the tris/edges are hoisted to
the model once and only vertex positions + per-cell area change per frame.
"""
import contextlib
import hashlib
import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "viewer" / "data"

from vivarium_tyssue.processes.utils import (  # noqa: E402
    rust_sheet_gradient, rust_update_geometry,
)


def snapshot(eptm, t, r=2):
    """Lean render snapshot for the big scale demos: vertices + fan topology +
    per-cell area only. Deliberately omits `centroids` (the viewer derives them
    from the vertices) and `perimeter`, and rounds coordinates to `r` decimals —
    together these ~4x shrink the file so tens of thousands of cells stay
    loadable. Topology (tris/edges/face_of_tri) is hoisted to the model later."""
    coords = eptm.coords
    V = np.asarray(eptm.vert_df[coords].values, float)
    if V.shape[1] == 2:
        V = np.column_stack([V, np.zeros(len(V))])
    Nv = len(V)
    vmap = {v: i for i, v in enumerate(eptm.vert_df.index)}
    fmap = {v: i for i, v in enumerate(eptm.face_df.index)}
    s = eptm.edge_df["srce"].map(vmap).values.astype(int)
    tt = eptm.edge_df["trgt"].map(vmap).values.astype(int)
    f = eptm.edge_df["face"].map(fmap).values.astype(int)
    num_sides = (eptm.edge_df.groupby("face").size()
                 .reindex(eptm.face_df.index).fillna(0).astype(int).tolist())
    return {
        "t": round(float(t), 3),
        "verts": np.round(V, r).ravel().tolist(),
        "fields": {"area": np.round(eptm.face_df["area"].values.astype(float), r).tolist(),
                   "num_sides": num_sides},
        "tris": np.column_stack([s, tt, Nv + f]).ravel().tolist(),
        "edges": np.column_stack([s, tt]).ravel().tolist(),
        "face_of_tri": f.tolist(),
    }


# --------------------------------------------------------------------------- #
# rust stepper (topology fixed → build the index arrays once)
# --------------------------------------------------------------------------- #
def make_stepper(sheet, dt=0.02, max_disp=0.15):
    from tyssue import SheetGeometry

    coords = sheet.coords
    vmap = {v: i for i, v in enumerate(sheet.vert_df.index)}
    fmap = {v: i for i, v in enumerate(sheet.face_df.index)}
    srce = np.ascontiguousarray(sheet.edge_df["srce"].map(vmap).values, np.uint32)
    trgt = np.ascontiguousarray(sheet.edge_df["trgt"].map(vmap).values, np.uint32)
    face = np.ascontiguousarray(sheet.edge_df["face"].map(fmap).values, np.uint32)
    topo = (srce, trgt, face)

    def step(n=1, inject=None):
        """Advance `n` explicit-Euler steps of real Rust vertex mechanics. If
        `inject(P)` is given it returns a per-vertex displacement added each step
        — a morphogenetic *growth* force (outward budding, out-of-plane folding).
        Pure area/tension relaxation only jiggles near equilibrium; the injected
        force is what drives the large-scale shape change, while the mechanics
        keep every cell well-formed as it deforms."""
        visc = sheet.vert_df["viscosity"].values[:, None]
        for _ in range(n):
            stash = rust_update_geometry(sheet, srce, trgt, face)
            g = rust_sheet_gradient(sheet, False, topo=topo, geom=stash)
            disp = np.clip(-dt * g / visc, -max_disp, max_disp)[:, : len(coords)]
            P = sheet.vert_df[coords].values
            if inject is not None:
                disp = disp + inject(P)
            sheet.vert_df[coords] = P + disp
        with contextlib.redirect_stdout(io.StringIO()):
            SheetGeometry.update_all(sheet)

    return step


def _vertex_field(sheet, face_field):
    """Scatter a per-face field onto vertices (mean over incident faces)."""
    vmap = {v: i for i, v in enumerate(sheet.vert_df.index)}
    fmap = {v: i for i, v in enumerate(sheet.face_df.index)}
    srce = sheet.edge_df["srce"].map(vmap).values
    face = sheet.edge_df["face"].map(fmap).values
    vf = np.zeros(sheet.Nv)
    cnt = np.zeros(sheet.Nv)
    np.add.at(vf, srce, face_field[face])
    np.add.at(cnt, srce, 1.0)
    return vf / np.maximum(cnt, 1)


def bake(sheet, ka=1.0, kp=0.1, lt=0.05):
    """Bake per-cell rest targets + moduli so the mesh starts stress-free."""
    from tyssue import SheetGeometry

    with contextlib.redirect_stdout(io.StringIO()):
        SheetGeometry.update_all(sheet)
    perim = sheet.edge_df.groupby("face")["length"].sum().reindex(sheet.face_df.index)
    sheet.face_df["area_elasticity"] = ka
    sheet.face_df["perimeter_elasticity"] = kp
    sheet.face_df["prefered_area"] = sheet.face_df["area"].values
    sheet.face_df["prefered_perimeter"] = perim.values
    sheet.face_df["is_alive"] = 1.0
    sheet.edge_df["line_tension"] = lt
    sheet.edge_df["is_active"] = 1.0
    sheet.vert_df["viscosity"] = 1.0
    sheet.vert_df["is_alive"] = 1.0
    sheet.face_df["rest_area"] = sheet.face_df["area"].values  # remember baseline


# --------------------------------------------------------------------------- #
# mesh builders
# --------------------------------------------------------------------------- #
def flat_sheet(n):
    from tyssue import Sheet, SheetGeometry

    s = Sheet.planar_sheet_2d(f"s{n}", nx=n, ny=n, distx=1, disty=1)
    s.sanitize(trim_borders=True)
    s.vert_df["z"] = 0.0
    s = Sheet(f"s{n}", {"vert": s.vert_df, "edge": s.edge_df, "face": s.face_df},
              coords=["x", "y", "z"])
    with contextlib.redirect_stdout(io.StringIO()):
        SheetGeometry.update_all(s)
    return s


def _fib_sphere(n, r=1.0):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)] * r


def sphere_sheet(ncells, R=10.0, name="organoid"):
    from scipy.spatial import SphericalVoronoi
    from tyssue import Sheet

    pts = _fib_sphere(ncells, R)
    sv = SphericalVoronoi(pts, radius=R, center=np.zeros(3))
    sv.sort_vertices_of_regions()
    V = sv.vertices
    srce, trgt, face = [], [], []
    for f, reg in enumerate(sv.regions):
        m = len(reg)
        for k in range(m):
            srce.append(reg[k]); trgt.append(reg[(k + 1) % m]); face.append(f)
    edge = pd.DataFrame({"srce": srce, "trgt": trgt, "face": face})
    vert = pd.DataFrame({"x": V[:, 0], "y": V[:, 1], "z": V[:, 2]})
    faced = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]})
    return Sheet(name, {"vert": vert, "edge": edge, "face": faced}, coords=["x", "y", "z"])


# --------------------------------------------------------------------------- #
# frame assembly (static topology → hoisted once)
# --------------------------------------------------------------------------- #
def assemble(slug, name, blurb, frames, is3d):
    verts0 = np.asarray(frames[0]["verts"]).reshape(-1, 3)
    allv = np.concatenate([np.asarray(f["verts"]).reshape(-1, 3) for f in frames])
    bounds = [allv.min(0).round(3).tolist(), allv.max(0).round(3).tolist()]
    n_cells = len(frames[0]["fields"]["num_sides"])
    model = {
        "name": name, "blurb": blurb, "is3d": is3d,
        "n_cells": int(n_cells), "n_verts": int(len(verts0)),
        "bounds": bounds, "face_fields": ["area", "num_sides"],
        "type_names": None, "frames": frames,
    }
    # topology is constant → hoist tris/edges/face_of_tri to the model
    keys = ("tris", "edges", "face_of_tri")
    for k in keys:
        model[k] = frames[0][k]
    for fr in frames:
        for k in keys:
            del fr[k]
        fr.pop("cell_type_raw", None)
    model["static_topology"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{slug}.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    version = hashlib.md5(path.read_bytes()).hexdigest()[:8]
    kb = path.stat().st_size / 1024
    print(f"  {slug:20s} {'3D' if is3d else '2D'}  {n_cells:>6d} cells  "
          f"{len(frames):>3d} frames  {kb:8.0f} KB")
    return {"file": f"{slug}.json", "slug": slug, "name": name, "version": version,
            "is3d": is3d, "n_cells": int(n_cells), "n_frames": len(frames)}


# --------------------------------------------------------------------------- #
# demo 1: differential-growth folding of a big flat epithelium
# --------------------------------------------------------------------------- #
def demo_folding(n=200, nframes=10, steps_per_frame=11, push=0.30, nfoci=30,
                 seed=3):
    print("2D epithelium folding into 3D ridges:")
    s = flat_sheet(n)
    bake(s, ka=1.0, kp=0.1, lt=0.06)
    fc = s.face_df[["x", "y"]].values
    xr = (fc[:, 0].min(), fc[:, 0].max())
    yr = (fc[:, 1].min(), fc[:, 1].max())
    rng = np.random.RandomState(seed)
    cx = rng.uniform(*xr, nfoci)
    cy = rng.uniform(*yr, nfoci)
    sgn = rng.choice([-1.0, 1.0], nfoci)  # alternating up / down folds
    sig = 0.09 * (xr[1] - xr[0])
    fz = np.zeros(len(fc))
    for k in range(nfoci):
        fz += sgn[k] * np.exp(-(((fc[:, 0]-cx[k])**2 + (fc[:, 1]-cy[k])**2) / (2*sig**2)))
    fz /= np.abs(fz).max()
    vz = _vertex_field(s, fz)          # per-vertex out-of-plane growth pattern
    inject = lambda P: np.column_stack([np.zeros(s.Nv), np.zeros(s.Nv), push * vz])
    step = make_stepper(s, dt=0.02, max_disp=0.15)
    frames = [snapshot(s, 0.0)]
    for fi in range(1, nframes):
        step(steps_per_frame, inject=inject)  # buckle deepens each frame
        frames.append(snapshot(s, round(fi / (nframes - 1), 3)))
    return assemble("scale_folding", "Folding epithelium · 40k cells",
                    "A flat epithelial sheet of ~40,000 individually simulated vertex-"
                    "model cells. Differential growth — patches of tissue growing faster "
                    "than their neighbours — buckles the sheet out of plane into a "
                    "landscape of folds and ridges (the same instability that shapes gut "
                    "villi and cortical folds). Real Rust vertex mechanics keep every "
                    "cell well-formed as it folds. Orbit to see the relief.",
                    frames, is3d=True)


# --------------------------------------------------------------------------- #
# demo 2: budding organoid sphere
# --------------------------------------------------------------------------- #
def demo_organoid(ncells=20000, R=12.0, ngrow=24, nframes=11, steps_per_frame=7,
                  push=0.06, sigma=0.26, seed=1):
    print("3D budding organoid sphere:")
    s = sphere_sheet(ncells, R=R)
    bake(s, ka=1.0, kp=0.08, lt=0.03)
    fc = s.face_df[["x", "y", "z"]].values
    fdir = fc / np.linalg.norm(fc, axis=1, keepdims=True)  # cell direction on sphere
    rng = np.random.RandomState(seed)
    centers = _fib_sphere(ngrow)[rng.permutation(ngrow)]  # scattered growth centres
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    ang = np.arccos(np.clip(fdir @ centers.T, -1, 1))  # (ncells, ngrow)
    bump = np.exp(-((ang / sigma) ** 2)).sum(1)
    bump = bump / bump.max()
    vbump = _vertex_field(s, bump)  # per-vertex growth strength
    # inject an OUTWARD (radial) growth force on the bud patches — this is what
    # physically protrudes them; area/tension alone just flattens in-surface.
    def inject(P):
        outward = P / np.linalg.norm(P, axis=1, keepdims=True)
        return push * vbump[:, None] * outward
    step = make_stepper(s, dt=0.02, max_disp=0.2)
    frames = [snapshot(s, 0.0)]
    for fi in range(1, nframes):
        step(steps_per_frame, inject=inject)  # buds grow out each frame
        frames.append(snapshot(s, round(fi / (nframes - 1), 3)))
    return assemble("scale_organoid", "Budding organoid · 20k cells (3D)",
                    "A hollow organoid: ~20,000 vertex-model cells tiling a sphere as a "
                    "spherical-Voronoï epithelium. Scattered growth centres drive patches "
                    "of cells to grow and bulge outward, so the smooth ball buds into a "
                    "lumpy organoid. Real Rust vertex mechanics keep the cells well-formed. "
                    "Orbit to inspect it; colour by area to see the buds.",
                    frames, is3d=True)


def main():
    t0 = time.time()
    manifest_new = [demo_folding(), demo_organoid()]
    idx_path = OUT / "index.json"
    existing = {}
    if idx_path.exists():
        for m in json.loads(idx_path.read_text()).get("models", []):
            existing[m["slug"]] = m
    for m in manifest_new:
        existing[m["slug"]] = m
    idx_path.write_text(json.dumps({"models": list(existing.values())}, indent=2),
                        encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s; manifest has {len(existing)} models")


if __name__ == "__main__":
    main()
