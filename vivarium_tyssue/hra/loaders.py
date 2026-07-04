"""Loaders that turn Human Reference Atlas (HRA) data into tyssue meshes.

Data sources (all public, cached under ``workspace/datasets/hra_cache/``):

- reference-organ GLB meshes .... ``https://apps.humanatlas.io/api/v1/reference-organs``
- 2D FTU illustrations (SVG) .... ``https://apps.humanatlas.io/api/v1/ftu-illustrations``

The FTU illustration JSON gives, per functional tissue unit, a ``mapping`` of
illustration nodes — one per drawn cell — each tagged with its ``svg_id`` and
``svg_group_id`` (the cell type) plus an ontology ``representation_of`` (a Cell
Ontology CL id). The cell *positions* live in the companion SVG, keyed by
``svg_id``; :func:`_ftu_cell_centroids` reads them back out.
"""

from __future__ import annotations

import io
import re
import json
import contextlib
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

_API = "https://apps.humanatlas.io/api/v1"
_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE = _ROOT / "workspace" / "datasets" / "hra_cache"

_NUM = re.compile(r"-?\d+\.?\d*(?:e-?\d+)?")


# --------------------------------------------------------------------------- #
# cached download
# --------------------------------------------------------------------------- #
def _fetch(url: str, name: str, binary: bool = False):
    """Download ``url`` once into the cache and return its bytes/text."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    dest = _CACHE / name
    if not dest.exists():
        with urlopen(url, timeout=120) as r:  # noqa: S310 (trusted HRA host)
            dest.write_bytes(r.read())
    return dest.read_bytes() if binary else dest.read_text(encoding="utf-8")


def _api(endpoint: str, name: str):
    return json.loads(_fetch(f"{_API}/{endpoint}", name))


# --------------------------------------------------------------------------- #
# catalogs
# --------------------------------------------------------------------------- #
def reference_organs() -> pd.DataFrame:
    """All HRA reference organs with their GLB download URLs."""
    rows = _api("reference-organs", "reference-organs.json")
    out = []
    for o in rows:
        obj = o.get("object", {}) or {}
        out.append({"label": o.get("label", ""), "id": o.get("@id", ""),
                    "glb": obj.get("file", ""), "subpath": obj.get("file_subpath", "")})
    return pd.DataFrame(out)


def ftu_catalog() -> pd.DataFrame:
    """All 2D FTU illustrations with cell counts and SVG URLs."""
    rows = _api("ftu-illustrations", "ftu-illustrations.json")
    rows = rows if isinstance(rows, list) else rows.get("@graph", [])
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        svg = next((f["file"] for f in r.get("illustration_files", [])
                    if "svg" in str(f.get("file_format", ""))), "")
        out.append({"label": r.get("label", ""), "id": r.get("@id", ""),
                    "n_cells": len(r.get("mapping", [])), "svg": svg})
    return pd.DataFrame(out)


def _ftu_entry(match: str) -> dict:
    rows = _api("ftu-illustrations", "ftu-illustrations.json")
    rows = rows if isinstance(rows, list) else rows.get("@graph", [])
    m = match.lower()
    for r in rows:
        if isinstance(r, dict) and m in (r.get("label") or "").lower():
            return r
    raise ValueError(f"no FTU illustration matching {match!r}; "
                     f"see ftu_catalog() for options")


# --------------------------------------------------------------------------- #
# FTU 2D illustration  ->  cell centroids + types
# --------------------------------------------------------------------------- #
def _coords_under(el) -> np.ndarray:
    """Every (x, y) coordinate appearing in the paths/polygons under ``el``."""
    pts = []
    for e in el.iter():
        tag = re.sub("{.*}", "", e.tag)
        if tag == "path" and e.get("d"):
            pts += [float(x) for x in _NUM.findall(e.get("d"))]
        elif tag in ("polygon", "polyline") and e.get("points"):
            pts += [float(x) for x in _NUM.findall(e.get("points"))]
    if not pts:
        return np.zeros((0, 2))
    return np.array(pts[: len(pts) // 2 * 2]).reshape(-1, 2)


def _ftu_cell_centroids(match: str):
    """Return (centroids Nx2, cell_types list, type->CL id dict) for an FTU.

    Cell type comes from the illustration node's ``svg_group_id``; the centroid
    is the mean of all path coordinates inside the SVG element with that node's
    ``svg_id`` (SVG y is flipped so +y points up, tissue-style).
    """
    entry = _ftu_entry(match)
    nodes = entry["mapping"]
    types = {n["svg_id"]: n["svg_group_id"] for n in nodes}
    onto = {n["svg_group_id"]: n.get("representation_of", "") for n in nodes}
    svg_url = next(f["file"] for f in entry["illustration_files"]
                   if "svg" in str(f.get("file_format", "")))
    slug = re.sub(r"\W+", "-", entry["label"].lower()).strip("-")
    svg = _fetch(svg_url, f"{slug}.svg")
    root = ET.fromstring(svg)
    by_id = {e.get("id"): e for e in root.iter() if e.get("id")}

    cents, cell_types = [], []
    for sid, ctype in types.items():
        el = by_id.get(sid)
        if el is None:
            continue
        a = _coords_under(el)
        if len(a):
            cents.append(a.mean(0))
            cell_types.append(ctype)
    cents = np.asarray(cents)
    cents[:, 1] *= -1.0  # SVG y-down -> up
    return cents, cell_types, onto


def sheet_from_ftu(match: str = "crypt of Lieberkuhn", tile=(1, 1), gap: float = 1.15):
    """Build a 2D :class:`~tyssue.Sheet` from an HRA 2D FTU illustration.

    Each drawn cell becomes a face (Voronoï tessellation of the cell centroids)
    tagged with its real HRA cell type in ``face_df['cell_type']``. ``tile`` lays
    down an ``(nx, ny)`` field of copies of the FTU — an epithelium is, after
    all, a field of these units — so the same real layout scales to thousands of
    cells. ``gap`` spaces the tiles (>1 leaves a margin between units).

    Returns ``(sheet, meta)`` where ``meta`` has ``type_names`` and the
    ``ontology`` (cell type -> CL id) crosswalk.
    """
    from scipy.spatial import Voronoi
    from tyssue import Sheet, SheetGeometry
    from tyssue.generation import from_2d_voronoi

    from scipy.spatial import cKDTree

    cents, cell_types, onto = _ftu_cell_centroids(match)
    # Faithful tessellation. The raw HRA centroids carry the real crypt structure:
    # stem + neuroendocrine cells at the narrow base, absorptive/goblet up the
    # column, flaring to a wide villus top. We deliberately do NOT Lloyd-relax
    # (that homogenizes spacing and erases the base→villus zonation) and do NOT
    # remove cells by area (the flared-top cells are legitimately large). Instead:
    # guard-ring the cloud so every real cell is bounded, Voronoï-tessellate, then
    # keep exactly the one face nearest each real centroid — that selects the real
    # cells (with their true shapes + types) and discards the huge guard/exterior
    # faces.
    guard = _guard_ring(cents)
    vor = Voronoi(np.vstack([cents, guard]))
    with contextlib.redirect_stdout(io.StringIO()):
        dsets = from_2d_voronoi(vor)
        unit = Sheet(_slug(match), dsets, coords=["x", "y"])
    unit = _promote_to_flat_3d(unit, _slug(match))
    SheetGeometry.update_all(unit)
    fc = unit.face_df[["x", "y"]].values
    _, face_of_cell = cKDTree(fc).query(cents)
    unit.face_df["cell_type"] = "guard"
    for i, fid in enumerate(face_of_cell):
        unit.face_df.loc[fid, "cell_type"] = cell_types[i]
    unit = _keep_faces(unit, np.array(sorted(set(face_of_cell))))
    SheetGeometry.update_all(unit)
    # Light weld only for exactly-coincident Voronoï vertices (keeps the real
    # shapes; just removes zero-length edges that would break geometry).
    unit = _weld(unit, tol=0.01 * float(unit.edge_df["length"].median()))
    SheetGeometry.update_all(unit)
    _rescale_to_unit_area(unit)  # median cell area ~1 before baking targets
    # Bake per-cell REST targets so no cell carries a large area/perimeter
    # mismatch force (real cells vary wildly in size) — this keeps the mechanics
    # stable at scale regardless of the irregular real geometry. A small line
    # tension (set in the composite) then drives gentle, bounded motion.
    perim = unit.edge_df.groupby("face")["length"].sum().reindex(unit.face_df.index)
    unit.face_df["prefered_area"] = unit.face_df["area"].values
    unit.face_df["prefered_perimeter"] = perim.values

    sheet = _tile_mesh(unit, tile, gap)
    SheetGeometry.update_all(sheet)
    meta = {"type_names": sorted(set(cell_types)), "ontology": onto,
            "source": f"HRA 2D FTU · {match}"}
    return sheet, meta


# --------------------------------------------------------------------------- #
# 3D reference-organ GLB  ->  draped Sheet
# --------------------------------------------------------------------------- #
def sheet_from_organ_glb(match: str = "large intestine", keep: float = 0.15,
                         cell_types=None, seed: int = 0):
    """Drape a :class:`~tyssue.Sheet` over a real HRA reference-organ surface.

    Downloads the organ's GLB, decimates it to ``keep`` (fraction of faces
    retained), and rebuilds it as a tyssue half-edge sheet — one face per mesh
    triangle. If ``cell_types`` (a ``{name: proportion}`` dict, e.g. from
    :func:`asctb_cell_types`) is given, faces are labelled by sampling it.

    Returns ``(sheet, meta)``.
    """
    import trimesh
    from fast_simplification import simplify
    from tyssue import Sheet, SheetGeometry

    organs = reference_organs()
    hits = organs[organs["label"].str.lower().str.contains(match.lower())]
    if hits.empty:
        raise ValueError(f"no reference organ matching {match!r}")
    row = hits.iloc[0]
    glb = _fetch(row["glb"], f"{_slug(row['label'])}.glb", binary=True)
    scene = trimesh.load(io.BytesIO(glb), file_type="glb")
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene

    V0 = np.asarray(mesh.vertices, float)
    F0 = np.asarray(mesh.faces)
    V, F = simplify(V0, F0, target_reduction=1.0 - keep)
    V = np.asarray(V, float)
    F = np.asarray(F)
    sheet = _sheet_from_triangles(V, F, name=_slug(row["label"]))

    if cell_types:
        rng = np.random.RandomState(seed)
        names = list(cell_types)
        p = np.array([cell_types[n] for n in names], float)
        p = p / p.sum()
        sheet.face_df["cell_type"] = rng.choice(names, size=sheet.face_df.shape[0], p=p)

    with contextlib.redirect_stdout(io.StringIO()):
        SheetGeometry.update_all(sheet)
    _rescale_to_unit_area(sheet)  # median cell area ~1 before baking targets
    # Per-cell rest targets: decimated triangles vary in size, so rest targets
    # keep the mechanics stable on the real (non-uniform) surface.
    perim = sheet.edge_df.groupby("face")["length"].sum().reindex(sheet.face_df.index)
    sheet.face_df["prefered_area"] = sheet.face_df["area"].values
    sheet.face_df["prefered_perimeter"] = perim.values
    meta = {"type_names": list(cell_types) if cell_types else None,
            "source": f"HRA 3D reference organ · {row['label']}",
            "bbox_mm": [V.min(0).round(1).tolist(), V.max(0).round(1).tolist()]}
    return sheet, meta


# --------------------------------------------------------------------------- #
# ASCT+B cell-type roster
# --------------------------------------------------------------------------- #
def asctb_cell_types(ftu_match: str = "crypt of Lieberkuhn") -> dict:
    """Cell-type -> proportion for a tissue, read off the FTU illustration.

    The FTU illustration's node roster is itself an ASCT+B-linked cell census
    (every node carries a CL ontology id), so the drawn-cell counts give a real,
    normalized cell-type composition for that functional tissue unit.
    """
    entry = _ftu_entry(ftu_match)
    from collections import Counter
    c = Counter(n["svg_group_id"] for n in entry["mapping"])
    total = sum(c.values())
    return {k: v / total for k, v in sorted(c.items(), key=lambda kv: -kv[1])}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _slug(s: str) -> str:
    return re.sub(r"\W+", "-", s.lower()).strip("-")


def _promote_to_flat_3d(sheet, name: str = "sheet"):
    """Rebuild a 2D sheet as a flat 3D sheet (z=0) so the rust kernels — which
    need face normals — apply, matching the flat-3D form of the demo sheets.

    tyssue derives ``dcoords``/``ncoords`` at construction from ``coords``, so a
    genuine 3D sheet has to be *reconstructed* (setting ``sheet.coords`` alone
    leaves the stale 2D bookkeeping). ``face_df`` columns (incl. ``cell_type``)
    carry over.
    """
    from tyssue import Sheet

    vert = sheet.vert_df.copy()
    vert["z"] = 0.0
    face = sheet.face_df.copy()
    if "z" not in face:
        face["z"] = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        s3 = Sheet(name, {"vert": vert, "edge": sheet.edge_df.copy(), "face": face},
                   coords=["x", "y", "z"])
        s3.reset_index()
        s3.reset_topo()
    return s3


def _guard_ring(cents: np.ndarray) -> np.ndarray:
    """A rectangular frame of dummy points ~2 cell-widths outside the centroid
    cloud. Feeding these into the Voronoï bounds every real cell to a normal
    size (the frame absorbs the unbounded exterior regions)."""
    from scipy.spatial import cKDTree

    d, _ = cKDTree(cents).query(cents, k=2)
    step = float(np.median(d[:, 1]))  # typical cell spacing
    lo = cents.min(0) - 2 * step
    hi = cents.max(0) + 2 * step
    xs = np.arange(lo[0], hi[0] + step, step)
    ys = np.arange(lo[1], hi[1] + step, step)
    return np.vstack([
        np.c_[xs, np.full(len(xs), lo[1])],
        np.c_[xs, np.full(len(xs), hi[1])],
        np.c_[np.full(len(ys), lo[0]), ys],
        np.c_[np.full(len(ys), hi[0]), ys],
    ])


def _lloyd(pts: np.ndarray, n_iter: int = 2) -> np.ndarray:
    """A few Lloyd relaxation steps: move each point to the centroid of its
    Voronoï cell. Regularizes cell shapes (kills stiff slivers) while keeping
    points in place order (so cell-type labels stay aligned). Points on the
    convex hull (unbounded cells) are left where they are."""
    from scipy.spatial import Voronoi

    pts = np.asarray(pts, float).copy()
    for _ in range(n_iter):
        vor = Voronoi(pts)
        moved = pts.copy()
        for i, reg_idx in enumerate(vor.point_region):
            reg = vor.regions[reg_idx]
            if reg and -1 not in reg:
                moved[i] = vor.vertices[reg].mean(0)
        pts = moved
    return pts


def _weld(sheet, tol: float):
    """Merge vertices closer than ``tol`` (snap to a grid), drop collapsed
    (zero-length) edges and faces left with fewer than three sides. Rebuilds a
    clean Sheet — removes the singular near-zero-length edges a raw Voronoï of
    real cell centroids leaves behind."""
    from tyssue import Sheet

    xyz = sheet.vert_df[sheet.coords].values
    keyed = np.round(xyz / tol).astype(np.int64) if tol > 0 else xyz
    _, first, inv = np.unique(keyed, axis=0, return_index=True, return_inverse=True)
    # new vertex table: mean position of each merged group, in first-seen order
    order = np.argsort(first)
    remap = np.empty(len(first), dtype=int)
    remap[order] = np.arange(len(first))
    new_idx = remap[inv]
    nv = len(first)
    pos = np.zeros((nv, xyz.shape[1]))
    cnt = np.zeros(nv)
    np.add.at(pos, new_idx, xyz)
    np.add.at(cnt, new_idx, 1.0)
    pos /= cnt[:, None]
    vert = pd.DataFrame(pos, columns=sheet.coords)

    edge = sheet.edge_df.copy()
    edge["srce"] = new_idx[edge["srce"].values]
    edge["trgt"] = new_idx[edge["trgt"].values]
    edge = edge[edge["srce"] != edge["trgt"]]  # drop collapsed edges
    sides = edge.groupby("face").size()
    good_faces = sides.index[sides >= 3]
    edge = edge[edge["face"].isin(good_faces)]
    # reindex faces contiguously
    fmap = {old: new for new, old in enumerate(good_faces)}
    edge["face"] = edge["face"].map(fmap)
    face = sheet.face_df.loc[good_faces].reset_index(drop=True)
    with contextlib.redirect_stdout(io.StringIO()):
        s = Sheet(sheet.identifier, {"vert": vert, "edge": edge.reset_index(drop=True),
                                     "face": face}, coords=sheet.coords)
        s.reset_index()
        s.reset_topo()
    return s


def _rescale_to_unit_area(sheet):
    """Scale vertex positions so the median cell area is ~1 (a sane unit size
    for the mechanics). Must run BEFORE baking rest targets."""
    from tyssue import SheetGeometry

    med = float(np.sqrt(np.median(sheet.face_df["area"])))
    if med > 0:
        sheet.vert_df[sheet.coords] = sheet.vert_df[sheet.coords].values / med
        with contextlib.redirect_stdout(io.StringIO()):
            SheetGeometry.update_all(sheet)


def _keep_faces(sheet, keep_ids: np.ndarray):
    """Rebuild a sheet keeping only ``keep_ids`` faces (dataset-level, avoiding
    tyssue's fragile ``remove_face``). Edges/verts are pruned and reindexed."""
    from tyssue import Sheet

    fmap = {old: new for new, old in enumerate(keep_ids)}
    edge = sheet.edge_df[sheet.edge_df["face"].isin(keep_ids)].copy()
    used = np.unique(edge[["srce", "trgt"]].values.ravel())
    vmap = {old: new for new, old in enumerate(used)}
    edge["face"] = edge["face"].map(fmap)
    edge["srce"] = edge["srce"].map(vmap)
    edge["trgt"] = edge["trgt"].map(vmap)
    vert = sheet.vert_df.loc[used].reset_index(drop=True)
    face = sheet.face_df.loc[keep_ids].reset_index(drop=True)
    with contextlib.redirect_stdout(io.StringIO()):
        s = Sheet(sheet.identifier, {"vert": vert, "edge": edge.reset_index(drop=True),
                                     "face": face}, coords=sheet.coords)
        s.reset_index()
        s.reset_topo()
    return s


def _tile_mesh(unit, tile, gap: float):
    """Replicate a sheet into an ``(nx, ny)`` grid of disjoint copies.

    Copies the whole half-edge mesh (offsetting vertex positions and shifting
    srce/trgt/face ids per tile), so every cell stays exactly the clean shape of
    the source unit — no re-tessellation, no giant seam cells. ``gap`` (>1)
    leaves a margin between units.
    """
    from tyssue import Sheet

    nx, ny = tile
    span = (unit.vert_df[["x", "y"]].max() - unit.vert_df[["x", "y"]].min()).values
    dx, dy = span * gap
    nv = unit.vert_df.shape[0]
    nf = unit.face_df.shape[0]
    verts, edges, faces = [], [], []
    k = 0
    for i in range(nx):
        for j in range(ny):
            v = unit.vert_df.copy()
            v["x"] = v["x"].values + i * dx
            v["y"] = v["y"].values + j * dy
            e = unit.edge_df.copy()
            e["srce"] = e["srce"].values + k * nv
            e["trgt"] = e["trgt"].values + k * nv
            e["face"] = e["face"].values + k * nf
            verts.append(v)
            edges.append(e)
            faces.append(unit.face_df.copy())
            k += 1
    dsets = {"vert": pd.concat(verts, ignore_index=True),
             "edge": pd.concat(edges, ignore_index=True),
             "face": pd.concat(faces, ignore_index=True)}
    with contextlib.redirect_stdout(io.StringIO()):
        sheet = Sheet("field", dsets, coords=["x", "y", "z"])
        sheet.reset_index()
        sheet.reset_topo()
    return sheet


def _sheet_from_triangles(V: np.ndarray, F: np.ndarray, name: str = "mesh"):
    """A tyssue :class:`~tyssue.Sheet` from a triangle soup (verts + faces).

    Each triangle becomes a face with three directed half-edges; tyssue's
    ``reset_topo`` wires up the opposite-edge bookkeeping.
    """
    from tyssue import Sheet

    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    nf = len(F)
    edge = pd.DataFrame({"srce": np.concatenate([a, b, c]),
                         "trgt": np.concatenate([b, c, a]),
                         "face": np.tile(np.arange(nf), 3)})
    vert = pd.DataFrame(V, columns=["x", "y", "z"])
    face = pd.DataFrame(V[F].mean(1), columns=["x", "y", "z"])
    with contextlib.redirect_stdout(io.StringIO()):
        sheet = Sheet(name, {"vert": vert, "edge": edge, "face": face},
                      coords=["x", "y", "z"])
        sheet.reset_index()
        sheet.reset_topo()
    return sheet
