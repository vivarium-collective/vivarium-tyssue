"""Rust-backend helpers for EulerSolver.

Single source of truth for (a) deciding whether the Rust `sheet_gradient`
kernel can serve a given model, and (b) marshalling an epithelium's geometry
columns into the kernel. Both the EulerSolver process and the equivalence test
import from here so the two paths can never drift.

The Rust path is opt-in (``backend: "rust"``) and only engages for the standard
3-effector sheet model; anything else falls back to Python transparently.
"""
import numpy as np

# The fused gradient kernel implements exactly these three edge effectors.
SUPPORTED_EFFECTORS = frozenset(
    {"LineTension", "FaceAreaElasticity", "PerimeterElasticity"}
)
# The cylinder demos add a vertex-based radial effector, handled as a cheap
# numpy term on top of the kernel (see rust_sheet_gradient).
VESSEL_EFFECTORS = SUPPORTED_EFFECTORS | {"VesselSurfaceElasticity"}
SUPPORTED_FACTORIES = frozenset({"model_factory", "model_factory_bound"})


def rust_kernels_available():
    """True if the compiled ``tyssue_kernels`` module is importable."""
    try:
        import tyssue_kernels  # noqa: F401

        return True
    except ImportError:
        return False


def gradient_supported(effectors, factory, dim):
    """Whether the Rust gradient path can reproduce this model's gradient.

    The three standard sheet effectors (any order), optionally plus the vessel
    radial effector; one of the supported factories; a 3D mesh; kernel present.
    """
    return (
        dim == 3
        and set(effectors) in (SUPPORTED_EFFECTORS, VESSEL_EFFECTORS)
        and factory in SUPPORTED_FACTORIES
        and rust_kernels_available()
    )


def has_vessel_effector(effectors):
    return "VesselSurfaceElasticity" in effectors


# Geometries whose update_all == SheetGeometry.update_all core (+ cheap vertex
# steps we replay in python). The Rust *sheet* kernel serves these.
_GEOM_SUPPORTED = {"SheetGeometry", "VesselGeometry"}
# 3D volumetric geometries (cells with volume) served by the Rust *bulk* kernel —
# they redefine the core with weighted centroids + per-cell areas/volumes.
_BULK_GEOM_SUPPORTED = {
    "BulkGeometry", "RNRGeometry", "MonolayerGeometry", "ClosedMonolayerGeometry",
}


def is_bulk_geometry(geom_name):
    """True if this geometry needs the Rust *bulk* (Monolayer/Bulk) kernel."""
    return geom_name in _BULK_GEOM_SUPPORTED


def geometry_supported(geom_name, dim):
    """Whether a Rust geometry path (sheet or bulk) reproduces this update_all."""
    return (
        dim == 3
        and (geom_name in _GEOM_SUPPORTED or geom_name in _BULK_GEOM_SUPPORTED)
        and rust_kernels_available()
    )


def rust_bulk_geometry_update(eptm, srce, trgt, face, cell, pos=None):
    """In-place Rust replacement for ``MonolayerGeometry/BulkGeometry.update_all``:
    ``compute_geometry_bulk`` then ``materialize_geometry_bulk`` (edge/face/cell
    frames). Bit-identical (~1e-13). ``cell`` is the positional cell index of each
    edge. Returns the ``geom`` dict."""
    coords = eptm.coords
    if pos is None:
        pos = eptm.vert_df[coords].values
    old_len = eptm.edge_df["length"].values.copy()  # stale length feeds ucoords
    geom = compute_geometry_bulk(eptm, srce, trgt, face, cell, pos, old_len)
    materialize_geometry_bulk(eptm, geom, which=("edge", "face", "cell"))
    return geom


def rust_geometry_update(eptm, geom, srce, trgt, face, pos=None, full=True):
    """Rust replacement for ``geom.update_all``: the SheetGeometry core via the
    kernel, plus any cheap geometry-specific vertex steps. VesselGeometry adds
    ``update_tangents`` + ``update_vert_distance`` (vectorized numpy, no groupby),
    replayed here so the result is bit-identical to the python update_all.

    ``pos`` is an optional full ``(Nv, dim)`` float64 vertex-position array the
    caller already holds (from the integration step), passed to skip re-reading
    it out of ``vert_df``. Only valid when it covers *all* vertices in index order.

    Returns the ``stash`` dict of gradient-input arrays from the kernel (see
    ``rust_update_geometry``) so the caller can feed it straight into
    ``rust_sheet_gradient`` and skip re-reading those columns from pandas."""
    stash = rust_update_geometry(eptm, srce, trgt, face, pos=pos, full=full)
    if geom.__name__ == "VesselGeometry":
        geom.update_tangents(eptm)
        geom.update_vert_distance(eptm)
    return stash


def compute_geometry(eptm, srce, trgt, face, pos, old_len):
    """Pure-native geometry compute via the Rust kernel — writes **no** DataFrames.

    Returns a ``geom`` dict of every array ``SheetGeometry.update_all`` would put
    in the edge/face frames (scoords/tcoords/dcoords/ucoords/length/centroid/
    fcoords/rcoords/normals/sub_area/area/perimeter), bit-identical (~1e-15),
    including tyssue's stale-length ``ucoords`` quirk — hence ``old_len`` (the
    *previous* step's edge length) is passed explicitly rather than read from the
    frame, so a native sub-step loop can chain steps without touching pandas.

    ``pos`` is the full ``(Nv, dim)`` vertex-position array in index order;
    ``srce``/``trgt``/``face`` are positional index arrays for the topology. The
    returned dict is also exactly what ``rust_sheet_gradient(geom=...)`` consumes.
    """
    import tyssue_kernels as tk

    pos = np.ascontiguousarray(pos, dtype=np.float64)
    g = tk.update_geometry(pos, srce, trgt, face, eptm.Nf)
    d = np.asarray(g["dcoords"]).reshape(-1, 3)
    cen = np.asarray(g["centroid"]).reshape(-1, 3)
    return {
        "scoords": pos[srce],
        "tcoords": pos[trgt],
        "dcoords": d,
        "ucoords": d / old_len[:, None],
        "length": np.asarray(g["length"]),
        "centroid": cen,
        "fcoords": cen[face],
        "rcoords": np.asarray(g["rcoords"]).reshape(-1, 3),
        "normals": np.asarray(g["normals"]).reshape(-1, 3),
        "sub_area": np.asarray(g["sub_area"]),
        "area": np.asarray(g["area"]),
        "perimeter": np.asarray(g["perimeter"]),
    }


def materialize_geometry(eptm, geom, which=("edge", "face"), full=True):
    """Modular converter: write native geometry arrays (from ``compute_geometry``)
    back into the epithelium's tyssue DataFrames — the single place the Rust path
    touches pandas for geometry.

    This is the user-facing "convert to DataFrames only where it matters" hook:
    the native integration keeps ``geom`` in arrays and calls this only at an
    interface (emit, inspection, before behaviours that read the frames). ``which``
    selects which frames to write (``"edge"``, ``"face"``); the result is
    bit-identical to ``SheetGeometry.update_all``'s column writes.

    ``full`` selects how many edge columns to write (Phase B — "materialize only
    what's observed"):

    - ``full=True`` (default): every column ``update_all`` writes, incl. the seven
      three-wide *intermediate* edge coordinate blocks (s/t/d/u/f/r-coords,
      normals) + sub_area — the bulk of the per-step pandas cost.
    - ``full=False``: only the *observable* geometry — edge ``length`` and the
      whole face frame (centroid/area/perimeter, +vol in 3D). The intermediate
      blocks are pure gradient inputs; the Rust gradient reads them from the native
      ``geom`` stash, not pandas, and nothing downstream (emitters, viz, behaviours)
      reads them, so skipping their write each step is unobservable. They stay
      materialisable on demand via ``EulerSolver.to_dataframes(full=True)``.
    """
    coords = eptm.coords
    ed, fd = eptm.edge_df, eptm.face_df
    has_height = "height" in eptm.vert_df.columns
    # A stash whose edge/face counts no longer match the frames is stale (a
    # topology change the solver didn't rebuild it after): writing it would raise
    # a length-mismatch. Skip rather than corrupt/crash — the next set_pos rebuilds
    # a fresh stash, and observable geometry is recomputed then.
    if "edge" in which and len(np.asarray(geom["length"])) != len(ed):
        return
    if "face" in which and len(np.asarray(geom["area"])) != len(fd):
        return
    if "edge" in which:
        ed["length"] = geom["length"]
        if full:
            ed[["s" + c for c in coords]] = geom["scoords"]
            ed[["t" + c for c in coords]] = geom["tcoords"]
            ed[["d" + c for c in coords]] = geom["dcoords"]
            ed[["u" + c for c in coords]] = geom["ucoords"]
            ed[["f" + c for c in coords]] = geom["fcoords"]
            ed[["r" + c for c in coords]] = geom["rcoords"]
            ed[eptm.ncoords] = geom["normals"]
        if full or has_height:  # sub_area feeds the 3D sub_vol below
            ed["sub_area"] = geom["sub_area"]
    if "face" in which:
        fd[coords] = geom["centroid"]
        fd["area"] = geom["area"]
        fd["perimeter"] = geom["perimeter"]
    if has_height and "edge" in which:
        ed["sub_vol"] = eptm.upcast_srce(eptm.vert_df["height"]) * ed["sub_area"]
        if "face" in which:
            fd["vol"] = eptm.sum_face(ed["sub_vol"])


def compute_geometry_bulk(eptm, srce, trgt, face, cell, pos, old_len):
    """Pure-native Bulk/Monolayer geometry via the Rust kernel — writes no
    DataFrames. Returns every array ``MonolayerGeometry.update_all`` produces
    (edge s/t/d/u/f/r/c/normals/sub_area/sub_vol/length, face centroid/area/
    perimeter, cell centroid/area/vol), bit-identical (~1e-13). ``cell`` is the
    positional cell index of each edge; ``old_len`` the previous edge length
    (feeds the stale ``ucoords`` quirk)."""
    import tyssue_kernels as tk

    pos = np.ascontiguousarray(pos, dtype=np.float64)
    g = tk.update_geometry_bulk(pos, srce, trgt, face, cell, eptm.Nf, eptm.Nc)
    d = np.asarray(g["dcoords"]).reshape(-1, 3)
    fcen = np.asarray(g["face_centroid"]).reshape(-1, 3)
    ccen = np.asarray(g["cell_centroid"]).reshape(-1, 3)
    return {
        "scoords": pos[srce],
        "tcoords": pos[trgt],
        "dcoords": d,
        "ucoords": d / old_len[:, None],
        "length": np.asarray(g["length"]),
        "face_centroid": fcen,
        "cell_centroid": ccen,
        "fcoords": fcen[face],
        "rcoords": np.asarray(g["rcoords"]).reshape(-1, 3),
        "ccoords": ccen[cell],
        "normals": np.asarray(g["normals"]).reshape(-1, 3),
        "sub_area": np.asarray(g["sub_area"]),
        "sub_vol": np.asarray(g["sub_vol"]),
        "face_area": np.asarray(g["face_area"]),
        "cell_area": np.asarray(g["cell_area"]),
        "cell_vol": np.asarray(g["cell_vol"]),
        "perimeter": np.asarray(g["perimeter"]),
    }


def materialize_geometry_bulk(eptm, geom, which=("edge", "face", "cell")):
    """Write native Bulk/Monolayer geometry arrays (from ``compute_geometry_bulk``)
    back into the epithelium's edge/face/cell DataFrames — the bulk analogue of
    ``materialize_geometry``, adding the c-coords, sub_vol, and the whole cell_df
    (centroid/area/vol)."""
    coords = eptm.coords
    ed, fd, cd = eptm.edge_df, eptm.face_df, eptm.cell_df
    if "edge" in which:
        ed[["s" + c for c in coords]] = geom["scoords"]
        ed[["t" + c for c in coords]] = geom["tcoords"]
        ed[["d" + c for c in coords]] = geom["dcoords"]
        ed[["u" + c for c in coords]] = geom["ucoords"]
        ed["length"] = geom["length"]
        ed[["f" + c for c in coords]] = geom["fcoords"]
        ed[["r" + c for c in coords]] = geom["rcoords"]
        ed[["c" + c for c in coords]] = geom["ccoords"]
        ed[eptm.ncoords] = geom["normals"]
        ed["sub_area"] = geom["sub_area"]
        ed["sub_vol"] = geom["sub_vol"]
    if "face" in which:
        fd[coords] = geom["face_centroid"]
        fd["area"] = geom["face_area"]
        fd["perimeter"] = geom["perimeter"]
    if "cell" in which:
        cd[coords] = geom["cell_centroid"]
        cd["area"] = geom["cell_area"]
        cd["vol"] = geom["cell_vol"]


def rust_update_geometry(eptm, srce, trgt, face, pos=None, full=True):
    """In-place replacement for ``SheetGeometry.update_all`` via the Rust kernel:
    ``compute_geometry`` then ``materialize_geometry`` (both edge and face frames).

    Reproduces every column update_all writes, bit-identical (~1e-15). Does NOT
    recompute the boundary index / opposite edges — topology-invariant, refreshed
    by the caller only on a topology change. Returns the ``geom`` dict so the
    caller can feed it straight to ``rust_sheet_gradient`` (skips re-reading the
    Ne×3 coordinate blocks from pandas).

    ``full=False`` writes only the observable geometry (edge length + face frame);
    the intermediate coordinate blocks stay in the returned ``geom`` stash. See
    ``materialize_geometry``."""
    coords = eptm.coords
    if pos is None:
        pos = eptm.vert_df[coords].values
    old_len = eptm.edge_df["length"].values.copy()  # stale length feeds ucoords
    geom = compute_geometry(eptm, srce, trgt, face, pos, old_len)
    materialize_geometry(eptm, geom, which=("edge", "face"), full=full)
    return geom


def rust_sheet_gradient(eptm, is_bound, with_vessel=False, topo=None, geom=None):
    """Return ``model.compute_gradient(eptm)`` for the standard sheet model as a
    ``(Nv, 3)`` ndarray in ``vert_df`` order, computed by the Rust kernel.

    Consumes tyssue's geometry columns as-is (so it reproduces the stale-length
    ``ucoords`` exactly). ``is_bound`` selects the boundary-vertex clamp that
    distinguishes ``model_factory_bound`` from ``model_factory``. ``with_vessel``
    adds the VesselSurfaceElasticity vertex term (a cheap radial numpy term).

    ``topo`` is an optional ``(srce, trgt, face)`` tuple of positional uint32
    index arrays for the current topology (from ``EulerSolver._topo_arrays``).
    Pass it to skip rebuilding the vertex/face lookup dicts and the three pandas
    ``.map`` calls every step — a pure per-update saving. When ``None`` they're
    derived here (keeps the standalone/equivalence-test call sites simple).

    ``geom`` is an optional stash of the geometry arrays (``ucoords``, ``normals``,
    ``sub_area``, ``rcoords``, ``tcoords``, ``fcoords``, ``perimeter``, ``area``)
    from the geometry kernel this step (``rust_update_geometry``'s return value).
    Pass it to skip re-reading those (Ne×3) coordinate blocks from pandas — the
    dominant per-update cost once the kernels themselves are ~free. When ``None``
    they're read from the DataFrame (bit-identical values either way).
    """
    import tyssue_kernels as tk

    coords = eptm.coords
    ed, fd = eptm.edge_df, eptm.face_df
    C = np.ascontiguousarray
    # Drop a stale geometry stash (wrong edge count after an untracked topology
    # change) and fall back to the DataFrame columns, which are always the current
    # length — otherwise the mismatched arrays index out of bounds in the kernel.
    if geom is not None and len(np.asarray(geom["sub_area"])) != eptm.Ne:
        geom = None
    if topo is None:
        vmap = {v: i for i, v in enumerate(eptm.vert_df.index)}
        fmap = {v: i for i, v in enumerate(fd.index)}
        srce = C(ed["srce"].map(vmap).values, dtype=np.uint32)
        trgt = C(ed["trgt"].map(vmap).values, dtype=np.uint32)
        face = C(ed["face"].map(fmap).values, dtype=np.uint32)
    else:
        srce, trgt, face = topo

    norm_factor = float(eptm.specs["settings"].get("nrj_norm_factor", 1.0))
    if geom is None:
        ucoords = ed[["u" + c for c in coords]].values
        normals = ed[eptm.ncoords].values
        sub_area = ed["sub_area"].values
        rcoords = ed[["r" + c for c in coords]].values
        r_aj = ed[["t" + c for c in coords]].values - ed[["f" + c for c in coords]].values
    else:
        ucoords, normals, sub_area, rcoords = (
            geom["ucoords"], geom["normals"], geom["sub_area"], geom["rcoords"],
        )
        r_aj = geom["tcoords"] - geom["fcoords"]
    perimeter = geom["perimeter"] if geom is not None else fd["perimeter"].values
    area = geom["area"] if geom is not None else fd["area"].values
    boundary = (
        eptm.vert_df["boundary"].values.astype(np.uint8)
        if is_bound and "boundary" in eptm.vert_df.columns
        else np.zeros(eptm.Nv, dtype=np.uint8)
    )
    flat = tk.sheet_gradient(
        C(ucoords, dtype=np.float64),
        C(normals, dtype=np.float64),
        C(sub_area, dtype=np.float64),
        C(rcoords, dtype=np.float64),
        C(r_aj, dtype=np.float64),
        srce,
        trgt,
        face,
        C((ed["line_tension"] * ed["is_active"]).values, dtype=np.float64),
        C(
            (fd["perimeter_elasticity"] * fd["is_alive"] * (perimeter - fd["prefered_perimeter"])).values,
            dtype=np.float64,
        ),
        C(
            (fd["area_elasticity"] * fd["is_alive"] * (area - fd["prefered_area"])).values,
            dtype=np.float64,
        ),
        C(boundary, dtype=np.uint8),
        eptm.Nv,
        norm_factor,
    )
    grad = np.asarray(flat).reshape(eptm.Nv, 3)

    if with_vessel:
        # VesselSurfaceElasticity: a per-vertex radial force added to grad_i,
        # ka = vessel_elasticity*is_alive*(distance_origin - prefered_radius),
        # along the radial unit (ox, oy, 0). Divided by norm_factor like the rest.
        vd = eptm.vert_df
        ka = (vd["vessel_elasticity"] * vd["is_alive"] * (vd["distance_origin"] - vd["prefered_radius"])).values
        radial = np.zeros((eptm.Nv, 3))
        radial[:, 0] = vd["ox"].values
        radial[:, 1] = vd["oy"].values
        grad += (ka[:, None] * radial) / norm_factor
    return grad
