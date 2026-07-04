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
# steps we replay in python). Monolayer/Bulk redefine the core -> unsupported.
_GEOM_SUPPORTED = {"SheetGeometry", "VesselGeometry"}


def geometry_supported(geom_name, dim):
    """Whether the Rust geometry path reproduces this geometry's update_all."""
    return dim == 3 and geom_name in _GEOM_SUPPORTED and rust_kernels_available()


def rust_geometry_update(eptm, geom, srce, trgt, face):
    """Rust replacement for ``geom.update_all``: the SheetGeometry core via the
    kernel, plus any cheap geometry-specific vertex steps. VesselGeometry adds
    ``update_tangents`` + ``update_vert_distance`` (vectorized numpy, no groupby),
    replayed here so the result is bit-identical to the python update_all."""
    rust_update_geometry(eptm, srce, trgt, face)
    if geom.__name__ == "VesselGeometry":
        geom.update_tangents(eptm)
        geom.update_vert_distance(eptm)


def rust_update_geometry(eptm, srce, trgt, face):
    """In-place replacement for ``SheetGeometry.update_all`` via the Rust kernel.

    Reproduces every column update_all writes — edge s*/t*/d*/u*/length/f*/r*/
    normals/sub_area/sub_vol and face centroid/area/perimeter/vol — bit-identical
    (~1e-15), including tyssue's stale-length ``ucoords`` quirk (ucoords uses the
    *previous* step's length). Does NOT recompute the boundary index / opposite
    edges: those are topology-invariant, so the caller refreshes them only on a
    topology change (a full python update_all after division/reconnect).

    ``srce``/``trgt``/``face`` are positional index arrays for the current
    topology (rebuild them when the mesh changes).
    """
    import tyssue_kernels as tk

    coords = eptm.coords
    ed, fd = eptm.edge_df, eptm.face_df
    pos = np.ascontiguousarray(eptm.vert_df[coords].values, dtype=np.float64)
    old_len = ed["length"].values.copy()  # stale length feeds ucoords, as in update_all
    g = tk.update_geometry(pos, srce, trgt, face, eptm.Nf)
    d = np.asarray(g["dcoords"]).reshape(-1, 3)
    cen = np.asarray(g["centroid"]).reshape(-1, 3)

    ed[["s" + c for c in coords]] = pos[srce]
    ed[["t" + c for c in coords]] = pos[trgt]
    ed[["d" + c for c in coords]] = d
    ed[["u" + c for c in coords]] = d / old_len[:, None]
    ed["length"] = np.asarray(g["length"])
    fd[coords] = cen
    ed[["f" + c for c in coords]] = cen[face]
    ed[["r" + c for c in coords]] = np.asarray(g["rcoords"]).reshape(-1, 3)
    ed[eptm.ncoords] = np.asarray(g["normals"]).reshape(-1, 3)
    ed["sub_area"] = np.asarray(g["sub_area"])
    fd["area"] = np.asarray(g["area"])
    fd["perimeter"] = np.asarray(g["perimeter"])
    if "height" in eptm.vert_df.columns:
        ed["sub_vol"] = eptm.upcast_srce(eptm.vert_df["height"]) * ed["sub_area"]
        fd["vol"] = eptm.sum_face(ed["sub_vol"])


def rust_sheet_gradient(eptm, is_bound, with_vessel=False, topo=None):
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
    """
    import tyssue_kernels as tk

    coords = eptm.coords
    ed, fd = eptm.edge_df, eptm.face_df
    C = np.ascontiguousarray
    if topo is None:
        vmap = {v: i for i, v in enumerate(eptm.vert_df.index)}
        fmap = {v: i for i, v in enumerate(fd.index)}
        srce = C(ed["srce"].map(vmap).values, dtype=np.uint32)
        trgt = C(ed["trgt"].map(vmap).values, dtype=np.uint32)
        face = C(ed["face"].map(fmap).values, dtype=np.uint32)
    else:
        srce, trgt, face = topo

    norm_factor = float(eptm.specs["settings"].get("nrj_norm_factor", 1.0))
    r_aj = ed[["t" + c for c in coords]].values - ed[["f" + c for c in coords]].values
    boundary = (
        eptm.vert_df["boundary"].values.astype(np.uint8)
        if is_bound and "boundary" in eptm.vert_df.columns
        else np.zeros(eptm.Nv, dtype=np.uint8)
    )
    flat = tk.sheet_gradient(
        C(ed[["u" + c for c in coords]].values, dtype=np.float64),
        C(ed[eptm.ncoords].values, dtype=np.float64),
        C(ed["sub_area"].values, dtype=np.float64),
        C(ed[["r" + c for c in coords]].values, dtype=np.float64),
        C(r_aj, dtype=np.float64),
        srce,
        trgt,
        face,
        C((ed["line_tension"] * ed["is_active"]).values, dtype=np.float64),
        C(
            (fd["perimeter_elasticity"] * fd["is_alive"] * (fd["perimeter"] - fd["prefered_perimeter"])).values,
            dtype=np.float64,
        ),
        C(
            (fd["area_elasticity"] * fd["is_alive"] * (fd["area"] - fd["prefered_area"])).values,
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
