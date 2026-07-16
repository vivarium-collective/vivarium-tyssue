"""Compositional, extensible Rust-backed gradient for the tyssue EulerSolver.

Where ``utils.rust_sheet_gradient`` fused exactly three effectors into one
monolithic kernel (all-or-nothing), this module assembles a model's gradient
**one effector at a time**:

* effectors with a registered Rust adapter (``EFFECTOR_KERNELS``) run through a
  shared Rust *primitive* (``unit_edge_gradient`` / ``area_gradient`` / ...);
* every other effector transparently **falls back** to its own tyssue
  ``.gradient(eptm)`` — so a model mixing supported and unsupported terms still
  runs on the rust backend, with the covered terms accelerated and nothing
  regressing.

The edge->vertex assembly and the factory boundary masks are applied once, at
the end, reproducing ``model_factory.compute_gradient`` to floating-point
tolerance.

Adding coverage for a new effector is a small, local change — see
``rust-kernels/ADDING_A_KERNEL.md``:

1. if it reuses an existing primitive, add one adapter here and register it in
   ``EFFECTOR_KERNELS`` (no Rust);
2. if it needs new math, add a Rust primitive + pyo3 binding, then the adapter;
3. it is then auto-covered by the generic equivalence harness in
   ``tests/test_rust_kernels_equiv.py``.
"""
import numpy as np

C = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731

# Geometry families the rust geometry kernels reproduce exactly. Sheet/bulk are
# wired through EulerSolver.set_pos; planar (2D) is served by
# ``compute_geometry_planar`` (utils) and proven in the equivalence harness.
SHEET_GEOMS = frozenset({"SheetGeometry", "VesselGeometry"})
BULK_GEOMS = frozenset(
    {"BulkGeometry", "RNRGeometry", "MonolayerGeometry", "ClosedMonolayerGeometry"}
)
PLANAR_GEOMS = frozenset({"PlanarGeometry"})

# Factories whose boundary mask we reproduce (``_apply_boundary_mask``).
SUPPORTED_FACTORIES = frozenset(
    {"model_factory", "model_factory_bound", "model_factory_vessel", "model_factory_cylinder"}
)


# ---------------------------------------------------------------------------
# Geometry-column access — read the gradient inputs from the native geometry
# stash when present (lean materialization: they're not in pandas), else from
# the DataFrames. Bit-identical either way.
# ---------------------------------------------------------------------------
def _ucoords(eptm, geom):
    if geom is not None:
        return geom["ucoords"]
    return eptm.edge_df[["u" + c for c in eptm.coords]].values


def _area_inputs(eptm, geom):
    """(normals, r_ak, r_aj, sub_area) for the 3D area primitive."""
    coords = eptm.coords
    if geom is not None:
        r_ak = geom["rcoords"]  # srce - face_centroid
        r_aj = geom["tcoords"] - geom["fcoords"]
        return geom["normals"], r_ak, r_aj, geom["sub_area"]
    ed = eptm.edge_df
    face_pos = ed[["f" + c for c in coords]].values
    r_ak = ed[["s" + c for c in coords]].values - face_pos
    r_aj = ed[["t" + c for c in coords]].values - face_pos
    # 2D planar meshes store the out-of-plane normal as the scalar column "nz";
    # 3D meshes expose the (nx, ny, nz) block via eptm.ncoords.
    ncols = ["nz"] if len(coords) == 2 else eptm.ncoords
    return ed[ncols].values, r_ak, r_aj, ed["sub_area"].values


# ---------------------------------------------------------------------------
# Effector adapters. Each returns (grad_srce, grad_trgt, grad_vert) as numpy
# arrays (edge-shaped Ne×dim for srce/trgt, vert-shaped Nv×dim for vert) or
# None. They mirror the corresponding tyssue effector's ``gradient`` exactly.
# ---------------------------------------------------------------------------
def _unit_edge(eptm, geom, coeff):
    """srce = -u*coeff, trgt = +u*coeff (the length/tension family)."""
    import tyssue_kernels as tk

    dim = len(eptm.coords)
    g = tk.unit_edge_gradient(C(_ucoords(eptm, geom)), C(coeff))
    return np.asarray(g["srce"]).reshape(-1, dim), np.asarray(g["trgt"]).reshape(-1, dim), None


def _area(eptm, geom, coeff_edge):
    """area-family gradient via the Rust cross-product primitive."""
    import tyssue_kernels as tk

    dim = len(eptm.coords)
    normals, r_ak, r_aj, sub_area = _area_inputs(eptm, geom)
    if dim == 2:
        g = tk.area_gradient_2d(C(np.ravel(normals)), C(r_ak), C(r_aj), C(sub_area), C(coeff_edge))
    else:
        g = tk.area_gradient(C(normals), C(r_ak), C(r_aj), C(sub_area), C(coeff_edge))
    return np.asarray(g["srce"]).reshape(-1, dim), np.asarray(g["trgt"]).reshape(-1, dim), None


def _adapt_line_tension(eptm, geom):
    ed = eptm.edge_df
    coeff = (ed["line_tension"] * ed["is_active"] * 0.5).values
    return _unit_edge(eptm, geom, coeff)


def _adapt_perimeter_elasticity(eptm, geom):
    fd = eptm.face_df
    gamma = fd["perimeter_elasticity"] * fd["is_alive"] * (fd["perimeter"] - fd["prefered_perimeter"])
    return _unit_edge(eptm, geom, eptm.upcast_face(gamma).values)


def _adapt_face_contractility(eptm, geom):
    fd = eptm.face_df
    gamma = fd["contractility"] * fd["perimeter"] * fd["is_alive"]
    return _unit_edge(eptm, geom, eptm.upcast_face(gamma).values)


def _adapt_length_elasticity(eptm, geom):
    ed = eptm.edge_df
    coeff = (ed["length_elasticity"] * ed["is_alive"] * (ed["length"] - ed["prefered_length"])).values
    return _unit_edge(eptm, geom, coeff)


def _adapt_border_elasticity(eptm, geom):
    ed = eptm.edge_df
    kl = ed["border_elasticity"] * ed["is_active"] * ed["is_border"] * (ed["length"] - ed["prefered_length"])
    # tyssue returns (+u*kl/2, -u*kl/2); in the -u*coeff convention that's coeff=-kl/2
    return _unit_edge(eptm, geom, (-kl / 2).values)


def _adapt_face_area_elasticity(eptm, geom):
    fd = eptm.face_df
    ka = fd["area_elasticity"] * fd["is_alive"] * (fd["area"] - fd["prefered_area"])
    return _area(eptm, geom, eptm.upcast_face(ka).values)


def _adapt_surface_tension(eptm, geom):
    return _area(eptm, geom, eptm.upcast_face(eptm.face_df["surface_tension"]).values)


def _adapt_cell_area_elasticity(eptm, geom):
    cd = eptm.cell_df
    ka = cd["area_elasticity"] * cd["is_alive"] * (cd["area"] - cd["prefered_area"])
    return _area(eptm, geom, eptm.upcast_cell(ka).values)


# name -> adapter(eptm, geom_stash) -> (grad_srce, grad_trgt, grad_vert)
EFFECTOR_KERNELS = {
    "LineTension": _adapt_line_tension,
    "PerimeterElasticity": _adapt_perimeter_elasticity,
    "FaceContractility": _adapt_face_contractility,
    "LengthElasticity": _adapt_length_elasticity,
    "BorderElasticity": _adapt_border_elasticity,
    "FaceAreaElasticity": _adapt_face_area_elasticity,
    "SurfaceTension": _adapt_surface_tension,
    "CellAreaElasticity": _adapt_cell_area_elasticity,
}


def effector_covered(name):
    """True if ``name`` runs through a Rust primitive (else Python fallback)."""
    return name in EFFECTOR_KERNELS


def _fallback(effector, eptm):
    """Per-term Python fallback: classify the effector's own gradient exactly as
    ``model_factory.compute_gradient`` does (edge-srce/edge-trgt vs vertex)."""
    g0, g1 = effector.gradient(eptm)
    a0 = np.asarray(g0, dtype=np.float64)
    if a0.shape[0] == eptm.Ne:
        trgt = None
        if g1 is not None:
            a1 = np.asarray(g1, dtype=np.float64)
            if a1.shape[0] == eptm.Ne:
                trgt = a1
        return a0, trgt, None
    if a0.shape[0] == eptm.Nv:
        return None, None, a0
    return None, None, None


def _apply_boundary_mask(eptm, grad, factory):
    """Reproduce the factory's boundary-vertex clamp on ``grad`` (Nv×dim, in
    vert_df order). Mirrors ``dynamics/factory.py``."""
    vd = eptm.vert_df
    if factory == "model_factory_bound":
        grad[vd["boundary"].values == 1] = 0.0
    elif factory == "model_factory_vessel":
        m = (vd["boundary"].values == 1) & (vd["z"].values < 1)
        grad[m, 2] = 0.0
    elif factory == "model_factory_cylinder":
        b = vd["boundary"].values == 1
        z = vd["z"].values
        grad[b & (z < 0)] = 0.0
        m2 = b & (z > 0)
        grad[m2, 0] = 0.0
        grad[m2, 1] = 0.0
    # model_factory: no mask


def rust_model_gradient(eptm, effectors, factory_name, topo=None, geom=None):
    """Return ``model.compute_gradient(eptm)`` as an ``(Nv, dim)`` ndarray in
    vert_df order, assembled compositionally: covered effectors via Rust
    primitives, the rest via their tyssue ``.gradient``. ``factory_name`` selects
    the boundary mask. ``topo`` is an optional ``(srce, trgt, face)`` tuple of
    positional uint32 arrays; ``geom`` an optional native geometry stash (skips
    re-reading edge columns from pandas)."""
    import tyssue_kernels as tk

    dim = len(eptm.coords)
    Ne, Nv = eptm.Ne, eptm.Nv
    # Guard against a stale geometry stash: if a topology change slipped through
    # without the solver rebuilding it (array length != current edge count), drop
    # it and read geometry from the DataFrames (right length) instead of feeding a
    # mismatched array to the rust primitives, which would index out of bounds.
    if geom is not None and len(np.asarray(geom["sub_area"])) != Ne:
        geom = None
    if topo is None:
        vmap = {v: i for i, v in enumerate(eptm.vert_df.index)}
        fd_index = eptm.face_df.index
        fmap = {v: i for i, v in enumerate(fd_index)}
        srce = C(eptm.edge_df["srce"].map(vmap).values).astype(np.uint32)
        trgt = C(eptm.edge_df["trgt"].map(vmap).values).astype(np.uint32)
    else:
        srce, trgt = topo[0], topo[1]

    srce_sum = trgt_sum = vert_sum = None
    for eff in effectors:
        adapter = EFFECTOR_KERNELS.get(eff.__name__)
        s, t, v = adapter(eptm, geom) if adapter is not None else _fallback(eff, eptm)
        if s is not None:
            srce_sum = s if srce_sum is None else srce_sum + s
        if t is not None:
            trgt_sum = t if trgt_sum is None else trgt_sum + t
        if v is not None:
            vert_sum = v if vert_sum is None else vert_sum + v

    grad = np.zeros((Nv, dim), dtype=np.float64)
    if srce_sum is not None:
        grad += np.asarray(tk.scatter_add(C(srce_sum), srce, Nv)).reshape(Nv, dim)
    if trgt_sum is not None:
        grad += np.asarray(tk.scatter_add(C(trgt_sum), trgt, Nv)).reshape(Nv, dim)
    if vert_sum is not None:
        grad += vert_sum

    _apply_boundary_mask(eptm, grad, factory_name)
    norm_factor = float(eptm.specs["settings"].get("nrj_norm_factor", 1.0))
    if norm_factor != 1.0:
        grad /= norm_factor
    return grad
