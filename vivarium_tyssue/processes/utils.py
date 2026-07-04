"""Rust-backend helpers for EulerSolver.

Single source of truth for (a) deciding whether the Rust `sheet_gradient`
kernel can serve a given model, and (b) marshalling an epithelium's geometry
columns into the kernel. Both the EulerSolver process and the equivalence test
import from here so the two paths can never drift.

The Rust path is opt-in (``backend: "rust"``) and only engages for the standard
3-effector sheet model; anything else falls back to Python transparently.
"""
import numpy as np

# The fused gradient kernel implements exactly these three effectors.
SUPPORTED_EFFECTORS = frozenset(
    {"LineTension", "FaceAreaElasticity", "PerimeterElasticity"}
)
SUPPORTED_FACTORIES = frozenset({"model_factory", "model_factory_bound"})


def rust_kernels_available():
    """True if the compiled ``tyssue_kernels`` module is importable."""
    try:
        import tyssue_kernels  # noqa: F401

        return True
    except ImportError:
        return False


def gradient_supported(effectors, factory, dim):
    """Whether the Rust gradient kernel can reproduce this model's gradient.

    Requires exactly the three standard sheet effectors (any order), one of the
    supported factories, a 3D mesh, and the compiled kernel present.
    """
    return (
        dim == 3
        and set(effectors) == SUPPORTED_EFFECTORS
        and factory in SUPPORTED_FACTORIES
        and rust_kernels_available()
    )


def rust_sheet_gradient(eptm, is_bound):
    """Return ``model.compute_gradient(eptm)`` for the standard sheet model as a
    ``(Nv, 3)`` ndarray in ``vert_df`` order, computed by the Rust kernel.

    Consumes tyssue's geometry columns as-is (so it reproduces the stale-length
    ``ucoords`` exactly). ``is_bound`` selects the boundary-vertex clamp that
    distinguishes ``model_factory_bound`` from ``model_factory``.
    """
    import tyssue_kernels as tk

    coords = eptm.coords
    ed, fd = eptm.edge_df, eptm.face_df
    vmap = {v: i for i, v in enumerate(eptm.vert_df.index)}
    fmap = {v: i for i, v in enumerate(fd.index)}
    C = np.ascontiguousarray

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
        C(ed["srce"].map(vmap).values, dtype=np.uint32),
        C(ed["trgt"].map(vmap).values, dtype=np.uint32),
        C(ed["face"].map(fmap).values, dtype=np.uint32),
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
        float(eptm.specs["settings"].get("nrj_norm_factor", 1.0)),
    )
    return np.asarray(flat).reshape(eptm.Nv, 3)
