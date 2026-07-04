"""Numeric-equivalence gate for the Phase-A Rust backend (``tyssue_kernels``).

This is the correctness contract the Rust hot-kernel crate must satisfy: every
kernel must reproduce the Python/tyssue reference to floating-point identity
(``atol=1e-12``) on the *actual* demo meshes. The composite run-one-step tests
in ``test_composites.py`` are the behavioral backstop on top of this.

The Rust module is optional — skipped cleanly if ``tyssue_kernels`` isn't built
(see ``rust-kernels/README.md`` for ``maturin develop``). Nothing here imports
the module at collection time so CI without Rust stays green.

As new kernels land (``update_geometry``, ``assemble_gradient``), add an
equivalence test here BEFORE wiring them into ``EulerSolver``.
"""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

# (mesh file, tissue_type, geometry) triples mirroring the shipped composites so
# the kernels are exercised on every geometry a demo actually uses.
MESHES = [
    ("test_square.hf5", "Sheet", "SheetGeometry"),        # flat sheet (2D-in-3D)
    ("test_cylinder.hf5", "Sheet", "VesselGeometry"),     # vessel / cylinder
    ("monolayer_box.hf5", "Monolayer", "MonolayerGeometry"),  # 3D monolayer
]

# The full-geometry kernel implements the SheetGeometry formula, which
# VesselGeometry inherits unchanged. MonolayerGeometry (a BulkGeometry) redefines
# centroid/area/normals, so it needs its own kernel — excluded here on purpose.
SHEET_MESHES = [m for m in MESHES if m[2] in ("SheetGeometry", "VesselGeometry")]


def _load_eptm(mesh, tissue, geom_key):
    """Build + update a demo epithelium exactly as ``EulerSolver.initialize`` does."""
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    import sys

    sys.path.insert(0, str(ROOT))
    from tyssue.io.hdf5 import load_datasets
    from vivarium_tyssue.maps import GEOMETRY_MAP, TISSUE_MAP

    path = ROOT / "workspace" / "datasets" / mesh
    if not path.exists():
        pytest.skip(f"mesh fixture missing: {mesh}")
    eptm = TISSUE_MAP[tissue]("e", load_datasets(str(path)))
    geom = GEOMETRY_MAP[geom_key]
    geom.update_all(eptm)
    return eptm


def _positional_edges(eptm):
    """Return (pos, srce, trgt) as contiguous arrays with 0-based vertex indices.

    tyssue's ``srce``/``trgt`` reference the vert_df *index* values, which need
    not be a 0..Nv-1 range; the kernels take positional indices, so remap once.
    """
    coords = eptm.coords
    pos = np.ascontiguousarray(eptm.vert_df[coords].values, dtype=np.float64)
    vmap = {v: i for i, v in enumerate(eptm.vert_df.index)}
    srce = np.ascontiguousarray(eptm.edge_df["srce"].map(vmap).values, dtype=np.uint32)
    trgt = np.ascontiguousarray(eptm.edge_df["trgt"].map(vmap).values, dtype=np.uint32)
    return pos, srce, trgt, len(coords)


@pytest.mark.parametrize("mesh,tissue,geom", MESHES, ids=lambda x: x if isinstance(x, str) else "")
def test_edge_lengths_match_tyssue(mesh, tissue, geom):
    """Rust ``edge_lengths`` == tyssue's own ``edge_df['length']``, bit-for-bit."""
    k = pytest.importorskip("tyssue_kernels", reason="Rust kernels not built")
    eptm = _load_eptm(mesh, tissue, geom)
    pos, srce, trgt, _ = _positional_edges(eptm)

    ref = eptm.edge_df["length"].values.astype(np.float64)
    got = np.asarray(k.edge_lengths(pos, srce, trgt))

    assert got.shape == ref.shape
    assert np.allclose(got, ref, atol=1e-12, rtol=0.0), (
        f"{mesh}: max|Δ|={np.max(np.abs(got - ref)):.3e}"
    )


@pytest.mark.parametrize("mesh,tissue,geom", MESHES, ids=lambda x: x if isinstance(x, str) else "")
def test_scatter_add_matches_numpy(mesh, tissue, geom):
    """Rust ``scatter_add`` == ``np.add.at`` — the edge->vertex gradient assembly."""
    k = pytest.importorskip("tyssue_kernels", reason="Rust kernels not built")
    eptm = _load_eptm(mesh, tissue, geom)
    pos, srce, trgt, dim = _positional_edges(eptm)

    rng = np.random.default_rng(0)
    vals = np.ascontiguousarray(rng.standard_normal((eptm.Ne, dim)), dtype=np.float64)
    ref = np.zeros((eptm.Nv, dim))
    np.add.at(ref, srce.astype(np.int64), vals)
    got = np.asarray(k.scatter_add(vals, srce, eptm.Nv)).reshape(eptm.Nv, dim)

    assert np.allclose(got, ref, atol=1e-12, rtol=0.0), (
        f"{mesh}: max|Δ|={np.max(np.abs(got - ref)):.3e}"
    )


@pytest.mark.parametrize("mesh,tissue,geom", SHEET_MESHES, ids=lambda x: x if isinstance(x, str) else "")
def test_update_geometry_matches_tyssue(mesh, tissue, geom):
    """Rust ``update_geometry`` == ``SheetGeometry.update_all`` output columns.

    Compares every derived quantity (edge vectors, lengths, face centroids,
    r-vectors, edge normals, sub-areas, face areas, perimeters) against the
    values tyssue itself wrote into edge_df/face_df. atol=1e-10 is comfortably
    above the observed ~1e-15 to absorb platform FP variance.
    """
    k = pytest.importorskip("tyssue_kernels", reason="Rust kernels not built")
    eptm = _load_eptm(mesh, tissue, geom)
    coords = eptm.coords
    dim = len(coords)
    pos, srce, trgt, _ = _positional_edges(eptm)
    fmap = {v: i for i, v in enumerate(eptm.face_df.index)}
    face = np.ascontiguousarray(eptm.edge_df["face"].map(fmap).values, dtype=np.uint32)

    g = k.update_geometry(pos, srce, trgt, face, eptm.Nf)

    def close(name, got, ref):
        got = np.asarray(got).ravel()
        ref = np.asarray(ref, dtype=np.float64).ravel()
        assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
        assert np.allclose(got, ref, atol=1e-10, rtol=0.0), (
            f"{mesh} {name}: max|Δ|={np.max(np.abs(got - ref)):.3e}"
        )

    close("length", g["length"], eptm.edge_df["length"].values)
    close("sub_area", g["sub_area"], eptm.edge_df["sub_area"].values)
    close("area", g["area"], eptm.face_df["area"].values)
    close("perimeter", g["perimeter"], eptm.face_df["perimeter"].values)
    close("dcoords", g["dcoords"], eptm.edge_df[["d" + c for c in coords]].values)
    close("rcoords", g["rcoords"], eptm.edge_df[["r" + c for c in coords]].values)
    close("normals", g["normals"], eptm.edge_df[eptm.ncoords].values)
    close("centroid", g["centroid"], eptm.face_df[coords].values)


# The fused gradient kernel targets the standard 3-effector sheet model
# (LineTension + PerimeterElasticity + FaceAreaElasticity). model_factory_bound
# additionally zeros the gradient at boundary vertices; model_factory does not —
# the caller supplies the boundary mask accordingly (as the EulerSolver wiring
# will). base_solver adds VesselSurfaceElasticity, so it's out of scope here.
GRADIENT_COMPOSITES = ["stochastic", "anisotropic"]


def _build_model_eptm(composite):
    """Build eptm + geometry + model from a composite spec, as EulerSolver does."""
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    import sys

    import yaml

    sys.path.insert(0, str(ROOT))
    from tyssue.io.hdf5 import load_datasets
    from vivarium_tyssue.maps import (
        EFFECTORS_MAP,
        FACTORY_MAP,
        GEOMETRY_MAP,
        TISSUE_MAP,
    )

    spec = yaml.safe_load(
        (ROOT / "vivarium_tyssue" / "composites" / f"{composite}.composite.yaml").read_text(encoding="utf-8")
    )
    cfg = spec["state"]["Tyssue"]["config"]
    path = ROOT / cfg["eptm"]
    if not path.exists():
        pytest.skip(f"mesh fixture missing: {cfg['eptm']}")
    eptm = TISSUE_MAP[cfg["tissue_type"]]("e", load_datasets(str(path)))
    geom = GEOMETRY_MAP[cfg["geom"]]
    effs = [EFFECTORS_MAP[e] for e in cfg["effectors"]]
    model = FACTORY_MAP[cfg["factory"]](effs, EFFECTORS_MAP[cfg["ref_effector"]])
    eptm.update_specs(model.specs, reset=True)
    # a non-uniform line tension makes the gradient non-trivial (as a behavior would)
    eptm.edge_df["line_tension"] = np.random.default_rng(1).uniform(0.5, 1.5, eptm.Ne)
    geom.update_all(eptm)
    is_bound = cfg["factory"] == "model_factory_bound"
    return eptm, model, is_bound


@pytest.mark.parametrize("composite", GRADIENT_COMPOSITES)
def test_sheet_gradient_matches_compute_gradient(composite):
    """Rust ``sheet_gradient`` == tyssue ``model.compute_gradient``, bit-for-bit."""
    k = pytest.importorskip("tyssue_kernels", reason="Rust kernels not built")
    eptm, model, is_bound = _build_model_eptm(composite)
    coords = eptm.coords
    ed, fd = eptm.edge_df, eptm.face_df
    vmap = {v: i for i, v in enumerate(eptm.vert_df.index)}
    fmap = {v: i for i, v in enumerate(fd.index)}
    C = lambda a, dt: np.ascontiguousarray(a, dtype=dt)  # noqa: E731

    r_aj = ed[["t" + c for c in coords]].values - ed[["f" + c for c in coords]].values
    boundary = (
        eptm.vert_df["boundary"].values.astype(np.uint8)
        if is_bound and "boundary" in eptm.vert_df.columns
        else np.zeros(eptm.Nv, dtype=np.uint8)
    )
    got = np.asarray(
        k.sheet_gradient(
            C(ed[["u" + c for c in coords]].values, np.float64),
            C(ed[eptm.ncoords].values, np.float64),
            C(ed["sub_area"].values, np.float64),
            C(ed[["r" + c for c in coords]].values, np.float64),
            C(r_aj, np.float64),
            C(ed["srce"].map(vmap).values, np.uint32),
            C(ed["trgt"].map(vmap).values, np.uint32),
            C(ed["face"].map(fmap).values, np.uint32),
            C((ed["line_tension"] * ed["is_active"]).values, np.float64),
            C((fd["perimeter_elasticity"] * fd["is_alive"] * (fd["perimeter"] - fd["prefered_perimeter"])).values, np.float64),
            C((fd["area_elasticity"] * fd["is_alive"] * (fd["area"] - fd["prefered_area"])).values, np.float64),
            C(boundary, np.uint8),
            eptm.Nv,
            float(eptm.specs["settings"].get("nrj_norm_factor", 1.0)),
        )
    ).reshape(eptm.Nv, 3)

    ref = np.asarray(model.compute_gradient(eptm)).astype(np.float64)
    assert np.allclose(got, ref, atol=1e-10, rtol=0.0), (
        f"{composite} ({'bound' if is_bound else 'plain'}): max|Δ|={np.max(np.abs(got - ref)):.3e}"
    )
