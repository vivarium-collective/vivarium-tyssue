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
