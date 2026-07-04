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
    """The shared ``rust_sheet_gradient`` helper (used by EulerSolver) ==
    tyssue ``model.compute_gradient``, bit-for-bit. Testing the helper rather
    than the raw kernel keeps this in lockstep with what the process runs."""
    pytest.importorskip("tyssue_kernels", reason="Rust kernels not built")
    import sys

    sys.path.insert(0, str(ROOT))
    from vivarium_tyssue.processes.utils import rust_sheet_gradient

    eptm, model, is_bound = _build_model_eptm(composite)
    got = rust_sheet_gradient(eptm, is_bound)
    ref = np.asarray(model.compute_gradient(eptm)).astype(np.float64)
    assert np.allclose(got, ref, atol=1e-10, rtol=0.0), (
        f"{composite} ({'bound' if is_bound else 'plain'}): max|Δ|={np.max(np.abs(got - ref)):.3e}"
    )


@pytest.mark.parametrize("mesh,tissue,geom", SHEET_MESHES,  # SheetGeometry + VesselGeometry
                         ids=lambda x: x if isinstance(x, str) else "")
def test_rust_geometry_update_matches_update_all(mesh, tissue, geom):
    """rust_geometry_update writes the SAME edge/face/vert columns as
    geom.update_all (bit-for-bit) — a drop-in per-step geometry replacement for
    both SheetGeometry and VesselGeometry."""
    pytest.importorskip("tyssue_kernels", reason="Rust kernels not built")
    import sys

    sys.path.insert(0, str(ROOT))
    from vivarium_tyssue.maps import GEOMETRY_MAP
    from vivarium_tyssue.processes.utils import rust_geometry_update

    geom_cls = GEOMETRY_MAP[geom]
    a = _load_eptm(mesh, tissue, geom)  # deterministic loads -> two identical meshes
    b = _load_eptm(mesh, tissue, geom)
    delta = np.random.default_rng(0).normal(0, 0.02, (a.Nv, len(a.coords)))
    a.vert_df[a.coords] += delta
    b.vert_df[b.coords] += delta

    geom_cls.update_all(a)  # python reference

    vmap = {v: i for i, v in enumerate(b.vert_df.index)}
    fmap = {v: i for i, v in enumerate(b.face_df.index)}
    srce = np.ascontiguousarray(b.edge_df["srce"].map(vmap).values, np.uint32)
    trgt = np.ascontiguousarray(b.edge_df["trgt"].map(vmap).values, np.uint32)
    face = np.ascontiguousarray(b.edge_df["face"].map(fmap).values, np.uint32)
    geom_cls.update_boundary_index(b)  # boundary set once (topology-invariant)
    rust_geometry_update(b, geom_cls, srce, trgt, face)

    for df in ("edge_df", "face_df", "vert_df"):
        da, db = getattr(a, df), getattr(b, df)
        for col in da.columns:
            if da[col].dtype.kind not in "fi":
                continue
            assert col in db.columns, f"rust path missing column {df}.{col}"
            va, vb = da[col].values.astype(float), db[col].values.astype(float)
            # equal_nan: unset parameter columns (e.g. line_tension) are NaN in
            # both and untouched by update_all — that's a match, not a diff.
            assert np.allclose(va, vb, atol=1e-9, rtol=0.0, equal_nan=True), (
                f"{df}.{col}: max|Δ|={np.nanmax(np.abs(va - vb)):.3e}"
            )


def test_vessel_gradient_matches_compute_gradient():
    """rust_sheet_gradient(with_vessel=True) == tyssue compute_gradient for the
    4-effector vessel model (base_solver). Built via the composite so the vessel
    parameters (vessel_elasticity, prefered_radius) are actually set."""
    pytest.importorskip("tyssue_kernels", reason="Rust kernels not built")
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    import sys

    sys.path.insert(0, str(ROOT))
    from pbg_superpowers.composite_spec import build_composite_from_spec, load_spec
    from vivarium_tyssue.core import build_core
    from vivarium_tyssue.processes.utils import rust_sheet_gradient

    spec = load_spec(ROOT / "vivarium_tyssue" / "composites" / "base_solver.composite.yaml")
    spec["emitters"] = []
    comp = build_composite_from_spec(spec, overrides={"interval": 0.001}, core=build_core())
    proc = comp.state["Tyssue"]["instance"]
    assert proc._with_vessel, "base_solver should carry VesselSurfaceElasticity"

    proc.eptm.edge_df["line_tension"] = np.random.default_rng(2).uniform(0.5, 1.5, proc.eptm.Ne)
    proc.geom.update_all(proc.eptm)
    ref = np.asarray(proc.model.compute_gradient(proc.eptm)).astype(np.float64)
    got = rust_sheet_gradient(proc.eptm, proc._is_bound, proc._with_vessel)
    assert np.allclose(got, ref, atol=1e-10, rtol=0.0), f"max|Δ|={np.max(np.abs(got - ref)):.3e}"


# ---------------------------------------------------------------------------
# Backend wiring: the `backend: rust` flag on EulerSolver.
# ---------------------------------------------------------------------------
def test_gradient_supported_gating():
    """The rust backend engages only for the standard 3-effector sheet model."""
    import sys

    sys.path.insert(0, str(ROOT))
    from vivarium_tyssue.processes.utils import gradient_supported, rust_kernels_available

    std = ["LineTension", "FaceAreaElasticity", "PerimeterElasticity"]
    if rust_kernels_available():
        assert gradient_supported(std, "model_factory", 3) is True
        assert gradient_supported(list(reversed(std)), "model_factory_bound", 3) is True
        # the vessel 4-effector set is also supported (radial vertex term)
        assert gradient_supported(std + ["VesselSurfaceElasticity"], "model_factory", 3) is True
    # unsupported: an effector not in the kernel, wrong factory, 2D — never engage
    assert gradient_supported(std + ["FaceContractility"], "model_factory", 3) is False
    assert gradient_supported(std, "model_factory_vessel", 3) is False
    assert gradient_supported(std, "model_factory", 2) is False


def _run_composite_backend(composite, backend, steps, interval=0.1):
    """Build + run a composite on a given backend; return (final positions, process)."""
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    import sys

    sys.path.insert(0, str(ROOT))
    from pbg_superpowers.composite_spec import build_composite_from_spec, load_spec
    from vivarium_tyssue.core import build_core

    core = build_core()
    spec = load_spec(ROOT / "vivarium_tyssue" / "composites" / f"{composite}.composite.yaml")
    spec["state"]["Tyssue"]["config"]["backend"] = backend
    comp = build_composite_from_spec(spec, overrides={"interval": interval}, core=core)
    comp.run(steps)
    proc = comp.state["Tyssue"]["instance"]
    pos = proc.eptm.vert_df[proc.eptm.coords].values.copy()
    return pos, proc


def test_backend_equivalence_anisotropic():
    """End-to-end: python and rust backends trace the same trajectory.

    anisotropic is the one supported composite with no RNG behavior, so two
    independent runs are directly comparable. Confirms the flag actually routes
    (rust engages, python doesn't) and the trajectories agree after several steps.
    """
    pytest.importorskip("tyssue_kernels", reason="Rust kernels not built")
    py_pos, py_proc = _run_composite_backend("anisotropic", "python", steps=5)
    ru_pos, ru_proc = _run_composite_backend("anisotropic", "rust", steps=5)

    assert py_proc._rust_gradient is False
    assert ru_proc._rust_gradient is True
    assert py_pos.shape == ru_pos.shape
    assert np.allclose(py_pos, ru_pos, atol=1e-8, rtol=0.0), (
        f"backends diverged after 5 steps: max|Δ|={np.max(np.abs(py_pos - ru_pos)):.3e}"
    )


@pytest.mark.parametrize("composite", ["stochastic", "anisotropic", "jamming", "gradient", "base_solver"])
def test_supported_composites_run_on_rust(composite):
    """Every rust-supported composite advances a step with the rust backend engaged."""
    pytest.importorskip("tyssue_kernels", reason="Rust kernels not built")
    _, proc = _run_composite_backend(composite, "rust", steps=1, interval=0.001)
    # base_solver (vessel) engages geometry + the vessel gradient; sheets both.
    assert proc._rust_gradient is True, f"{composite} did not engage the rust gradient"
    assert proc._rust_geometry is True, f"{composite} did not engage rust geometry"


def _build_proc(composite, substeps, interval):
    import sys
    sys.path.insert(0, str(ROOT))
    from pbg_superpowers.composite_spec import build_composite_from_spec, load_spec
    from vivarium_tyssue.core import build_core
    spec = load_spec(ROOT / "vivarium_tyssue" / "composites" / f"{composite}.composite.yaml")
    spec["emitters"] = []
    spec["state"]["Tyssue"]["config"]["backend"] = "rust"
    spec["state"]["Tyssue"]["config"]["substeps"] = substeps
    comp = build_composite_from_spec(spec, overrides={"interval": interval}, core=build_core())
    return comp.state["Tyssue"]["instance"]


def test_native_substeps_match_single_steps():
    """Phase B: N native substeps in one update(interval) == N single-step updates
    at dt=interval/N, sampled at the end — the DataFrames are just materialized
    once instead of per step. Bit-identical (machine eps), not merely close."""
    pytest.importorskip("tyssue_kernels", reason="Rust kernels not built")
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    import contextlib, io

    N, T = 10, 0.1
    fine = _build_proc("anisotropic", substeps=1, interval=T / N)
    coarse = _build_proc("anisotropic", substeps=N, interval=T)
    assert coarse._native_substeps is True and coarse._substeps == N

    with contextlib.redirect_stdout(io.StringIO()):
        for i in range(N):
            fine.update({"behaviors": [], "global_time": i * T / N}, T / N)
        coarse.update({"behaviors": [], "global_time": 0.0}, T)

    for frame, col in (("vert_df", None), ("face_df", "area"), ("edge_df", "length")):
        a = getattr(fine.eptm, frame)
        b = getattr(coarse.eptm, frame)
        va = a[fine.eptm.coords].values if col is None else a[col].values
        vb = b[coarse.eptm.coords].values if col is None else b[col].values
        assert np.allclose(va, vb, atol=1e-11, rtol=0.0), (
            f"{frame}{'' if col is None else '.'+col} diverged: "
            f"max|Δ|={np.max(np.abs(va - vb)):.3e}"
        )


def test_to_dataframes_materializes_and_returns_eptm():
    """The public converter returns the epithelium with geometry frames current."""
    pytest.importorskip("tyssue_kernels", reason="Rust kernels not built")
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    import contextlib, io

    proc = _build_proc("anisotropic", substeps=5, interval=0.05)
    with contextlib.redirect_stdout(io.StringIO()):
        proc.update({"behaviors": [], "global_time": 0.0}, 0.05)
    eptm = proc.to_dataframes()
    assert eptm is proc.eptm
    # face area equals the native stash it was materialized from
    assert np.allclose(eptm.face_df["area"].values, proc._geom_stash["area"], atol=0, rtol=0)
