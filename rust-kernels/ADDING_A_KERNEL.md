# Adding a kernel — new effector or geometry

The rust backend is **compositional**: `EulerSolver` assembles a model's gradient
one effector at a time (`vivarium_tyssue/processes/kernels.py:rust_model_gradient`).
Each effector runs through a rust *primitive* if one is registered, otherwise it
falls back to its own tyssue `.gradient(eptm)`. So **coverage is additive**:
nothing you add can break an existing model, and an unregistered effector simply
runs in Python. You never edit a monolithic fused kernel.

The shared gradient primitives already in the crate:

| primitive (`tyssue_kernels`) | covers |
|---|---|
| `unit_edge_gradient(ucoords, coeff)` | LineTension, PerimeterElasticity, FaceContractility, LengthElasticity, BorderElasticity — anything of the form `grad_srce = -u·c`, `grad_trgt = +u·c` |
| `area_gradient(normals, r_ak, r_aj, sub_area, coeff)` (3D) / `area_gradient_2d(nz, …)` (2D) | FaceAreaElasticity, SurfaceTension, CellAreaElasticity |
| `scatter_add(values, index, n_vert)` | edge→vertex assembly (used once, at the end) |

Geometry base kernels: `update_geometry` (3D sheet), `update_geometry_bulk` (3D
monolayer/bulk), `update_geometry_planar` (2D planar).

---

## Add an effector

### Case A — it reuses an existing primitive (no Rust)

1. Write an adapter in `vivarium_tyssue/processes/kernels.py` returning
   `(grad_srce, grad_trgt, grad_vert)` — assemble the per-edge/-face **coefficient**
   in numpy (matching the effector's tyssue `gradient`), then call the primitive.
   Follow `_adapt_face_contractility` / `_adapt_surface_tension`.
2. Register it: add `"YourEffector": _adapt_your_effector` to `EFFECTOR_KERNELS`.
3. Add an equivalence case to `tests/test_rust_kernels_equiv.py::COVERED_SHEET`
   (with a `setup` callback if its gradient needs extra columns). The generic
   harness then checks it against `compute_gradient` to `atol=1e-9` for both
   factories.

### Case B — it needs new math (new Rust primitive)

1. Add a pure function to `rust-kernels/crates/tyssue-core/src/lib.rs` returning
   the edge/vertex gradient arrays, with an in-crate `#[test]`.
2. Expose it in `rust-kernels/crates/tyssue-py/src/lib.rs` (zero-copy
   `PyReadonlyArray` in, `PyDict`/`PyArray` out) and add it to the `#[pymodule]`.
3. `cd rust-kernels && conda run -n vivarium-tyssue maturin develop --release`.
4. Then do Case A steps 1–3 with an adapter that calls your new primitive.

> Vertex-element effectors (BarrierElasticity, VesselSurfaceElasticity, …) and
> other cheap `O(Nv)` terms are usually best left on the per-term Python fallback
> — they aren't the hot path. Register them only if profiling says so.

## Add a geometry

1. If its `update_all` core matches an existing base kernel (sheet/bulk/planar),
   just make sure `EulerSolver.set_pos` routes it (see `utils.geometry_supported`
   and the `_bulk_geometry` branch) and replay any cheap geometry-specific vertex
   steps in Python (as `rust_geometry_update` does for VesselGeometry's tangents).
2. If it redefines centroid/area/normals, add a new `update_geometry_*` kernel
   (Case B, steps 1–3) and an equivalence test against `geom.update_all`
   (follow `test_update_geometry_planar_matches_tyssue`).

## Build & verify

```bash
cd rust-kernels && conda run -n vivarium-tyssue maturin develop --release   # (.venv is broken; use the conda env)
conda run -n vivarium-tyssue python -m pytest tests/test_rust_kernels_equiv.py -q
```

The equivalence gate (`atol=1e-12`–`1e-9` on the real demo meshes) is the
correctness contract — a new kernel must pass it before it ships.
