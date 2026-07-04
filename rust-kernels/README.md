# tyssue-kernels — Phase-A Rust hot-loop backend

Rust/pyo3 numeric kernels for the tyssue vertex-model inner loop. This is
**Phase A** of the tyssue backend speedup: drop-in accelerated kernels behind
the *identical* `EulerSolver` process-bigraph interface, so every existing
composite/demo keeps working. Phase B grows this crate into a native Rust mesh
core that owns state across steps and materializes pandas DataFrames only at
emit time.

## Why

Profiling showed per-step cost is ~flat across cell count (~7 ms whether 100 or
1600 cells) — dominated by pandas per-call overhead in `geom.update_all()` and
`model.compute_gradient()`, not by float math or algorithmic complexity. These
kernels move that work to compiled Rust over contiguous arrays.

## Layout

```
crates/tyssue-core/   pure Rust kernels (no Python) + unit tests
crates/tyssue-py/     pyo3 bindings -> the `tyssue_kernels` Python module
pyproject.toml        maturin build backend
```

## Build

Requires the Rust toolchain (`cargo`) and `maturin` (installed in the workspace
`.venv`). From this directory:

```bash
VIRTUAL_ENV=../.venv ../.venv/bin/maturin develop --release
```

Then `import tyssue_kernels` works in the venv. The module is **optional**: the
equivalence tests (`tests/test_rust_kernels_equiv.py`) `importorskip` it, so CI
without Rust stays green.

## Correctness contract

Every kernel must reproduce the tyssue/Python reference to `atol=1e-12` on the
real demo meshes (flat sheet, vessel cylinder, monolayer). That gate lives in
`../tests/test_rust_kernels_equiv.py` and must pass **before** any kernel is
wired into `EulerSolver`. Add a new equivalence test alongside each new kernel.

## Kernels

| Function | Replaces | Status |
|---|---|---|
| `edge_lengths(pos, srce, trgt)` | tyssue `update_length` | ✅ proven (1e-12) |
| `scatter_add(values, index, n_vert)` | the two `groupby(...).sum()` in `compute_gradient` (edge→vertex assembly) | ✅ proven (1e-12) |
| `update_geometry(pos, srce, trgt, face, n_face)` | `SheetGeometry.update_all` stateless core: dcoords/length/centroid/normals/area/perimeter | ✅ proven (~1e-15, **~20×**; Sheet + Vessel — see note) |
| `assemble_gradient(...)` | per-effector kernels + assembly | ⏳ next |

**Geometry-kernel scope:** `update_geometry` implements the `SheetGeometry`
formula, which `VesselGeometry` inherits unchanged (both match to ~1e-15). It
excludes `update_ucoords` (divides by the *stale* previous-step length —
stateful) and `update_vol` (geometry-specific vertex heights); those stay in
Python. `MonolayerGeometry`/`BulkGeometry` redefine centroid/area/normals and
need their own kernel — the equivalence test excludes them on purpose.

`srce`/`trgt`/`index` are **positional** vertex indices (0..Nv-1), remapped from
tyssue's DataFrame index by the caller; positions are C-contiguous `(Nv, dim)`
float64.
