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

Requires the Rust toolchain (`cargo`, via `rustup`) and `maturin`. Build into the
**`vivarium-tyssue` conda env** — that's the env the tests and notebooks run in
(the repo `.venv` is broken). From this directory:

```bash
conda run -n vivarium-tyssue pip install "maturin>=1.5,<2"   # once
cd rust-kernels && conda run -n vivarium-tyssue maturin develop --release
```

Then `import tyssue_kernels` works in that env. The module is **optional**: the
equivalence tests (`tests/test_rust_kernels_equiv.py`) `importorskip` it, so CI
without Rust stays green. The extension is tied to the interpreter it was built
against (no abi3) — rebuild if the env's Python changes.

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
| `sheet_gradient(...)` | all of `compute_gradient` for the standard 3-effector sheet model (LineTension + PerimeterElasticity + FaceAreaElasticity) incl. edge→vertex assembly — the fast fused special case | ✅ proven (~1e-15; both factories — see note) |
| `update_geometry_planar(...)` | 2D `PlanarGeometry.update_all` core (signed `nz`, `sub_area = nz/2`) | ✅ proven (~1e-10) |
| `unit_edge_gradient` / `area_gradient` / `area_gradient_2d` | shared gradient **primitives** for the compositional path (see below) — length/tension family and area family | ✅ proven (~1e-9 vs `compute_gradient`) |

## Compositional gradient (all effectors + geometries)

Beyond the fused `sheet_gradient`, the backend has an **extensible compositional
path**: `vivarium_tyssue/processes/kernels.py:rust_model_gradient` assembles a
model's gradient one effector at a time — covered effectors run through the shared
primitives above, everything else falls back to its own tyssue `.gradient`. So any
effector/geometry combination runs on the rust backend (covered terms accelerated,
nothing regressing), and adding coverage for a new effector or geometry is a small,
tested, local change. See **`ADDING_A_KERNEL.md`**.

**Gradient-kernel scope:** `sheet_gradient` fuses the three standard sheet
effectors and the two `groupby.sum()` reductions into one pass, consuming
tyssue's geometry columns as-is (so it reproduces even the stale-length
`ucoords` quirk). It matches both `model_factory` and `model_factory_bound` to
~1e-15 — the caller supplies the boundary mask (bound factory zeros boundary
vertices; plain does not). Models with other effectors (e.g. base_solver's
`VesselSurfaceElasticity`) fall back to Python. The ~2.4× measured is limited by
marshalling ~10 pandas columns per call; in Phase B those inputs live in Rust
and the pure kernel is far faster.

**Geometry-kernel scope:** `update_geometry` implements the `SheetGeometry`
formula, which `VesselGeometry` inherits unchanged (both match to ~1e-15). It
excludes `update_ucoords` (divides by the *stale* previous-step length —
stateful) and `update_vol` (geometry-specific vertex heights); those stay in
Python. `MonolayerGeometry`/`BulkGeometry` redefine centroid/area/normals and
need their own kernel — the equivalence test excludes them on purpose.

`srce`/`trgt`/`index` are **positional** vertex indices (0..Nv-1), remapped from
tyssue's DataFrame index by the caller; positions are C-contiguous `(Nv, dim)`
float64.
