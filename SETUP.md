# Environment setup — vivarium-tyssue + the `vivatyssue` tyssue fork

This repo does **not** run on PyPI `tyssue`. It runs on the vivarium-collective
**fork** (`vivatyssue`), plus several vivarium-collective packages that are not on
PyPI. This document is the from-scratch recipe for building a working environment with
**conda**, with a **stdlib `venv`**, or with **uv**.

> **If you are an automated agent:** read *§0 Rules* first. If you have `uv`, *§2.0* is
> a two-command install and you are done. Otherwise execute one of the §2 variants plus
> *§3* **verbatim** — do not substitute versions, and never run a bare
> `pip install -e .` (*§6.2*). The whole thing is also available as one copy-paste
> script in *§8*.

**`tyssue` is in the optional `sim` extra.** `.[dev]` gives you a backend-free
environment — `tests/conftest.py` skips every simulator test module — which is what CI
installs. Anything that actually simulates needs **`.[dev,sim]`**.

Validated on 2026-09-02 (macOS 15, arm64) by building environments from scratch:
`.[dev,sim]` → **124 passed, 45 skipped, 1 xfailed** (the 45 skips are all
`tests/test_rust_kernels_equiv.py`, which needs the optional Rust kernels from *§5.1*).

---

## 0. Rules

1. **Python 3.12.** The reference environment is 3.12.11; validation used 3.12.10.
   `pyproject.toml` says `>=3.12`, but the pin set below has only been checked on 3.12.
2. **The dependency names are `viva-*`.** The vivarium-collective packages were
   renamed (`pbg-superpowers` → `viva-superpowers`, `pbg-copasi` → `viva-copasi`,
   `pbg-emitters` → `viva-emitters`, `vivarium-dashboard` → `vivarium-workbench`), and
   the Python modules the code imports were renamed with them (`viva_superpowers`,
   `viva_emitters`, …). `pyproject.toml` is on that generation. Do not reintroduce a
   `pbg-*` name — see *§6.1* for the failure it produces.
3. **On the pip paths (conda / stdlib venv), install the git dependencies by URL.** pip
   does not read `[tool.uv.sources]`, so it cannot see them at all; those packages are
   not on PyPI and the install simply fails. On the uv path (*§2.0*) none of this
   applies — uv reads the sources itself.
4. **On the pip paths, install this repo with `--no-deps`.** pip would otherwise try to
   resolve `tyssue` from PyPI and get the wrong package (see *§4*). The repo does not
   need to be installed at all for the tests or the dashboard (both put the workspace
   root on `sys.path`), but an editable `--no-deps` install makes
   `import vivarium_tyssue` work from any directory.
5. **Never edit `vivarium_tyssue/processes/eulersolver.py`** — see `CLAUDE.md`. A
   pre-commit hook blocks agent commits that touch it.

---

## 1. Prerequisites

| Need | Why | Install (macOS) |
|---|---|---|
| **git** | everything is installed from git | preinstalled / `brew install git` |
| **Python 3.12** | the pin set | `conda`, `uv python install 3.12`, or `brew install python@3.12` |
| **ImageMagick** (`magick` on PATH) | `tyssue.draw.create_gif` shells out to it; every experiment notebook renders GIFs | `brew install imagemagick` |
| **Rust toolchain** (optional) | the `tyssue_kernels` hot-loop backend (*§5.1*) | `brew install rustup` then `rustup default stable` |

One-time git config — **required**, not optional:

```bash
git config --global url."https://github.com/".insteadOf "git@github.com:"
```

The fork declares a docs-only submodule (`doc/notebooks`) with an SSH URL. Any clone
or pip-from-git of the fork initialises it and fails with *"Host key verification
failed"* without this rewrite. The repo's own `Dockerfile` and CI workflow apply the
same line.

---

## 2. Step 1 — create the interpreter

### 2.0. The uv fast path (if you have uv, start here)

`uv` reads `[tool.uv.sources]` from `pyproject.toml`, so the git sources are applied
for you and the whole install is two commands:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev,sim]"     # drop `sim` for a backend-free env (what CI installs)
```

Then skip to *§5* (optional components) and *§7* (verify) — §3 is only for pip-based
environments. It resolves to a newer science stack than the reference conda env
(numpy 2.5 / pandas 3.0 rather than 1.26 / 2.3); the suite passes on both.

`uv sync --no-install-project` also works and is what the Dockerfile uses.

### 2.1–2.3. pip-based environments

Neither conda nor a stdlib `venv` reads `[tool.uv.sources]`, so with those the git
sources have to be spelled out — pick **one** interpreter variant below, then do *§3*.
Each variant ends by defining `$PY`, the interpreter every later step uses.

### 2.1. conda (this is the reference environment)

```bash
conda create -y -n vivarium-tyssue python=3.12
PY="$(conda run -n vivarium-tyssue python -c 'import sys; print(sys.executable)')"
```

Interactive use: `conda activate vivarium-tyssue`. Non-interactive / agent use:
`conda run -n vivarium-tyssue <cmd>`.

### 2.2. stdlib venv

```bash
python3.12 -m venv .venv-tyssue
PY="$PWD/.venv-tyssue/bin/python"
"$PY" -m pip install -U pip
```

Do **not** reuse the `.venv/` checked into the repo — it is stale (Python 3.13, no
packages) and `.gitignore`d.

### 2.3. uv, but driven with pip

```bash
uv venv --seed --python 3.12 .venv-tyssue      # --seed installs pip into the venv
PY="$PWD/.venv-tyssue/bin/python"
```

Only useful if you want uv's fast interpreter provisioning but the explicit,
resolver-independent install of §3. `--seed` matters: a bare `uv venv` has no `pip`.
For normal use prefer *§2.0*.

---

## 3. Step 2 — install, in this order

Order is load-bearing. Read the comments before reordering anything.

```bash
# --- 3.1 the vivarium-collective packages (none are on PyPI) ------------------------
# ALL EIGHT are required. viva-superpowers pulls viva-workspace, viva-marketplace and
# investigation-contracts by bare name, and vivarium-workbench pulls
# pbg-basic-processes the same way — pip cannot resolve any of them without a URL, and
# omitting one fails with e.g. "No matching distribution found for viva-marketplace".
# (uv gets them from [tool.uv.sources] transitively, which is why §2.0 needs none of this.)
"$PY" -m pip install \
  "viva-emitters[parquet] @ git+https://github.com/vivarium-collective/viva-emitters.git" \
  "viva-superpowers @ git+https://github.com/vivarium-collective/viva-superpowers.git" \
  "viva-workspace @ git+https://github.com/vivarium-collective/viva-workspace.git" \
  "viva-marketplace @ git+https://github.com/vivarium-collective/viva-marketplace.git" \
  "viva-copasi @ git+https://github.com/vivarium-collective/viva-copasi.git" \
  "vivarium-workbench @ git+https://github.com/vivarium-collective/vivarium-dashboard.git" \
  "investigation-contracts @ git+https://github.com/vivarium-collective/investigation-contracts.git" \
  "pbg-basic-processes @ git+https://github.com/vivarium-collective/pbg-basic-processes.git"

# --- 3.2 the science stack ----------------------------------------------------------
# scikit-image/vispy/quantities/tables/ipywidgets are tyssue's undeclared transitive
# needs — the fork's pyproject declares NO runtime dependencies at all.
# NOTE: do NOT pin numpy below 2. The viva-* stack (viva-emitters -> zarr 3.x /
# numcodecs) requires numpy>=2, and pinning 1.26 makes the resolution impossible:
#   ERROR: Cannot install numpy==1.26.4, viva-emitters[parquet]==0.3.0, … conflicting
#   dependencies. ResolutionImpossible
# numpy 2.5 / pandas 2.3 is the validated combination.
# quantities MUST be >=0.16.4. 0.16.2 does `@with_doc(np.ndarray.ptp)` at import time
# and `ndarray.ptp` was removed in numpy 2.0, so it dies with
#   AttributeError: type object 'numpy.ndarray' has no attribute 'ptp'
# during test collection. A fresh install gets a new enough one; an environment being
# upgraded from the numpy-1 era does not, which is exactly how it bites.
"$PY" -m pip install \
  "pandas==2.3.3" scipy matplotlib plotly \
  scikit-image vispy "quantities>=0.16.4" tables ipywidgets \
  pyyaml "jsonschema[format-nongpl]" jinja2 pypdf bigraph-viz pytest

# --- 3.3 the tyssue fork, editable ---------------------------------------------------
# Branch `intersections` is REQUIRED — see §4.  Clone wherever you keep sources.
git clone --branch intersections https://github.com/vivarium-collective/vivatyssue.git \
  ~/PycharmProjects/vivatyssue          # skip if already cloned
"$PY" -m pip install -e ~/PycharmProjects/vivatyssue

# --- 3.4 this repo, WITHOUT its dependencies ------------------------------------------
cd /path/to/vivarium-tyssue
"$PY" -m pip install -e . --no-deps
```

---

## 4. Why the fork, and why branch `intersections`

`pyproject.toml` points `tyssue` at `https://github.com/vivarium-collective/vivatyssue.git`.
Stock PyPI `tyssue==1.1.0` is missing symbols this workspace imports by name:

| Fork-only symbol | Used by |
|---|---|
| `VesselGeometry`, `CylinderGeometryInit`, `RNRGeometry` | `vivarium_tyssue/core_maps.py` — the cylinder/crypt composites |
| `model_factory_vessel` / `_cylinder` / `_bound` | `vivarium_tyssue/maps/factory_maps.py` |
| `VesselSurfaceElasticity` | `vivarium_tyssue/maps/effectors_maps.py` |
| `tyssue.collisions.intersections.Result` | `tests/test_composites.py`, `EulerSolver`'s `check_intersections` |

The maps import defensively, so a stock-tyssue install still *imports* — it just fails
at run time with a `KeyError` on every composite except `anisotropic`.

Branch choice: `VesselGeometry` and the factories exist on `main`, `updates` and
`intersections`; **`tyssue.collisions.intersections` exists only on `intersections`**
(that branch also drops the CGAL dependency and makes `History` self-sufficient).
`pyproject.toml` pins the source to `7b5dedcce` on that branch — the fork's default
branch resolves *without* the collisions module, and
`test_intersection_check_arms_the_behaviour` then fails on import.

---

## 5. Optional components

### 5.1 Rust hot-loop kernels (`tyssue_kernels`)

Unlocks `backend="rust"` on `EulerSolver` and the 45 skipped equivalence tests.

```bash
export PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH"  # if cargo isn't already on PATH
"$PY" -m pip install "maturin>=1.5,<2"
cd rust-kernels
# conda:  conda run -n vivarium-tyssue maturin develop --release
# venv/uv: activate the venv first, then:
maturin develop --release
```

Verify with `PYTHONUTF8=1 "$PY" -m pytest tests -q` — the 45 skips become passes
(169 passed, 1 xfailed).

The extension binds to the exact interpreter it was built against (no abi3) — rebuild
after any Python change. If maturin exits with *"Both VIRTUAL_ENV and CONDA_PREFIX are
set"*, unset the one you are not targeting: `env -u CONDA_PREFIX maturin develop
--release` (venv target) or `env -u VIRTUAL_ENV …` (conda target).

### 5.2 Faithful 3-D crypt animation

```bash
"$PY" -m pip install ipyvolume pythreejs
```

Without it `TissueCryptGif3D` falls back to a matplotlib edge-mesh animation.

### 5.3 Human Reference Atlas loaders (`vivarium_tyssue/hra`)

```bash
"$PY" -m pip install trimesh fast-simplification
```

### 5.4 Notebooks / dashboard extras

```bash
"$PY" -m pip install jupyterlab notebook ipython requests fire
```

`requests` and `fire` silence the `process_bigraph.protocols` /
`process_bigraph.server.start` skip messages printed by `build_core()`; they are
cosmetic unless you use those subsystems.

---

## 6. Pins, renames and traps (read before "fixing" anything)

### 6.1 The `pbg-*` → `viva-*` rename

The vivarium-collective packages were renamed, distributions *and* Python modules:

| old | new |
|---|---|
| `pbg-superpowers` / `pbg_superpowers` | `viva-superpowers` / `viva_superpowers` |
| `pbg-copasi` / `pbg_copasi` | `viva-copasi` / `viva_copasi` |
| `pbg-emitters` / `pbg_emitters` | `viva-emitters` / `viva_emitters` |
| `pbg-biomodels` | `viva-biomodels` |
| `vivarium-dashboard` | `vivarium-workbench` |

`pyproject.toml` and the code are on the new names. Asking for an old *distribution*
name now fails, because a resolver checks the name it requested against the name the
package publishes:

```
× Failed to download and build `pbg-superpowers @ git+https://github.com/vivarium-collective/pbg-superpowers.git`
╰─▶ Package metadata name `viva-superpowers` does not match given name `pbg-superpowers`
```

The old *module* names still import — they are deprecation shims that warn and are
slated for removal — so a stale `from pbg_superpowers…` looks fine until it suddenly
is not. Don't add new ones.

One straggler, inherited from `main`: `scripts/fetch_tumor_biomodel.py` still does
`from pbg_biomodels.run_biomodels import load_biomodel` while the `fetch` extra
installs `viva-biomodels`. Nothing in CI runs it and the model it fetches is committed
at `workspace/datasets/BIOMD0000000903.xml`, so it is latent rather than urgent.

### 6.2 Plain `pip install -e .` (no `--no-deps`) — will not resolve

pip ignores `[tool.uv.sources]`, so it falls back to PyPI for the bare names and gets
`tyssue` 1.1.0 (the wrong package, see *§4*) and nothing at all for the
vivarium-collective packages. Either use uv (*§2.0*), which reads the sources, or the
explicit pip order in *§3*.

### 6.3 The dashboard CLI name

The dashboard's CLI is **`vivarium-dashboard`** on the old generation and
**`vivarium-workbench`** on the current one. `scripts/serve.sh` tries both (current
name first); the Dockerfile `CMD` uses `vivarium-workbench`. To run it directly:

```bash
cd /path/to/vivarium-tyssue
PYTHONUTF8=1 conda run -n vivarium-tyssue vivarium-workbench serve --workspace .
```

`PYTHONUTF8=1` is required: COPASI/basico resets `LC_CTYPE` to `C` on import, which
makes the YAML reads crash on the non-ASCII characters in generated composite specs.

### 6.4 The repo's checked-in `.venv/`

Empty and stale (Python 3.13, zero packages) — `.gitignore`d, and not what any of the
recipes here build. Recreate it with *§2.0* rather than reusing it.

---

## 7. Verify

```bash
# a. the stack imports and the fork's geometries are registered
"$PY" - <<'PY'
import numpy, pandas, tyssue
from vivarium_tyssue.core import build_core
from vivarium_tyssue.core_maps import GEOMETRY_MAP
build_core()
print("numpy", numpy.__version__, "| pandas", pandas.__version__)
print("VesselGeometry present:", "VesselGeometry" in GEOMETRY_MAP)
assert "VesselGeometry" in GEOMETRY_MAP, "stock tyssue installed — see §4"
PY

# b. the test suite (run from the repo root)
cd /path/to/vivarium-tyssue && PYTHONUTF8=1 "$PY" -m pytest tests -q
```

Expected:

```
numpy 2.5.2 | pandas 2.3.3
VesselGeometry present: True
124 passed, 45 skipped, 1 xfailed
```

Add a third check that the environment matches its extras: with `.[dev]` alone (no
`sim`), `pytest tests -q` gives **8 passed** — `tests/conftest.py` skips every module
that imports the backend. If you instead get `ModuleNotFoundError: No module named
'tyssue'` at collection, the conftest skip logic is not doing its job.

45 skips = the Rust-kernel equivalence tests; with *§5.1* built the same suite gives
**169 passed, 1 xfailed** (verified in both a fresh venv and the reference conda env;
it takes ~2.5 min). `tests/tests.py` collects nothing by design — don't chase it. Two warnings are normal:
`backend='rust' requested but … tyssue_kernels … not importable` (until *§5.1*) and
tyssue's `Duplicated (srce, trgt) values detected`.

---

## 8. One-shot script

Copy-paste, or hand to an agent. Edit the two paths at the top; nothing else.

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO="$HOME/PycharmProjects/vivarium-tyssue"     # this repo
FORK="$HOME/PycharmProjects/vivatyssue"          # where to clone the tyssue fork
ENV_NAME="vivarium-tyssue"                       # conda env name

git config --global url."https://github.com/".insteadOf "git@github.com:"

conda create -y -n "$ENV_NAME" python=3.12
PY="$(conda run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"

"$PY" -m pip install \
  "viva-emitters[parquet] @ git+https://github.com/vivarium-collective/viva-emitters.git" \
  "viva-superpowers @ git+https://github.com/vivarium-collective/viva-superpowers.git" \
  "viva-workspace @ git+https://github.com/vivarium-collective/viva-workspace.git" \
  "viva-marketplace @ git+https://github.com/vivarium-collective/viva-marketplace.git" \
  "viva-copasi @ git+https://github.com/vivarium-collective/viva-copasi.git" \
  "vivarium-workbench @ git+https://github.com/vivarium-collective/vivarium-dashboard.git" \
  "investigation-contracts @ git+https://github.com/vivarium-collective/investigation-contracts.git" \
  "pbg-basic-processes @ git+https://github.com/vivarium-collective/pbg-basic-processes.git"

"$PY" -m pip install \
  "pandas==2.3.3" scipy matplotlib plotly \
  scikit-image vispy "quantities>=0.16.4" tables ipywidgets \
  pyyaml "jsonschema[format-nongpl]" jinja2 pypdf bigraph-viz pytest

[ -d "$FORK" ] || git clone --branch intersections \
  https://github.com/vivarium-collective/vivatyssue.git "$FORK"
"$PY" -m pip install -e "$FORK"

cd "$REPO"
"$PY" -m pip install -e . --no-deps

PYTHONUTF8=1 "$PY" -m pytest tests -q
```

For a venv instead of conda, replace the two conda lines with §2.2 or §2.3 and keep
everything else.

---

## 9. Reference: known-good versions

The environment this was validated against. Re-checked 2026-09-02 — the §7 verify and
the full suite still give the expected output; the loose PyPI entries had drifted since
first writing and are refreshed here.

| Package | Version | Source |
|---|---|---|
| python | 3.12.11 (ref) / 3.12.10 (validated) | conda / uv |
| tyssue | `0.1.dev*` @ `vivatyssue@intersections` (`7b5dedcce`) | editable clone |
| numpy | 2.5.2 | PyPI |
| pandas | 2.3.3 | PyPI (pinned) |
| scipy | 1.16.3 | PyPI |
| matplotlib | 3.11.0 | PyPI |
| scikit-image | 0.26.0 | PyPI |
| vispy | 0.16.0 | PyPI |
| quantities | 0.16.4 (>=0.16.4 required under numpy 2) | PyPI |
| tables | 3.10.2 | PyPI |
| ipywidgets | 8.1.7 | PyPI |
| ipyvolume / pythreejs | 0.6.3 / 2.4.2 | PyPI (optional, §5.2) |
| process-bigraph | 1.8.3 | PyPI |
| bigraph-schema | 1.6.0 | PyPI |
| bigraph-viz | 2.0.3 | PyPI |
| viva-superpowers | 0.22.0, git `HEAD` | git |
| viva-emitters | 0.3.0, git `HEAD` | git |
| viva-workspace | 0.1.0, git `HEAD` | git |
| viva-marketplace | 0.1.0, git `HEAD` | git |
| investigation-contracts | 0.2.0, git `HEAD` | git |
| pbg-basic-processes | 0.1.0, git `HEAD` | git (name NOT renamed — see §6.1) |
| vivarium-workbench | 0.3.78, git `HEAD` | git |
| viva-copasi | 0.1.0, git `HEAD` | git |
| copasi-basico / python-copasi | 0.86 / 4.46.300 | PyPI (via viva-copasi) |
| polars / pyarrow / duckdb / fsspec | 1.41.2 / 24.0.0 / 1.5.4 / 2026.6.0 | PyPI |
| tyssue-kernels | 0.1.0 (editable, `rust-kernels/`) | maturin (optional, §5.1) |
| maturin | 1.14.1 | PyPI (optional, §5.1) |

Newer versions of the loose PyPI entries are fine. The one hard pin that matters is
`tyssue` → the fork's `intersections` branch (*§4*); the vivarium-collective packages
track their default branches, so a break there shows up as a resolution or import
error rather than a silent drift.
