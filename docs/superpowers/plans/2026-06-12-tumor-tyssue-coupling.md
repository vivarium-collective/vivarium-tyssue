# Tumor population dynamics on a tyssue 2D epithelium — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Couple BioModels `BIOMD0000000903` (breast-cancer population ODE, run in COPASI) to a tyssue 2D vertex-model sheet so the SBML model's per-step birth/death fluxes drive discrete division/apoptosis/differentiation events on the mesh; deliver one investigation with two studies and a draft PR.

**Architecture:** A new tyssue-side `TumorCoupling` process is *"Gillespie, but driven by COPASI reaction fluxes instead of internal rates."* Each step it reads the COPASI `fluxes` output, scales each birth/death flux into a fractional event accumulator per cell type, fires `floor(accumulator)` discrete behaviors (`divide_crypt` / `apoptosis_extrusion` / `differentiation`) on cells it selects from the tyssue `datasets`, and emits scalar observables (`tumor_births`, `healthy_deaths`, `tumor_count`, …). Three processes share `Behaviors` / `Datasets` / `global_time` stores exactly like the existing `gillespie.composite.yaml`. Births/deaths/composition timeseries reuse the framework's `local:TimeSeriesFromObservables`; one new `TissueSheetSnapshots` viz renders cell-type-colored panels.

**Tech Stack:** process-bigraph, bigraph-schema, tyssue, pbg-copasi (`CopasiUTCProcess` + basico/COPASI), pbg-biomodels (`load_biomodel`), pbg-superpowers (Visualization, TimeSeriesFromObservables, study/investigation spine), matplotlib, pytest.

## Key facts discovered (do not re-derive)

- **COPASI process:** `pbg_copasi.processes.CopasiUTCProcess`, type address `local:CopasiUTCProcess`. Config `{model_source, time, intervals}`. Outputs `species_concentrations: map[float]` (SBML ids), `fluxes: map[float]`, `time: list[float]`. The `fluxes` dict is keyed by `get_reactions(dm).index` — the **reaction display names with spaces** (verified by probing the model).
- **Exact flux keys for BIOMD0000000903** (verified — match these strings *exactly*, including the model's own typos "Decrase", "strogen", "y immune"):
  - tumor birth: `"Induction of tumor cell"`
  - tumor death: `"Removal of tumor cell y immune cell and other immune response"`
  - healthy birth: `"Increase in the healthy cell in the system"`
  - healthy death: `"Decrase of healthy cell due to cancer"`
  - stem birth: `"Formation of Stem cell"`
  - stem death: `"Removal of stem cell from the system"`
  - (immune/estrogen reactions exist but are unused on the mesh)
- **Species:** `C` (stem, init 7.37e5), `T` (tumor, 7.62e6), `H` (healthy, 2.5e7), `I` (immune, 0), `E` (estrogen, 0). Fluxes are O(1e3–1e7).
- **tyssue process:** `vivarium_tyssue.processes.eulersolver.EulerSolver`, `local:EulerSolver`. Inputs `behaviors: list[node]`, `global_time: float`. Outputs `datasets: tyssue_data`, `network_changed`, `behaviors_update`. Consumes behaviors via `BEHAVIOR_MAP` + `EventManager` (`vivarium_tyssue/processes/eulersolver.py:185-227`).
- **Behavior dicts** (from `vivarium_tyssue/processes/gillespie.py:164-198`, executed by `vivarium_tyssue/behaviors/behaviors.py`):
  - divide: `{"func": "divide_crypt", "geom": <str>, "cell_uid": <int>, "dt": <float>, "cell_type": <str>, "crit_area": <float>, "growth_rate": <float>}` → `cell_division` (immediate when `area > crit_area`).
  - death: `{"func": "apoptosis_extrusion", "geom": <str>, "cell_uid": <int>, "dt": <float>, "crit_area": <float>, "shrink_rate": <float>}` → `remove_face` (immediate when `area < crit_area`).
  - differentiate: `{"func": "differentiation", "cell_uid": <int>, "new_type": <str>}` → relabels `face_df.cell_type`.
  - Immediate events: relaxed sheet faces have `area ≈ 1.0`, so `crit_area_div=0.5` (area>0.5 ⇒ divide) and `crit_area_apop=2.0` (area<2.0 ⇒ remove) make events fire on first execution.
- **Composite wiring** (mirror `vivarium_tyssue/composites/gillespie.composite.yaml`): each behavior-emitting process wires `behaviors → [Behaviors]`, EulerSolver wires `behaviors ← [Behaviors]`, `datasets ↔ [Datasets]`, `global_time ← [global_time]`. Default emitter `local:DataFrameParquetEmitter` over `Datasets/*`.
- **Registration:** processes in `vivarium_tyssue/processes/__init__.py::register_processes`; viz in `vivarium_tyssue/visualizations/__init__.py::__all__`; both invoked by `vivarium_tyssue/core.py::build_core`. `CopasiUTCProcess` must be registered there too.
- **Visualizations:** subclass `pbg_superpowers.visualization.Visualization`; Path-C viz read runs.db via `cfg["_runs_db_path"]` (`history` table: `step, global_time, state` JSON; rows filtered by `simulation_id`). `local:TimeSeriesFromObservables` plots *any* top-level scalar key in `history.state` whose name is listed in its `observables` config — so the adapter's scalar outputs become charts with **no new viz code**.

---

## Phase 0 — Dependencies, model caching, COPASI registration

### Task 1: Add pbg-copasi + pbg-biomodels deps

**Files:**
- Modify: `pyproject.toml` (dependencies array)

- [ ] **Step 1: Add the two git deps** after the `pbg-emitters` line in `pyproject.toml`:

```toml
    # COPASI process (CopasiUTCProcess) wrapping basico/COPASI for the SBML tumor
    # model, and the BioModels loader (load_biomodel) used to fetch + cache
    # BIOMD0000000903. Not on PyPI — git+https direct references like the others.
    "pbg-copasi @ git+https://github.com/vivarium-collective/pbg-copasi.git",
    "pbg-biomodels @ git+https://github.com/vivarium-collective/pbg-biomodels.git",
```

- [ ] **Step 2: Resolve and install**

Run: `cd /Users/eranagmon/code/vivarium-tyssue && uv sync`
Expected: resolves; installs `pbg-copasi`, `pbg-biomodels`, `copasi-basico`, `python-libsbml`, `biomodels` (or similar) with no conflict.

- [ ] **Step 3: Verify imports**

Run: `.venv/bin/python -c "import basico, pbg_copasi.processes, pbg_biomodels; from pbg_copasi.processes import CopasiUTCProcess; print('copasi+biomodels OK')"`
Expected: `copasi+biomodels OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add pbg-copasi + pbg-biomodels for SBML tumor coupling"
```

### Task 2: Fetch + cache the BIOMD0000000903 SBML

**Files:**
- Create: `workspace/datasets/BIOMD0000000903.xml` (committed)
- Create: `scripts/fetch_tumor_biomodel.py`

- [ ] **Step 1: Write the fetch script** at `scripts/fetch_tumor_biomodel.py`:

```python
"""Fetch + cache BioModels BIOMD0000000903 SBML into workspace/datasets/.

Run once; the cached XML is committed so composite runs are reproducible/offline.
"""
import shutil
from pathlib import Path

from pbg_biomodels.run_biomodels import load_biomodel

DEST = Path(__file__).resolve().parents[1] / "workspace" / "datasets" / "BIOMD0000000903.xml"


def main() -> None:
    result = load_biomodel("BIOMD0000000903", None)
    src = Path(result.sbml_path)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, DEST)
    print(f"cached {src} -> {DEST} ({DEST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

> If `load_biomodel`'s signature differs at runtime, fall back to a direct download:
> `urllib.request.urlretrieve("https://www.ebi.ac.uk/biomodels/model/download/BIOMD0000000903?filename=BIOMD0000000903_url.xml", DEST)`.

- [ ] **Step 2: Run it**

Run: `.venv/bin/python scripts/fetch_tumor_biomodel.py`
Expected: prints `cached … -> …/workspace/datasets/BIOMD0000000903.xml (… bytes)`; file exists.

- [ ] **Step 3: Verify the cached model loads + has the expected reactions**

Run:
```bash
.venv/bin/python -c "
import basico
dm = basico.load_model('workspace/datasets/BIOMD0000000903.xml')
rx = set(basico.get_reactions(model=dm).index)
need = {'Induction of tumor cell','Decrase of healthy cell due to cancer','Formation of Stem cell'}
assert need <= rx, sorted(rx)
print('model OK, reactions present')
"
```
Expected: `model OK, reactions present`

- [ ] **Step 4: Commit**

```bash
git add workspace/datasets/BIOMD0000000903.xml scripts/fetch_tumor_biomodel.py
git commit -m "data: cache BIOMD0000000903 SBML + fetch script"
```

### Task 3: Register CopasiUTCProcess in the workspace core

**Files:**
- Modify: `vivarium_tyssue/core.py`
- Test: `tests/test_tumor_core.py`

- [ ] **Step 1: Write the failing test** at `tests/test_tumor_core.py`:

```python
from vivarium_tyssue.core import build_core


def test_copasi_process_registered():
    core = build_core()
    assert "CopasiUTCProcess" in core.link_registry


def test_timeseries_from_observables_registered():
    # Shipped by pbg-superpowers; build_core should surface it for studies.
    core = build_core()
    assert "TimeSeriesFromObservables" in core.link_registry
```

- [ ] **Step 2: Run it, expect failure**

Run: `.venv/bin/python -m pytest tests/test_tumor_core.py -v`
Expected: FAIL (`CopasiUTCProcess` not in registry).

- [ ] **Step 3: Register COPASI + the framework timeseries viz in `build_core`.** In `vivarium_tyssue/core.py`, after the visualizations block (before `return core`), add:

```python
    # COPASI SBML process (pbg-copasi) — needed by the tumor composite. Degrade
    # gracefully if the COPASI/basico stack isn't installed.
    try:
        from pbg_copasi.processes import CopasiUTCProcess
        if "CopasiUTCProcess" not in core.link_registry:
            core.register_link("CopasiUTCProcess", CopasiUTCProcess)
    except Exception as exc:  # noqa: BLE001
        print(f"vivarium_tyssue.core: CopasiUTCProcess not registered ({type(exc).__name__}: {exc})")

    # Framework timeseries viz (pbg-superpowers) — plots scalar observables
    # (tumor_births, healthy_deaths, *_count) the TumorCoupling process emits.
    try:
        from pbg_superpowers.visualizations.timeseries_from_observables import TimeSeriesFromObservables
        if "TimeSeriesFromObservables" not in core.link_registry:
            core.register_link("TimeSeriesFromObservables", TimeSeriesFromObservables)
    except Exception as exc:  # noqa: BLE001
        print(f"vivarium_tyssue.core: TimeSeriesFromObservables not registered ({type(exc).__name__}: {exc})")
```

- [ ] **Step 4: Run the test, expect pass**

Run: `.venv/bin/python -m pytest tests/test_tumor_core.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add vivarium_tyssue/core.py tests/test_tumor_core.py
git commit -m "core: register CopasiUTCProcess + TimeSeriesFromObservables"
```

---

## Phase 1 — The TumorCoupling adapter (TDD, pure logic first)

### Task 4: Fractional-event accumulator (pure helper)

**Files:**
- Create: `vivarium_tyssue/processes/tumor_coupling.py`
- Test: `tests/test_tumor_coupling.py`

- [ ] **Step 1: Write the failing test** at `tests/test_tumor_coupling.py`:

```python
from vivarium_tyssue.processes.tumor_coupling import fractional_events


def test_accumulator_fires_when_crossing_one():
    # flux 4.0, scale 0.1, dt 1.0 -> 0.4 per step; needs 3 steps to fire once.
    acc = {"x": 0.0}
    n0, acc = fractional_events({"x": 4.0}, {"x": 0.1}, acc, dt=1.0)
    assert n0["x"] == 0 and abs(acc["x"] - 0.4) < 1e-9
    n1, acc = fractional_events({"x": 4.0}, {"x": 0.1}, acc, dt=1.0)
    assert n1["x"] == 0 and abs(acc["x"] - 0.8) < 1e-9
    n2, acc = fractional_events({"x": 4.0}, {"x": 0.1}, acc, dt=1.0)
    assert n2["x"] == 1 and abs(acc["x"] - 0.2) < 1e-9  # 1.2 -> fire 1, carry 0.2


def test_no_event_lost_to_rounding():
    acc = {"x": 0.0}
    total = 0
    for _ in range(100):
        n, acc = fractional_events({"x": 10.0}, {"x": 0.1}, acc, dt=1.0)
        total += n["x"]
    # 100 steps * 1.0 expected events = 100, give or take the carried remainder (<1).
    assert total in (99, 100)


def test_negative_and_missing_flux_clamped_to_zero():
    n, acc = fractional_events({"x": -5.0}, {"x": 0.1}, {"x": 0.0}, dt=1.0)
    assert n["x"] == 0 and acc["x"] == 0.0
    n, acc = fractional_events({}, {"x": 0.1}, {"x": 0.0}, dt=1.0)
    assert n["x"] == 0
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_tumor_coupling.py -v`
Expected: FAIL (`ImportError: cannot import name 'fractional_events'`).

- [ ] **Step 3: Implement the helper.** Create `vivarium_tyssue/processes/tumor_coupling.py`:

```python
"""TumorCoupling — drive discrete tyssue cell events from COPASI reaction fluxes.

"Gillespie, but driven by COPASI fluxes instead of internal rates." Each step:
read the SBML model's per-reaction birth/death fluxes, scale each into a
fractional event accumulator per cell type, fire floor(accumulator) discrete
behaviors on selected mesh cells, and emit scalar observables for the timeseries
charts. See docs/superpowers/specs/2026-06-12-tumor-tyssue-coupling-design.md.
"""
from __future__ import annotations

import math


def fractional_events(
    fluxes: dict, scales: dict, accumulators: dict, *, dt: float,
) -> tuple[dict, dict]:
    """Convert scaled fluxes into integer event counts via fractional accumulators.

    For each key in ``scales``: add ``max(flux, 0) * scale * dt`` to the running
    accumulator, fire ``floor(accumulator)`` events, and carry the remainder. A
    missing or negative flux contributes zero (no event is lost to rounding —
    the fractional remainder persists across steps).

    Returns ``(counts, new_accumulators)``.
    """
    counts: dict = {}
    new_acc = dict(accumulators)
    for key, scale in scales.items():
        flux = fluxes.get(key, 0.0)
        increment = max(float(flux), 0.0) * float(scale) * float(dt)
        total = new_acc.get(key, 0.0) + increment
        fired = int(math.floor(total))
        counts[key] = fired
        new_acc[key] = total - fired
    return counts, new_acc
```

- [ ] **Step 4: Run, expect pass**

Run: `.venv/bin/python -m pytest tests/test_tumor_coupling.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add vivarium_tyssue/processes/tumor_coupling.py tests/test_tumor_coupling.py
git commit -m "feat(tumor): fractional-event accumulator helper"
```

### Task 5: Cell-selection helper (which uids to divide/kill, with stem→tumor seeding)

**Files:**
- Modify: `vivarium_tyssue/processes/tumor_coupling.py`
- Test: `tests/test_tumor_coupling.py`

- [ ] **Step 1: Add failing tests** to `tests/test_tumor_coupling.py`:

```python
from vivarium_tyssue.processes.tumor_coupling import select_uids


def _face_df(uids_types):
    # Minimal face_df-as-dict-of-lists, like the emitted tyssue datasets.
    return {
        "unique_id": [u for u, _ in uids_types],
        "cell_type": [t for _, t in uids_types],
    }


def test_select_uids_picks_matching_type():
    fdf = _face_df([(1, "tumor"), (2, "healthy"), (3, "tumor")])
    picked = select_uids(fdf, "tumor", 2, exclude=set(), rng_pick=lambda items, k: items[:k])
    assert set(picked) == {1, 3}


def test_select_uids_caps_to_available():
    fdf = _face_df([(1, "tumor")])
    picked = select_uids(fdf, "tumor", 5, exclude=set(), rng_pick=lambda items, k: items[:k])
    assert picked == [1]  # capped


def test_select_uids_excludes_already_chosen():
    fdf = _face_df([(1, "healthy"), (2, "healthy")])
    picked = select_uids(fdf, "healthy", 1, exclude={1}, rng_pick=lambda items, k: items[:k])
    assert picked == [2]
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_tumor_coupling.py::test_select_uids_picks_matching_type -v`
Expected: FAIL (`cannot import name 'select_uids'`).

- [ ] **Step 3: Implement `select_uids`.** Append to `vivarium_tyssue/processes/tumor_coupling.py`:

```python
def _rows(face_df: dict) -> list[tuple[int, str]]:
    """[(unique_id, cell_type)] from a face_df-as-dict-of-lists (or empty)."""
    uids = face_df.get("unique_id") or []
    types = face_df.get("cell_type") or []
    return [(int(u), str(t)) for u, t in zip(uids, types)]


def select_uids(face_df: dict, cell_type: str, n: int, *, exclude: set, rng_pick) -> list:
    """Choose up to ``n`` unique_ids of cells whose type is ``cell_type``,
    skipping any in ``exclude``. Caps to availability. ``rng_pick(items, k)``
    selects k items from a list (injected for deterministic tests; the process
    passes a numpy-random sampler)."""
    candidates = [u for (u, t) in _rows(face_df) if t == cell_type and u not in exclude]
    if n <= 0 or not candidates:
        return []
    k = min(n, len(candidates))
    return list(rng_pick(candidates, k))
```

- [ ] **Step 4: Run, expect pass**

Run: `.venv/bin/python -m pytest tests/test_tumor_coupling.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add vivarium_tyssue/processes/tumor_coupling.py tests/test_tumor_coupling.py
git commit -m "feat(tumor): cell-selection helper"
```

### Task 6: The TumorCoupling Process

**Files:**
- Modify: `vivarium_tyssue/processes/tumor_coupling.py`
- Test: `tests/test_tumor_coupling.py`

**Behavior contract:**
- Config carries `birth_fluxes`/`death_fluxes` (cell_type → exact flux key), `scales` (event key → float), behavior params (`geom`, crit areas, rates, dt), and `seed` (initial tumor/stem focus counts).
- Inputs: `fluxes: map[float]`, `datasets: tyssue_data`, `global_time: float`.
- Outputs: `behaviors: list[node]` plus nine scalar floats: `{tumor,healthy,stem}_{births,deaths}` and `{tumor,healthy,stem}_count`.
- On the **first** update (global_time == 0), emit `differentiation` behaviors converting `seed.tumor` healthy cells → tumor and `seed.stem` → stem (seeding the focus); no flux events that step.
- Otherwise: for each cell type, fire `floor(scaled birth flux)` divisions and `floor(scaled death flux)` apoptoses. A **tumor birth** is realized by differentiating a `stem` cell → tumor when one is free, else dividing a `tumor` cell (honors C→T). `births`/`deaths` scalars report events fired this step; `*_count` reports current mesh cells of each type.

- [ ] **Step 1: Add a failing process test** to `tests/test_tumor_coupling.py`:

```python
import numpy as np
from vivarium_tyssue.processes.tumor_coupling import TumorCoupling
from vivarium_tyssue.core import build_core


def _make_proc(seed_tumor=0, seed_stem=0, scales=None):
    config = {
        "birth_fluxes": {"tumor": "T_birth", "healthy": "H_birth", "stem": "C_birth"},
        "death_fluxes": {"tumor": "T_death", "healthy": "H_death", "stem": "C_death"},
        "scales": scales or {
            "tumor_births": 1.0, "tumor_deaths": 1.0,
            "healthy_births": 1.0, "healthy_deaths": 1.0,
            "stem_births": 1.0, "stem_deaths": 1.0,
        },
        "geom": "SheetGeometry",
        "dt": 1.0, "growth_rate": 0.1, "shrink_rate": 0.1,
        "division_crit": 0.5, "apoptosis_crit": 2.0,
        "seed": {"tumor": seed_tumor, "stem": seed_stem},
    }
    return TumorCoupling(config=config, core=build_core())


def _datasets(types):
    return {"face_df": {"unique_id": list(range(len(types))), "cell_type": list(types)}}


def test_seed_step_differentiates_focus():
    proc = _make_proc(seed_tumor=2, seed_stem=1)
    ds = _datasets(["healthy"] * 10)
    out = proc.update({"fluxes": {}, "datasets": ds, "global_time": 0.0}, 1.0)
    funcs = [b["func"] for b in out["behaviors"]]
    new_types = [b["new_type"] for b in out["behaviors"]]
    assert funcs == ["differentiation"] * 3
    assert new_types.count("tumor") == 2 and new_types.count("stem") == 1


def test_flux_step_fires_division_and_apoptosis():
    np.random.seed(0)
    proc = _make_proc()
    proc._seeded = True  # skip seeding
    ds = _datasets(["tumor", "tumor", "healthy", "healthy", "stem"])
    fluxes = {"T_birth": 1.0, "H_death": 1.0, "T_death": 0.0,
              "H_birth": 0.0, "C_birth": 0.0, "C_death": 0.0}
    out = proc.update({"fluxes": fluxes, "datasets": ds, "global_time": 5.0}, 1.0)
    funcs = sorted(b["func"] for b in out["behaviors"])
    # one tumor birth (divide_crypt or differentiation of stem) + one healthy death
    assert "apoptosis_extrusion" in funcs
    assert out["healthy_deaths"] == 1.0 and out["tumor_births"] == 1.0
    assert out["tumor_count"] == 2.0 and out["healthy_count"] == 2.0


def test_counts_reported_each_step():
    proc = _make_proc()
    proc._seeded = True
    ds = _datasets(["tumor", "healthy", "healthy", "stem"])
    out = proc.update({"fluxes": {}, "datasets": ds, "global_time": 3.0}, 1.0)
    assert out["tumor_count"] == 1.0 and out["healthy_count"] == 2.0 and out["stem_count"] == 1.0
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_tumor_coupling.py::test_seed_step_differentiates_focus -v`
Expected: FAIL (`cannot import name 'TumorCoupling'`).

- [ ] **Step 3: Implement the Process.** Append to `vivarium_tyssue/processes/tumor_coupling.py`:

```python
import numpy as np
from process_bigraph import Process

_EVENT_KEYS = [
    "tumor_births", "tumor_deaths", "healthy_births", "healthy_deaths",
    "stem_births", "stem_deaths",
]
_TYPES = ["tumor", "healthy", "stem"]


class TumorCoupling(Process):
    """Drive tyssue cell events from COPASI reaction fluxes (see module docstring)."""

    config_schema = {
        "birth_fluxes": "map[string]",   # cell_type -> exact COPASI flux key
        "death_fluxes": "map[string]",   # cell_type -> exact COPASI flux key
        "scales": "map[float]",          # event key (e.g. 'tumor_births') -> scale
        "geom": "string",
        "dt": "float",
        "growth_rate": "float",
        "shrink_rate": "float",
        "division_crit": "float",
        "apoptosis_crit": "float",
        "seed": "map[integer]",          # {'tumor': n, 'stem': m} initial focus
    }

    def initialize(self, config):
        self.birth_fluxes = config["birth_fluxes"]
        self.death_fluxes = config["death_fluxes"]
        self.scales = config["scales"]
        self.geom = config["geom"]
        self.dt = config["dt"]
        self.growth_rate = config["growth_rate"]
        self.shrink_rate = config["shrink_rate"]
        self.division_crit = config["division_crit"]
        self.apoptosis_crit = config["apoptosis_crit"]
        self.seed = config.get("seed", {}) or {}
        self._birth_acc = {t: 0.0 for t in _TYPES}
        self._death_acc = {t: 0.0 for t in _TYPES}
        self._seeded = False

    def inputs(self):
        return {"fluxes": "map[float]", "datasets": "tyssue_data", "global_time": "float"}

    def outputs(self):
        out = {"behaviors": "list[node]"}
        for k in _EVENT_KEYS:
            out[k] = "float"
        for t in _TYPES:
            out[f"{t}_count"] = "float"
        return out

    # -- behavior dict builders (shapes match gillespie.py / behaviors.py) --
    def _divide(self, uid, cell_type):
        return {"func": "divide_crypt", "geom": self.geom, "cell_uid": int(uid),
                "dt": self.dt, "cell_type": cell_type, "crit_area": self.division_crit,
                "growth_rate": self.growth_rate}

    def _kill(self, uid):
        return {"func": "apoptosis_extrusion", "geom": self.geom, "cell_uid": int(uid),
                "dt": self.dt, "crit_area": self.apoptosis_crit, "shrink_rate": self.shrink_rate}

    def _differentiate(self, uid, new_type):
        return {"func": "differentiation", "cell_uid": int(uid), "new_type": new_type}

    @staticmethod
    def _pick(items, k):
        idx = np.random.choice(len(items), size=k, replace=False)
        return [items[i] for i in idx]

    def _counts(self, face_df):
        rows = _rows(face_df)
        return {t: float(sum(1 for (_, ct) in rows if ct == t)) for t in _TYPES}

    def update(self, inputs, interval):
        face_df = inputs["datasets"]["face_df"]
        behaviors = []
        used: set = set()
        fired = {k: 0 for k in _EVENT_KEYS}

        # --- Seeding step: convert the initial focus, then return. ---
        if not self._seeded:
            self._seeded = True
            for healthy_uid in select_uids(face_df, "healthy", int(self.seed.get("tumor", 0)),
                                           exclude=used, rng_pick=self._pick):
                behaviors.append(self._differentiate(healthy_uid, "tumor")); used.add(healthy_uid)
            for healthy_uid in select_uids(face_df, "healthy", int(self.seed.get("stem", 0)),
                                           exclude=used, rng_pick=self._pick):
                behaviors.append(self._differentiate(healthy_uid, "stem")); used.add(healthy_uid)
            counts = self._counts(face_df)
            return self._result(behaviors, fired, counts)

        # --- Flux-driven events. ---
        births, self._birth_acc = fractional_events(
            {t: inputs["fluxes"].get(self.birth_fluxes.get(t, ""), 0.0) for t in _TYPES},
            {t: self.scales.get(f"{t}_births", 0.0) for t in _TYPES},
            self._birth_acc, dt=self.dt)
        deaths, self._death_acc = fractional_events(
            {t: inputs["fluxes"].get(self.death_fluxes.get(t, ""), 0.0) for t in _TYPES},
            {t: self.scales.get(f"{t}_deaths", 0.0) for t in _TYPES},
            self._death_acc, dt=self.dt)

        # Births. Tumor birth prefers differentiating a free stem cell (C->T).
        for t in _TYPES:
            for _ in range(births[t]):
                if t == "tumor":
                    stem = select_uids(face_df, "stem", 1, exclude=used, rng_pick=self._pick)
                    if stem:
                        behaviors.append(self._differentiate(stem[0], "tumor")); used.add(stem[0])
                        fired["tumor_births"] += 1; continue
                pick = select_uids(face_df, t, 1, exclude=used, rng_pick=self._pick)
                if pick:
                    behaviors.append(self._divide(pick[0], t)); used.add(pick[0])
                    fired[f"{t}_births"] += 1

        # Deaths.
        for t in _TYPES:
            for _ in range(deaths[t]):
                pick = select_uids(face_df, t, 1, exclude=used, rng_pick=self._pick)
                if pick:
                    behaviors.append(self._kill(pick[0])); used.add(pick[0])
                    fired[f"{t}_deaths"] += 1

        return self._result(behaviors, fired, self._counts(face_df))

    def _result(self, behaviors, fired, counts):
        result = {"behaviors": behaviors}
        for k in _EVENT_KEYS:
            result[k] = float(fired[k])
        for t in _TYPES:
            result[f"{t}_count"] = counts[t]
        return result
```

- [ ] **Step 4: Run, expect pass**

Run: `.venv/bin/python -m pytest tests/test_tumor_coupling.py -v`
Expected: PASS (9 passed). If `ProcessTypes` isn't the right core ctor for a bare process in this version, build with `from vivarium_tyssue.core import build_core; TumorCoupling(config=config, core=build_core())` in the test helper.

- [ ] **Step 5: Commit**

```bash
git add vivarium_tyssue/processes/tumor_coupling.py tests/test_tumor_coupling.py
git commit -m "feat(tumor): TumorCoupling process (flux-driven cell events)"
```

### Task 7: Register TumorCoupling

**Files:**
- Modify: `vivarium_tyssue/processes/__init__.py`
- Test: `tests/test_tumor_core.py`

- [ ] **Step 1: Add failing test** to `tests/test_tumor_core.py`:

```python
def test_tumor_coupling_registered():
    core = build_core()
    assert "TumorCoupling" in core.link_registry
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_tumor_core.py::test_tumor_coupling_registered -v`
Expected: FAIL.

- [ ] **Step 3: Register it.** In `vivarium_tyssue/processes/__init__.py` add the import and link:

```python
from vivarium_tyssue.processes.tumor_coupling import TumorCoupling
```
and inside `register_processes`, before `return core`:
```python
    core.register_link("TumorCoupling", TumorCoupling)
```

- [ ] **Step 4: Run, expect pass**

Run: `.venv/bin/python -m pytest tests/test_tumor_core.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vivarium_tyssue/processes/__init__.py tests/test_tumor_core.py
git commit -m "core: register TumorCoupling process"
```

### Task 8: Cell-type colors for the tumor mesh

**Files:**
- Modify: `vivarium_tyssue/visualizations/tissue_gif.py` (CELL_TYPE_COLORS)

- [ ] **Step 1: Extend the palette.** Add tumor/healthy/stem entries to `CELL_TYPE_COLORS` in `vivarium_tyssue/visualizations/tissue_gif.py`:

```python
    "healthy": "#4a90d9",   # blue — normal epithelium
    "tumor": "#c0392b",     # red — tumor cells
    "stem": "#8e44ad",      # purple — cancer stem cells
```

- [ ] **Step 2: Sanity import**

Run: `.venv/bin/python -c "from vivarium_tyssue.visualizations.tissue_gif import CELL_TYPE_COLORS; assert 'tumor' in CELL_TYPE_COLORS; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add vivarium_tyssue/visualizations/tissue_gif.py
git commit -m "viz: tumor/healthy/stem cell-type colors"
```

---

## Phase 2 — Snapshots viz + composites

### Task 9: TissueSheetSnapshots visualization (cell-type-colored panels over time)

**Files:**
- Create: `vivarium_tyssue/visualizations/tissue_snapshots.py`
- Modify: `vivarium_tyssue/visualizations/__init__.py`
- Test: `tests/test_tissue_snapshots.py`

- [ ] **Step 1: Write the failing test** at `tests/test_tissue_snapshots.py`:

```python
from vivarium_tyssue.visualizations.tissue_snapshots import _panels_html, _cell_type_color


def test_color_lookup_falls_back():
    assert _cell_type_color("tumor").startswith("#")
    assert _cell_type_color("unknown-type").startswith("#")  # default, no crash


def test_panels_html_handles_empty_frames():
    html = _panels_html([], coords=["x", "y"], n_panels=4, title="t")
    assert "No snapshots" in html


def test_panels_html_emits_figure_for_frames():
    # one frame with two triangular faces via edge source/target coords
    frame = {"t": 0.0, "datasets": {"face_df": {"unique_id": [0], "cell_type": ["tumor"]},
             "edge_df": {"face": [0, 0, 0], "sx": [0.0, 1.0, 0.5], "sy": [0.0, 0.0, 1.0],
                         "tx": [1.0, 0.5, 0.0], "ty": [0.0, 1.0, 0.0]}}}
    html = _panels_html([frame], coords=["x", "y"], n_panels=1, title="snap")
    assert "data:image/png;base64," in html
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_tissue_snapshots.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement.** Create `vivarium_tyssue/visualizations/tissue_snapshots.py`:

```python
"""TissueSheetSnapshots — multi-panel static figure of the sheet over time,
faces colored by cell_type. Path C: reads runs.db (shares the loader with
tissue_gif). Renders filled face polygons by grouping edge_df rows per face."""
from __future__ import annotations

import base64
import io
from pathlib import Path

from pbg_superpowers.visualization import Visualization
from vivarium_tyssue.visualizations.tissue_gif import (
    _load_frames, _subsample, _empty_html, CELL_TYPE_COLORS,
)

_DEFAULT_COLOR = "#9aa0a6"


def _cell_type_color(cell_type: str) -> str:
    return CELL_TYPE_COLORS.get(str(cell_type), _DEFAULT_COLOR)


def _face_polygons(datasets: dict, coords: list[str]):
    """Yield (face_uid, cell_type, [(x,y), ...]) for each face from edge_df.

    Uses s<axis>/t<axis> edge endpoint columns grouped by edge_df['face'].
    """
    import pandas as pd
    face = datasets.get("face_df") or {}
    edge = datasets.get("edge_df") or {}
    if not face or not edge:
        return
    fdf = pd.DataFrame(face)
    edf = pd.DataFrame(edge)
    a, b = coords[0], coords[1]
    needed = {"face", f"s{a}", f"s{b}", f"t{a}", f"t{b}"}
    if not needed <= set(edf.columns):
        return
    type_by_idx = dict(zip(range(len(fdf)), fdf.get("cell_type", ["?"] * len(fdf))))
    uid_by_idx = dict(zip(range(len(fdf)), fdf.get("unique_id", list(range(len(fdf))))))
    for face_idx, grp in edf.groupby("face"):
        pts = list(zip(grp[f"s{a}"].tolist(), grp[f"s{b}"].tolist()))
        idx = int(face_idx)
        yield uid_by_idx.get(idx, idx), type_by_idx.get(idx, "?"), pts


def _panels_html(frames: list, coords: list[str], n_panels: int, title: str) -> str:
    if not frames:
        return _empty_html("No snapshots: runs.db has no tissue datasets.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    sampled = _subsample(frames, n_panels)
    ncols = min(len(sampled), 4) or 1
    nrows = (len(sampled) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for i, fr in enumerate(sampled):
        ax = axes[i // ncols][i % ncols]
        any_poly = False
        for _uid, ctype, pts in _face_polygons(fr["datasets"], coords):
            if len(pts) >= 3:
                ax.add_patch(Polygon(pts, closed=True, facecolor=_cell_type_color(ctype),
                                     edgecolor="black", linewidth=0.4, alpha=0.85))
                any_poly = True
        if any_poly:
            ax.autoscale_view()
            ax.set_aspect("equal")
        ax.set_title(f"t = {fr['t']:.2f}", fontsize=9)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (f'<figure style="margin:0;text-align:center"><img alt="{title}" '
            f'style="max-width:100%;height:auto" src="data:image/png;base64,{b64}"/>'
            f'<figcaption style="font-family:system-ui;font-size:0.85rem;color:#666">'
            f'{title} — {len(frames)} steps, {len(sampled)} panels</figcaption></figure>')


class TissueSheetSnapshots(Visualization):
    """Multi-panel snapshots of the sheet over time, faces colored by cell_type."""

    config_schema = {
        **Visualization.config_schema,
        "coords": {"_type": "list[string]", "_default": ["x", "y"]},
        "n_panels": {"_type": "integer", "_default": 6},
        "_runs_db_path": {"_type": "string", "_default": ""},
    }

    def inputs(self) -> dict:
        return {}

    def _render_html(self) -> str:
        cfg = getattr(self, "config", None) or {}
        frames = _load_frames(cfg.get("_runs_db_path") or "")
        return _panels_html(frames, list(cfg.get("coords") or ["x", "y"]),
                            int(cfg.get("n_panels") or 6), cfg.get("title") or "tissue snapshots")

    def render(self) -> str:
        return self._render_html()

    def update(self, state: dict) -> dict:
        return {"html": self._render_html()}
```

- [ ] **Step 4: Register in `__init__.py`.** Update `vivarium_tyssue/visualizations/__init__.py`:

```python
"""vivarium_tyssue visualization Steps (auto-discovered by allocate_core)."""

from .tissue_gif import TissueSheetGif, TissueCryptGif3D
from .tissue_snapshots import TissueSheetSnapshots

__all__ = ["TissueSheetGif", "TissueCryptGif3D", "TissueSheetSnapshots"]
```

- [ ] **Step 5: Run, expect pass**

Run: `.venv/bin/python -m pytest tests/test_tissue_snapshots.py -v && .venv/bin/python -c "from vivarium_tyssue.core import build_core; assert 'TissueSheetSnapshots' in build_core().link_registry; print('registered')"`
Expected: tests PASS; prints `registered`.

- [ ] **Step 6: Commit**

```bash
git add vivarium_tyssue/visualizations/tissue_snapshots.py vivarium_tyssue/visualizations/__init__.py tests/test_tissue_snapshots.py
git commit -m "viz: TissueSheetSnapshots (cell-type-colored panels)"
```

### Task 10: Baseline 2D-sheet composite (mechanics only)

**Files:**
- Create: `vivarium_tyssue/composites/epithelium_2d.composite.yaml`
- Modify: `tests/test_composites.py` (`ALL_NAMES` set)
- Test: `tests/test_tumor_composites.py`

> **Loader is the repo's own** (`tests/test_composites.py:57-63`): `load_spec(path)` then `build_composite_from_spec(spec, overrides={"interval": ...}, core=core)` then `comp.run(N)`. Use exactly this — do not invent a loader.

- [ ] **Step 1: Write the failing test** at `tests/test_tumor_composites.py`:

```python
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "vivarium_tyssue" / "composites"


def _run(spec_name, steps, interval=1.0):
    import sys
    sys.path.insert(0, str(ROOT))
    pytest.importorskip("tables", reason="HDF5 mesh loading needs pytables")
    from pbg_superpowers.composite_spec import load_spec, build_composite_from_spec
    from vivarium_tyssue.core import build_core

    core = build_core()
    spec = load_spec(SPECS / f"{spec_name}.composite.yaml")
    comp = build_composite_from_spec(spec, overrides={"interval": interval}, core=core)
    comp.run(steps)
    return comp


def test_baseline_runs_without_behaviors():
    comp = _run("epithelium_2d", 3)
    assert comp is not None  # smoke: flat-sheet mechanics complete
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_tumor_composites.py::test_baseline_runs_without_behaviors -v`
Expected: FAIL (spec file missing).

- [ ] **Step 2b: Update `ALL_NAMES` in `tests/test_composites.py`** so `test_all_composites_present` accepts the two new composites:

```python
ALL_NAMES = {"base_solver", "regulation", "stochastic", "jamming", "gradient",
             "anisotropic", "gillespie", "epithelium_2d", "tumor"}
```

- [ ] **Step 3: Create the baseline spec** `vivarium_tyssue/composites/epithelium_2d.composite.yaml` — a copy of `stochastic.composite.yaml` with the `Stochastic` process removed (EulerSolver only), the long `migration_strength` list preserved, and `cell_type: healthy` added under `parameters.face_df`. Keep the `name`, `emitters`, `parameters.interval`, and the full `Tyssue` block from `stochastic.composite.yaml`; set:

```yaml
name: epithelium_2d
description: Baseline flat (square) 2D sheet epithelium — pure mechanical relaxation,
  no tumor coupling. Control for the tumor-composite comparison study.
# ... requires/emitters/parameters as in stochastic.composite.yaml ...
state:
  Tyssue:
    _type: process
    address: local:EulerSolver
    config:
      name: Epithelium 2D
      eptm: workspace/datasets/test_square.hf5
      tissue_type: Sheet
      parameters:
        face_df:
          area_elasticity: 1.0
          prefered_area: 1.0
          perimeter_elasticity: 0.1
          prefered_perimeter: 3.6
          cell_type: healthy
          is_alive: 1.0
        edge_df: {line_tension: 0.0, is_active: 1.0}
        vert_df: {viscosity: 1.0, is_alive: 1.0}
      geom: SheetGeometry
      effectors: [LineTension, FaceAreaElasticity, PerimeterElasticity]
      ref_effector: FaceAreaElasticity
      factory: model_factory
      settings: {threshold_length: 0.03}
      auto_reconnect: true
      bounds: {}
      output_columns: {}
      maps: {}
    inputs:
      behaviors: [Behaviors]
      global_time: [global_time]
    outputs:
      datasets: [Datasets]
      network_changed: [Network Changed]
      behaviors_update: [Behaviors]
    interval: ${interval}
  Network Changed: false
  Behaviors: {}
```

> **IMPORTANT (verified):** use `factory: model_factory` and `effectors: [LineTension, FaceAreaElasticity, PerimeterElasticity]` / `ref_effector: FaceAreaElasticity` — this exactly mirrors `anisotropic.composite.yaml`, the one composite the existing suite runs end-to-end. Do NOT use `model_factory_bound` (it is not registered in the installed tyssue — `stochastic` fails with `KeyError: 'model_factory_bound'`). Copy the `emitters:` block verbatim from `stochastic.composite.yaml`. `line_tension` defaults to 0.0 so the LineTension effector is inert (pure area+perimeter mechanics).

- [ ] **Step 4: Run, expect pass**

Run: `.venv/bin/python -m pytest tests/test_tumor_composites.py::test_baseline_runs_without_behaviors -v`
Expected: PASS. (If the run errors on a missing effector/column, align the config with a known-good `*.composite.yaml` that uses `test_square.hf5`.)

- [ ] **Step 5: Commit**

```bash
git add vivarium_tyssue/composites/epithelium_2d.composite.yaml tests/test_tumor_composites.py tests/test_composites.py
git commit -m "composite: baseline 2D epithelium sheet (mechanics only)"
```

### Task 11: Tumor composite (Copasi + TumorCoupling + EulerSolver)

**Files:**
- Create: `vivarium_tyssue/composites/tumor.composite.yaml`
- Test: `tests/test_tumor_composites.py`

- [ ] **Step 1: Add a failing integration test** to `tests/test_tumor_composites.py`:

```python
def test_tumor_composite_runs_end_to_end():
    # Smoke: COPASI + TumorCoupling + EulerSolver step together without error.
    comp = _run("tumor", 6)
    assert comp is not None
```

> This mirrors the repo's own `test_anisotropic_runs_end_to_end` smoke philosophy. Behavior *correctness* (events fire, types shift, counts reported) is covered deterministically by the `TumorCoupling` unit tests in Task 6; the "tumor cells actually appear on the mesh" check is verified against a real dashboard run + runs.db in Step 5 below, where state access is concrete.

- [ ] **Step 2: Run, expect failure**

Run: `.venv/bin/python -m pytest tests/test_tumor_composites.py::test_tumor_composite_fires_events_and_shifts_types -v`
Expected: FAIL (spec missing).

- [ ] **Step 3: Create `vivarium_tyssue/composites/tumor.composite.yaml`.** Same `Tyssue` block as `epithelium_2d` (flat square, `cell_type: healthy`), plus the `Copasi` and `TumorCoupling` processes and observable stores:

```yaml
name: tumor
description: Breast-cancer population ODE (BIOMD0000000903 in COPASI) coupled to a
  flat 2D tyssue sheet — per-step birth/death fluxes drive discrete division /
  apoptosis / differentiation on the mesh (tumor / healthy / stem cell types).
tags: [tissue, multi-cell, cells, tumor, sbml]
requires:
  processes: [EulerSolver, TumorCoupling, CopasiUTCProcess]
  types: [tyssue_data, behaviors]
emitters:
- address: local:DataFrameParquetEmitter
  config: {out_dir: out/parquet}
  paths: [Datasets/vert_df, Datasets/face_df, Datasets/edge_df, Datasets/cell_df]
parameters:
  interval: {type: float, default: 1.0, description: Solver / coupling step (dt).}
state:
  Tyssue:
    _type: process
    address: local:EulerSolver
    config:
      name: Tumor Epithelium 2D
      eptm: workspace/datasets/test_square.hf5
      tissue_type: Sheet
      parameters:
        face_df:
          area_elasticity: 1.0
          prefered_area: 1.0
          perimeter_elasticity: 0.1
          prefered_perimeter: 3.6
          cell_type: healthy
          is_alive: 1.0
        edge_df: {line_tension: 0.0, is_active: 1.0}
        vert_df: {viscosity: 1.0, is_alive: 1.0}
      geom: SheetGeometry
      effectors: [LineTension, FaceAreaElasticity, PerimeterElasticity]
      ref_effector: FaceAreaElasticity
      factory: model_factory
      settings: {threshold_length: 0.03}
      auto_reconnect: true
      bounds: {}
      output_columns: {}
      maps: {}
    inputs:
      behaviors: [Behaviors]
      global_time: [global_time]
    outputs:
      datasets: [Datasets]
      network_changed: [Network Changed]
      behaviors_update: [Behaviors]
    interval: ${interval}
  Copasi:
    _type: process
    address: local:CopasiUTCProcess
    config:
      model_source: workspace/datasets/BIOMD0000000903.xml
      time: 1.0
      intervals: 10
    inputs: {}
    outputs:
      fluxes: [Fluxes]
      species_concentrations: [Species]
      time: [SbmlTime]
    interval: ${interval}
  TumorCoupling:
    _type: process
    address: local:TumorCoupling
    config:
      birth_fluxes:
        tumor: Induction of tumor cell
        healthy: Increase in the healthy cell in the system
        stem: Formation of Stem cell
      death_fluxes:
        tumor: Removal of tumor cell y immune cell and other immune response
        healthy: Decrase of healthy cell due to cancer
        stem: Removal of stem cell from the system
      scales:
        tumor_births: 1.0e-6
        tumor_deaths: 1.0e-6
        healthy_births: 1.0e-7
        healthy_deaths: 3.0e-8
        stem_births: 3.0e-7
        stem_deaths: 1.0e-4
      geom: SheetGeometry
      dt: 1.0
      growth_rate: 0.1
      shrink_rate: 0.1
      division_crit: 0.5
      apoptosis_crit: 2.0
      seed: {tumor: 3, stem: 1}
    inputs:
      fluxes: [Fluxes]
      datasets: [Datasets]
      global_time: [global_time]
    outputs:
      behaviors: [Behaviors]
      tumor_births: [tumor_births]
      tumor_deaths: [tumor_deaths]
      healthy_births: [healthy_births]
      healthy_deaths: [healthy_deaths]
      stem_births: [stem_births]
      stem_deaths: [stem_deaths]
      tumor_count: [tumor_count]
      healthy_count: [healthy_count]
      stem_count: [stem_count]
    interval: ${interval}
  Network Changed: false
  Behaviors: {}
  Fluxes: {}
  Species: {}
  SbmlTime: []
  tumor_births: 0.0
  tumor_deaths: 0.0
  healthy_births: 0.0
  healthy_deaths: 0.0
  stem_births: 0.0
  stem_deaths: 0.0
  tumor_count: 0.0
  healthy_count: 0.0
  stem_count: 0.0
```

> Note: `Copasi.inputs: {}` leaves the COPASI species inputs unwired so the ODE evolves on its own internal state (`update_model=True`); we only read its `fluxes`. Verify in Step 4 that concentrations actually change across steps (if they don't, wire `species_concentrations` both in and out of the `Species` store).

- [ ] **Step 4: Run, expect pass**

Run: `.venv/bin/python -m pytest tests/test_tumor_composites.py -v`
Expected: PASS — `tumor` appears in the mesh `cell_type`s after 6 steps.

- [ ] **Step 5: Verify the COPASI ODE actually advances each step** (concrete probe — independent of Composite-state access):

```bash
.venv/bin/python -c "
from vivarium_tyssue.core import build_core
from pbg_copasi.processes import CopasiUTCProcess
p = CopasiUTCProcess(config={'model_source':'workspace/datasets/BIOMD0000000903.xml','time':1.0,'intervals':10}, core=build_core())
s = p.initial_state()
conc = s.get('species_concentrations', s)
o1 = p.update({'species_concentrations': conc, 'species_counts': {}}, 1.0)
o2 = p.update({'species_concentrations': o1['species_concentrations'], 'species_counts': {}}, 1.0)
print('tumor birth flux step1:', o1['fluxes'].get('Induction of tumor cell'))
print('T conc changed:', o1['species_concentrations'] != o2['species_concentrations'])
"
```
Expected: a nonzero `'Induction of tumor cell'` flux and `T conc changed: True`. If concentrations are static, the ODE isn't advancing — wire `species_concentrations` both in *and* out of the `Species` store in the composite (and pass the carried-forward concentrations as the process input). Watchable-cadence `scales` tuning happens against the real dashboard run in Task 12 Step 4; document the final values there. No silent caps — the process only fires up to available cells.

- [ ] **Step 6: Commit**

```bash
git add vivarium_tyssue/composites/tumor.composite.yaml tests/test_tumor_composites.py
git commit -m "composite: tumor (COPASI BIOMD0000000903 coupled to 2D sheet)"
```

---

## Phase 3 — Investigation + two studies + reports

### Task 12: Create the investigation + tumor study (Study 1)

**Files:**
- Create: `workspace/investigations/tumor-tyssue/investigation.yaml` (via skill)
- Create: `workspace/studies/tumor-composite/study.yaml` (via skill)

- [ ] **Step 1: Create the investigation** with the pbg-investigation skill:

Run (via the Skill tool, not bash): `/pbg-investigation new` — name `tumor-tyssue`, overview: *"Couple BIOMD0000000903 breast-cancer population ODE (COPASI) to a tyssue 2D epithelial sheet; demonstrate SBML-driven discrete cell birth/death and compare against a baseline sheet."*

- [ ] **Step 2: Create Study 1 (`tumor-composite`)** with the pbg-study skill. Phases Design→Build→Simulate→Evaluate. Baseline composite = `tumor`. Declare observables (names matched by `TimeSeriesFromObservables`) with `store_path`:

```yaml
observables:
- {name: tumor_births,   store_path: tumor_births,   units: cells/step}
- {name: healthy_births, store_path: healthy_births, units: cells/step}
- {name: stem_births,    store_path: stem_births,    units: cells/step}
- {name: tumor_deaths,   store_path: tumor_deaths,   units: cells/step}
- {name: healthy_deaths, store_path: healthy_deaths, units: cells/step}
- {name: stem_deaths,    store_path: stem_deaths,    units: cells/step}
- {name: tumor_count,    store_path: tumor_count,    units: cells}
- {name: healthy_count,  store_path: healthy_count,  units: cells}
- {name: stem_count,     store_path: stem_count,     units: cells}
```

- [ ] **Step 3: Declare the five visualizations** in `study.yaml.visualizations`:

```yaml
visualizations:
- {address: local:TissueSheetSnapshots, config: {title: Tumor sheet snapshots, coords: [x, y], n_panels: 6}}
- {address: local:TissueSheetGif, config: {title: Tumor sheet animation, coords: [x, y], num_frames: 60}}
- {address: local:TimeSeriesFromObservables, config: {title: Cell births over time, observables: [tumor_births, healthy_births, stem_births]}}
- {address: local:TimeSeriesFromObservables, config: {title: Cell deaths over time, observables: [tumor_deaths, healthy_deaths, stem_deaths]}}
- {address: local:TimeSeriesFromObservables, config: {title: Cell types over time, observables: [tumor_count, healthy_count, stem_count]}}
```

- [ ] **Step 4: Run a Simulate-phase run** of the `tumor` composite (~60–100 steps) via the dashboard / `/pbg-study` run, so runs.db is populated and the viz light up.

- [ ] **Step 5: Add the study to the investigation** (`/pbg-investigation add-study tumor-tyssue tumor-composite`).

- [ ] **Step 6: Commit**

```bash
git add workspace/investigations workspace/studies
git commit -m "study: tumor-composite (Study 1) with 5 visualizations"
```

### Task 13: Comparison study (Study 2) — tumor vs baseline

**Files:**
- Create: `workspace/studies/tumor-vs-baseline/study.yaml` (via skill)

- [ ] **Step 1: Create Study 2 (`tumor-vs-baseline`)** with pbg-study. Two simulation sources: `tumor` composite and `epithelium_2d` baseline. Same observable names; `TimeSeriesFromObservables` overlays both runs (it colors per run). For the baseline (no TumorCoupling), the `*_count` observables won't exist — so for composition the comparison reads `cell_type` counts from each run's emitted `face_df` instead. Declare the four comparison metrics as visualizations / findings:

```yaml
visualizations:
- {address: local:TimeSeriesFromObservables, config: {title: Cell-type counts — tumor vs baseline, observables: [tumor_count, healthy_count, stem_count], sources: [tumor, epithelium_2d]}}
- {address: local:TimeSeriesFromObservables, config: {title: Births & deaths — tumor vs baseline, observables: [tumor_births, healthy_deaths], sources: [tumor, epithelium_2d]}}
- {address: local:TissueSheetSnapshots, config: {title: Baseline sheet snapshots, coords: [x, y], n_panels: 6, sources: [epithelium_2d]}}
- {address: local:TissueSheetSnapshots, config: {title: Tumor sheet snapshots, coords: [x, y], n_panels: 6, sources: [tumor]}}
```

> The four design metrics — (1) cell-type composition, (2) total count + cumulative births/deaths, (3) tissue morphology/area, (4) healthy-cell survival — map to: composition + counts charts above; a morphology/area finding computed from `face_df.area` per run (add a `face_area_mean` observable to TumorCoupling/baseline if a numeric chart is wanted, or report it as a finding from the emitted datasets); healthy-cell survival = the `healthy_count` (tumor) vs healthy `face_df` count (baseline) trajectory. Record these as `findings` in study.yaml with the biology-forward narrative (Task 14).

- [ ] **Step 2: Run both composites** (tumor + baseline) for matched step counts via the dashboard so both appear in runs.db with distinct `sim_name`.

- [ ] **Step 3: Add the study to the investigation**; commit.

```bash
git add workspace/studies/tumor-vs-baseline
git commit -m "study: tumor-vs-baseline (Study 2) comparison"
```

### Task 14: Author findings + regenerate reports

**Files:**
- Modify: both `study.yaml` files (findings), investigation overview

- [ ] **Step 1: Author findings** for each study with the spine narrative skill (`/pbg-biology-forward`) — quantitative slots filled from the runs, then biological interpretation (tumor expansion, healthy displacement, qualitative-not-quantitative framing from the spec).

- [ ] **Step 2: Regenerate** dashboard + investigation reports: `/pbg-report` (runs reviewer-readiness audit + lint + render).

- [ ] **Step 3: Lint clean**

Run: `.venv/bin/python scripts/lint-workspace.py`
Expected: `workspace lint: OK`, 1 investigation, 2 studies.

- [ ] **Step 4: Commit**

```bash
git add workspace/ 
git commit -m "report: author findings + regenerate tumor investigation reports"
```

---

## Phase 4 — Full check + draft PR

### Task 15: Whole-suite check + draft PR

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass (existing 10 + new tumor/coupling/snapshots/composite tests).

- [ ] **Step 2: Push the branch**

```bash
git push -u origin feat/tumor-tyssue-investigation
```

- [ ] **Step 3: Confirm PR base** with the user (`master` vs the `vivarium-tyssue` working branch), then open the **draft** PR:

```bash
gh pr create --draft --base <confirmed-base> \
  --title "Tumor population dynamics on a tyssue 2D epithelium" \
  --body "Couples BIOMD0000000903 (COPASI) to a tyssue 2D sheet via a new flux-driven TumorCoupling process. Adds the tumor + baseline composites, a cell-type snapshots viz, and an investigation with two studies (tumor composite; tumor vs baseline). See docs/superpowers/specs/2026-06-12-tumor-tyssue-coupling-design.md.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 4: Report the draft PR URL to the user.**

---

## Notes / risks for the implementer

- **`scales` tuning** (Task 11 Step 5) is the main empirical knob — the ODE is stiff and multi-scale (healthy turnover ≫ tumor growth initially). Start from the defaults above, run 20–60 steps, and adjust so each process is visible without collapsing the mesh. Document the final values in the spec/study.
- **Composite loader / Composite construction:** reuse whatever `tests/test_composites.py` already uses to load `*.composite.yaml` and build a `Composite`; the snippets above name `pbg_superpowers.composite_spec.load_composite_spec` but the existing repo test is the source of truth — do not invent a loader.
- **runs.db observables:** `TimeSeriesFromObservables` reads top-level scalar keys from `history.state`. Confirm during Task 12 Step 4 that the dashboard's SQLite emitter records the `*_births/_deaths/_count` stores; if it only records declared `observables`, the study.yaml `observables` block (Task 12 Step 2) is what makes them appear — that's why each has a `store_path`.
- **COPASI inputs unwired:** verify the ODE advances with `Copasi.inputs: {}` (Task 11 Step 5). If concentrations are static, wire `species_concentrations` in *and* out of the `Species` store to round-trip state.
- **immune (I) / estrogen (E):** intentionally have no mesh cells — they still drive the ODE fluxes. Do not add mesh cell types for them.
</content>
