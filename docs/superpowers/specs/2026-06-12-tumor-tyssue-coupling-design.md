# Tumor population dynamics on a tyssue 2D epithelium — design

**Date:** 2026-06-12
**Status:** approved (brainstorming) → ready for implementation plan

## Goal

Couple a non-spatial SBML tumor-population model (BioModels `BIOMD0000000903`,
run in COPASI) to a tyssue 2D vertex-model epithelial sheet, so that each
timestep the SBML model's computed cell births and deaths are executed as
discrete division / apoptosis / differentiation events on the mesh. Deliver one
**investigation** with two **studies** and a **draft PR**.

### Honest framing

`BIOMD0000000903` is *"A fractional mathematical model of breast cancer
competition"* (Solís-Pérez et al., 2019) — a **non-spatial, deterministic ODE**
over 5 populations with counts in the **millions**. The tyssue mesh has ~tens of
cells. The coupling is therefore a **qualitative / proportional** mechanism, not
a quantitatively validated spatial tumor model. The studies demonstrate that the
mechanism works and produces the expected qualitative trends (tumor expansion,
healthy-cell displacement); they do **not** claim spatial-quantitative accuracy.

## The model (BIOMD0000000903)

Species (initial amounts): **C** cancer-stem (7.37e5), **T** tumor (7.62e6),
**H** healthy (2.5e7), **I** immune (0), **E** estrogen (0). Reactions used by
the coupling:

| Population | Birth reaction | Death reaction |
|---|---|---|
| tumor (T)  | `Induction_of_tumor_cell` | `Removal_of_tumor_cell_y_immune_cell_and_other_immune_response` |
| healthy (H)| `Increase_in_the_healthy_cell_in_the_system` | `Decrase_of_healthy_cell_due_to_cancer` |
| stem (C)   | `Formation_of_Stem_cell` | `Removal_of_stem_cell_from_the_system` |

Immune (**I**) and estrogen (**E**) remain SBML-internal drivers — they modulate
the fluxes above but have **no mesh cells** (no epithelial analog).

## Architecture

Three processes in the tumor composite, all at `dt = 1.0`:

```
CopasiUTCProcess(model_source=BIOMD0000000903.xml)      # local:CopasiUTCProcess (pbg-copasi)
   outputs: fluxes{reaction→rate}, species_concentrations{}, time
        │ fluxes
        ▼
TumorCoupling   (NEW: vivarium_tyssue/processes/tumor_coupling.py)
   inputs:  fluxes (COPASI), datasets (tyssue face_df → cell_uids per type)
   outputs: behaviors:list[node]; emits births{}, deaths{}, type_counts{}
        │ behaviors
        ▼
EulerSolver (tyssue)                                     # local:EulerSolver
   inputs:  behaviors:list[node]
   outputs: datasets (vert/edge/face dfs; face_df.cell_type)
```

`TumorCoupling` is structurally **"Gillespie, but driven by COPASI fluxes
instead of internal rates"** — it reads the tyssue datasets to find target cell
uids per type and emits the same behavior dicts the existing Gillespie process
emits (`divide_crypt`, `apoptosis_extrusion`, `differentiation`), which
`EulerSolver` executes via its `EventManager`.

### The coupling rule (proportional Δ + accumulator, flux-based)

Per step, for each mapped population `p ∈ {tumor, healthy, stem}`:

1. Read birth flux `b_p` and death flux `d_p` from the COPASI `fluxes` output.
2. `births_p  += b_p * mesh_scale * dt`  (separate fractional accumulator)
   `deaths_p  += d_p * mesh_scale * dt`
3. Fire `floor(births_p)` divisions and `floor(deaths_p)` apoptoses; subtract the
   fired integer parts back out of the accumulators (no event lost to rounding).
4. **Cap** fired events to the number of available cells of that type so the mesh
   stays bounded; report any capped/dropped events via `log` (no silent caps).

Reading **fluxes** (not net species delta) keeps births and deaths **separate**,
which directly feeds the "births over time" / "deaths over time" visualizations.

- **C→T differentiation:** a tumor birth is realized by **differentiating a stem
  cell** (`differentiation` behavior, `new_type="tumor"`) when a stem cell is
  available; otherwise by dividing an existing tumor cell. This honors "stem
  cells source tumor cells" and solves the first-tumor-cell seeding problem.
- **Births** → `divide_crypt` (tyssue `cell_division`) on a chosen same-type cell.
- **Deaths** → `apoptosis_extrusion` (tyssue `remove_face`) on a chosen same-type cell.
- **Target selection:** pick the cell uid to divide/kill from the matching-type
  cells in `face_df` (uniform or area-weighted; uniform for v1).
- **`mesh_scale`** is a config param, default tuned so a 60–100 step run yields a
  watchable cadence of events; exposed for tuning per study.
- **One-directional** (SBML → mesh). Mesh counts are **not** fed back into the
  ODE (YAGNI). The `species_counts` input port on `CopasiUTCProcess` is the future
  hook if bidirectional coupling is wanted later.

### Initial mesh

Flat 2D `SheetGeometry` square (reuse `test_square.hf5` / the existing
`stochastic`-style 2D sheet init). Seed `face_df.cell_type` mostly **healthy**
with a small **tumor** focus and a few **stem** cells (early-tumor state).

## Study 1 — Tumor composite (Design → Build → Simulate → Evaluate)

Build the coupled composite, run it, and produce five visualizations
(`vivarium_tyssue/visualizations/`):

1. **Snapshots over time** — multi-panel static figure of the mesh at selected
   timepoints, colored by `cell_type`. *(new)*
2. **Gif animation** — `TissueSheetGif` (2D), with `CELL_TYPE_COLORS` extended for
   `healthy` / `tumor` / `stem`. *(extend existing)*
3. **Cell deaths over time** — timeseries from adapter-emitted `deaths{}`. *(new)*
4. **Cell births over time** — timeseries from adapter-emitted `births{}`. *(new)*
5. **Cell types over time** — composition timeseries (count per type). *(new)*

Expected qualitative finding: tumor (and stem-derived tumor) cells expand while
healthy cells are displaced; births/deaths timeseries track the SBML fluxes.

## Study 2 — Tumor vs baseline comparison

**Baseline:** the same flat 2D sheet with **mechanics only** (no COPASI, no tumor
behaviors) — pure epithelial relaxation. Clean control: identical geometry,
tumor coupling on vs off.

Compare on four metrics (drive the verdict + charts):

1. **Cell-type composition over time** — baseline stays ~uniform; tumor composite
   shifts toward tumor.
2. **Total cell count + cumulative births/deaths** — divergent growth + healthy loss.
3. **Tissue morphology / area** — does tumor growth distort the sheet vs the
   relaxed baseline?
4. **Healthy-cell survival** — H trajectory; quantifies displacement of normal tissue.

## Plumbing

- **Dependencies:** add `pbg-copasi` and `pbg-biomodels` to `pyproject.toml` as
  `git+https` direct references (matching the existing `vivarium-dashboard` /
  `pbg-emitters` pattern); refresh `uv.lock`.
- **Model caching:** fetch once via `pbg_biomodels.load_biomodel("BIOMD0000000903")`
  and commit the SBML to `workspace/datasets/BIOMD0000000903.xml` so runs are
  reproducible and offline.
- **Registration:** register `TumorCoupling` and the new Visualization classes in
  `vivarium_tyssue/core.py` (and the relevant `maps`), so bigraph-schema
  auto-discovery and the dashboard pick them up.
- **Composites:** `vivarium_tyssue/composites/tumor.composite.yaml` (coupled) and a
  baseline 2D-sheet composite for the comparison study.
- **Investigation + studies:** authored via the spine (`study.yaml`, v4 schema);
  regenerate reports with `/pbg-report`.

## Testing

- **Unit (adapter):** accumulator math (flux → integer events), sign handling,
  capping to available cells, seeding tumor via differentiation when no tumor cell
  exists, no-event-lost-to-rounding.
- **Integration:** short tumor-composite run (a few steps) asserting behaviors fire
  and `cell_type` counts shift in the expected direction; baseline run produces no
  births/deaths and a stable composition.

## Deliverable

Feature branch `feat/tumor-tyssue-investigation` → **draft PR**. (PR base —
`vivarium-tyssue` working branch vs `master` — to be confirmed at PR time.)

## Implementation note — environment constraint (added during execution)

The COPASI/SBML stack hard-forces `numpy~=2.2` (libroadrunner) and `pandas>=3.0`
(pint-pandas). tyssue 1.1.0 — the latest release — is incompatible with those for
its **topology operations**: `cell_division` and `remove_face` (the Gillespie
behaviors we planned to use for births/deaths) crash under pandas 3.0 / numpy 2.x
(read-only arrays, multi-column assignment). Downpinning is impossible (the SBML
deps forbid it) and no pandas-3-compatible tyssue exists.

Resolution (approved): the coupling defaults to a **fixed-topology cell-fate**
model rather than physical division/extrusion. `TumorCoupling` drives cell-type
changes on a mesh of constant topology — a birth relabels a source cell into the
target type (stem→tumor realizes C→T; healthy→stem; dead→healthy regenerates),
a death relabels a cell to `dead`. Composition still evolves under SBML control
and every visualization (snapshots, gif, births/deaths/composition timeseries)
works off `cell_type`. A `topology_ops: true` config flag selects the real
`divide_crypt`/`apoptosis_extrusion` behaviors for when a pandas-3-compatible
tyssue is available (covered by a unit test). Compatibility shims also fixed three
other tyssue/numpy-2.x/pandas-3.0 issues (StringDtype schema coercion in
EulerSolver, read-only `shuffle` in `merge_vertices`). The default interval is
`0.1` (larger steps overflow the vertex mechanics).

This keeps the **honest framing** above intact and arguably sharpens it: the demo
is a qualitative SBML-driven cell-fate model on an epithelial sheet, not a
quantitatively validated spatial vertex-model tumor.

## Decisions locked in brainstorming

- Coupling rule: **proportional Δ + fractional accumulator**, flux-based (births
  and deaths read separately).
- Mesh cell types: **tumor + healthy + stem**; immune + estrogen off-mesh.
- Tumor births realized by **differentiating stem cells** when available.
- **One-directional** coupling (no mesh→ODE feedback in v1).
- Baseline: flat 2D sheet, **mechanics only**.
- Comparison metrics: composition, count + births/deaths, morphology/area,
  healthy-cell survival (all four).
