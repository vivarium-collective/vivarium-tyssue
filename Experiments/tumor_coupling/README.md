# Tumor-coupling experiment

A non-spatial breast-cancer **population ODE** (BioModels `BIOMD0000000903`,
integrated in COPASI) coupled to a flat 2-D tyssue epithelial sheet
(`test_square.hf5`, `SheetGeometry`) through the `TumorCoupling` process.

Each step the process reads the SBML model's per-reaction birth/death fluxes and
fires `floor(flux · scale · dt)` discrete vertex-model events on the mesh:

- **births** → real `cell_division` (the cell splits, the mesh gains a face),
- **deaths** → real `apoptosis_extrusion` (the cell shrinks and is removed),
- **tumor induction** → `differentiation` of a cancer stem cell into a tumor cell.

A compact **cancer-stem-cell** focus is seeded at the sheet centre, matching the SBML
model's nonzero initial stem-cell population. The stem cells self-renew into a
persistent central core and commit their first cell to tumor; the tumor then grows
outward into one contiguous clone as the coupled fluxes drive divisions, while healthy
cells are progressively displaced. This mirrors
`vivarium_tyssue/composites/tumor.composite.yaml` and the `get_test_tumor_*`
helpers in `tests/tests.py` / `Notebooks/simulation_walkthrough.ipynb` (§9).

Cells are coloured by `cell_type`: **healthy = blue**, **tumor = red**,
**stem = purple**, transient **dividing = yellow** / **extruding = black**.

The experiment is split into a **simulation** script and an **analysis** notebook,
like the `discrete_events` experiment:

- **`tumor_coupling.py`** — runs the coupled simulation and archives the tyssue
  `History` (thinned to `TUMOR_ARCHIVE_FRAMES`) to `outputs/history.hf5`.
- **`tumor_coupling_analysis.ipynb`** — reopens that archive with
  `HistoryHdf5.from_archive` and produces the GIF, stills, and all analyses. Its
  first markdown cells explain the coupling with the governing equations. Re-analyse
  without re-simulating.

## Run

Use the repo's `vivarium-tyssue` conda env (the `.venv`/`uv` are broken). Needs
ImageMagick (`magick`) on PATH for GIF rendering, and the pandas-3-compatible
forked tyssue for the real topology ops (`topology_ops=True`).

```bash
conda activate vivarium-tyssue
cd Experiments/tumor_coupling
python tumor_coupling.py                     # 1. simulate -> outputs/history.hf5
jupyter nbconvert --execute --to notebook \
    --inplace tumor_coupling_analysis.ipynb  # 2. analyse (or open it interactively)
```

`TUMOR_TF` (default 300) is elapsed **global (tyssue) time**; the coupling steps at
`TUMOR_DT` (0.01). Every knob — the flux→event `SCALES`, `COPASI_TIME`, the
reaction-key maps, the seed focus, `division_crit` / `apoptosis_crit`, `TUMOR_TF` /
`TUMOR_DT` — is a module-level constant at the top of `tumor_coupling.py`, and the
notebook imports them so the simulation config stays the single source of truth.

### Timescale coupling (why divisions used to overlap)

Divisions used to inflate ~14× faster than the vertex-model sheet could relax
(τ_mech ≈ 0.36 tyssue-time units, measured by `calibrate_timescale.py`), so
freshly-split cells overlapped. The fix separates the timescales while keeping the
mechanics untouched:

- **Gradual inflation (the key fix)** — a committed cell now grows at the *real*
  solver step (fixed in `processes/tumor_coupling.py`), so `GROWTH_RATE` is a true
  per-tyssue-time rate. `GROWTH_RATE = 0.5` inflates a cell over
  `τ_grow = ln2/0.5 ≈ 1.4` units (~4× τ_mech): slow enough that the sheet fully
  relaxes around each division — the minimum face area stays above its t=0 value, so
  no cells overlap — while the compact seed keeps the clone growing. (The old `0.1`
  with `dt=1.0` ballooned a cell in ~0.07 units — ~5× faster than the sheet relaxes.)
- **Fewer, slower divisions** — the `SCALES` set the discrete-event rate while
  keeping the birth:death balance; `TUMOR_TF` is long enough to reach a clear takeover.
- **Cancer-stem-cell seed** — the focus is **6** cancer stem cells (one central
  patch), like the SBML model's nonzero initial stem population. They self-renew into
  a persistent core (`stem_births` > `stem_deaths`, unlike the raw SBML flux ratio,
  which would wipe out the handful of seeded cells) and commit their first cell to
  tumor, so the clone always has an eligible cell to grow from.

`COPASI_TIME` (α) — the tumor-model clock relative to tyssue time — is kept at
**1.0**; α<1 slows the tumor ODE but also delays its induction flux, which starves
the seed. Run `python calibrate_timescale.py` to re-measure τ_mech / the division
rate.

## Outputs

Everything lands under `outputs/` (git-ignored). The simulation writes the archive;
the notebook writes the rest:

- **`history.hf5`** — the archived tyssue `History` (compressed, thinned to
  `TUMOR_ARCHIVE_FRAMES` frames), written by `tumor_coupling.py`. Everything below is
  produced by `tumor_coupling_analysis.ipynb` from this archive.
- **`tumor.gif`** — 2-D animation of the sheet, faces coloured by cell type; watch
  the tissue gain cells (divisions) and one central tumor clone spread outward.
- **`still_t*.png`** — six evenly spaced stills, same renderer as the gif.
- **`population_over_time.png`** — tumor vs healthy (and stem/dead) cell counts,
  plus total tissue size, over time — the requested population analysis.
- **`population_over_time.csv`** — the per-frame counts behind that plot.
- **`tumor_area_over_time.png`** — total tumor **size**: the summed face area of all
  tumor cells over time (with total tissue area and the tumor area fraction). This
  captures both proliferation and per-cell growth, complementing the cell counts.
- **`tumor_area_over_time.csv`** — per-frame tumor area, total area, and fraction.
- **`face_area_floor_over_time.png`** — the minimum / 5th-percentile face area over
  time: the **overlap diagnostic**. Overlap shows up as faces collapsing toward zero
  area; with the timescale coupling these floors stay comfortably above 0.
- **`face_area_floor_over_time.csv`** — per-frame min / 5th-pct / median face area.
- **`bigraphs/tumor_coupling_bigraph.png`** — the composite wiring (`Tyssue` +
  `TumorCoupling` over the shared stores), drawn with `bigraph_viz`: processes in
  peach/pink, stores in light blue. The 3-D run writes
  `bigraphs/tumor_coupling_3d_bigraph.png` alongside it.

Stills, analysis figures and bigraphs render at 300 dpi; the GIF at 120 dpi.

Counts are taken from the tyssue mesh history (not the coupling's scalar `*_count`
stores, which accumulate additively across steps — a process-bigraph `map[float]`
quirk), so they are the true instantaneous per-frame counts.

## 3D monolayer variant

The same coupling on a **3-D monolayer** (`monolayer_box.hf5`, `MonolayerGeometry`):
births are real volumetric divisions (a cell grows its preferred **volume** until
`vol > division_crit`, then splits) with a randomised division-plane orientation.

Unlike the flat sheet, a confluent 3-D monolayer is **volumetrically jammed** — cells
can't freely enlarge, so many committed cells never reach their division volume. In
vertex-model rigidity theory the tissue is solid below a preferred 3-D shape index
`s0 = S0·V0^(-2/3) ≈ 5.4` and fluid above it, where cells rearrange (T1 swaps) and
divisions find room (Azote & Manning, *3D vertex models of stratified epithelia*,
2025; cancer tissue is the fluid side, Grosser++ PRX 2020). To **allow growth** the 3-D
run nudges the basal layer toward the fluid regime with a small T1 bump
(`threshold_length` 0.03 → 0.05; tyssue's monolayer topology corrupts above ~0.1) and
switches **necrotic death off** so the clone isn't eroded. The tumor then grows by
real divisions and **holds at a stable, mechanics-limited size** — a genuine 3-D mass
that saturates, in contrast to the non-spatial ODE, whose tumor grows without bound.
Real monolayer division is numerically fragile, so this is a modest (few-division)
clone rather than the large clone the 2-D sheet reaches. Split like the 2-D
experiment, into a simulation script + notebooks:

- **`tumor_coupling_3d.py`** — runs the 3-D simulation and archives the apical tissue
  surface to `outputs/monolayer_tumor_history.hf5`, plus the live internal-SBML
  trajectory to `outputs/sbml_population_3d.csv`.
- **`tumor_coupling_3d_analysis.ipynb`** — the matplotlib analyses (a 3-D
  apical-surface still, cell population over time, and the internal-SBML-model vs
  tissue comparison), with the governing equations in its opening markdown.
- **`tumor_coupling_3d_viz.ipynb`** — an interactive **ipyvolume** 3-D viewer of the
  archived tissue.

```bash
python tumor_coupling_3d.py                 # simulate -> archive + sbml csv
# then run either notebook against the archive
TUMOR3D_TF=2 TUMOR3D_DT=0.5 python tumor_coupling_3d.py   # quick smoke run
```
