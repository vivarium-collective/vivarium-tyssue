# Discrete-event experiments

Three discrete-event scenarios exercising the rate-based `CellDivisions` /
`CellDeaths` processes and the Gillespie model.

- **divisions** — a `CellDivisions` process fires cell divisions as a **Poisson
  process** (rate-based random times) on a plain **flat square sheet**
  (`test_square.hf5`, `SheetGeometry`) — the same tissue the other Experiments use.
  There are no biological cell types here: every cell starts `normal` (grey) and a
  cell being actively grown toward division is highlighted `dividing` (magenta).
  This simply demonstrates that random divisions work. 2-D GIF + stills.
- **deaths** — a `CellDeaths` process fires apoptotic extrusions as a Poisson
  process on the same flat square, using the same `apoptosis_extrusion` behaviour
  the Gillespie process drives; a dying cell is highlighted `extruding` (black).
  2-D GIF + stills.
- **gillespie** — the full Gillespie biochemistry (`Gillespie` process) on the 3-D
  **crypt cylinder** (`crypt_cylinder.hf5`, `VesselGeometry`) exactly as in
  `tests/tests.py` and `Notebooks/simulation_walkthrough.ipynb` (`tf=72`,
  `dt=0.005`). Cells are drawn with the `CELL_TYPE_COLORS` palette; three analyses
  (see below).

`CellDivisions` is the renamed, rate-based successor of the old `TestRegulations`
process; `CellDeaths` is its death counterpart. Both live in
`vivarium_tyssue/processes/regulations.py`.

## Workflow — simulate, then analyse

Simulation and analysis are split into two steps so you can **re-analyse without
re-simulating** (and re-simulate without disturbing earlier analysis):

1. **`discrete_events.py`** runs the simulations and archives each scenario's
   `History` to `outputs/<scenario>/history.hf5` (compressed HDF5), plus — for
   gillespie — the emitted discrete events to `outputs/gillespie/events.csv`
   (events come from the process emitter, not the History). It produces **no**
   figures or videos.
2. **`discrete_events_analysis.ipynb`** reopens those archives
   (`HistoryHdf5.from_archive`) and produces all GIFs, stills and analyses.

## Run

Use the repo's `vivarium-tyssue` conda env (the `.venv`/`uv` are broken). The
notebook needs ImageMagick (`magick`) on PATH for GIF rendering.

```bash
conda activate vivarium-tyssue
cd Experiments/discrete_events

# 1) simulate (writes outputs/<scenario>/history.hf5 [+ gillespie/events.csv])
python discrete_events.py            # all three scenarios
python discrete_events.py divisions  # or a single one: divisions | deaths | gillespie

# 2) analyse & visualise
jupyter lab discrete_events_analysis.ipynb   # then Run All
```

The three scenarios are independent, so the simulations can be launched as separate
processes to run concurrently. The notebook skips any scenario whose `history.hf5`
is missing, so it works on whichever subset you have simulated.

## Gillespie analyses

1. **cell-type distribution over time** — stacked cell counts per type at every
   recorded timepoint (`cell_type_over_time.png` / `.csv`).
2. **cell-type spatial distribution along z** — a line per cell type giving mean
   per-frame occupancy, finely binned along the crypt's z axis
   (`cell_type_along_z.png` / `.csv`).
3. **event-type spatial distribution along z** — division / differentiation /
   extrusion event counts binned along z, each event placed at the mean z of its
   cell (`events_along_z.png` / `.csv`, raw events in `events.csv`).

All three analysis figures place their legend outside the plot area.

## Outputs (git-ignored)

Written by `discrete_events.py` (`history.hf5`, `events.csv`); everything else is
written by `discrete_events_analysis.ipynb`.

```
data/test_square.hf5        # flat-sheet mesh (divisions / deaths)
data/crypt_cylinder.hf5     # crypt mesh (gillespie); both copied on first run
outputs/
  divisions/   history.hf5                    # <- simulation (archived History)
               divisions.gif   still_t*.png   # <- analysis notebook (flat 2-D)
  deaths/      history.hf5
               deaths.gif      still_t*.png   # <- analysis notebook (flat 2-D)
  gillespie/   history.hf5     events.csv     # <- simulation
               gillespie.gif   still_t*.png   # <- analysis notebook (crypt 3-D)
               cell_type_over_time.png / .csv
               cell_type_along_z.png   / .csv
               events_along_z.png      / .csv
```

Stills and analysis figures render at 300 dpi; GIFs at 120 dpi. Nothing under
`data/` or `outputs/` is tracked by git — re-running regenerates everything.
Tune rates, `tf`/`dt` and archive frame caps via the constants at the top of
`discrete_events.py`; z-binning and plot styling live in the analysis notebook.
