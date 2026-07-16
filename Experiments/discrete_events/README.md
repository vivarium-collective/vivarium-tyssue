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

## Run

Use the repo's `vivarium-tyssue` conda env (the `.venv`/`uv` are broken). Needs
ImageMagick (`magick`) on PATH for GIF rendering.

```bash
conda activate vivarium-tyssue
cd Experiments/discrete_events
python discrete_events.py            # all three scenarios
python discrete_events.py divisions  # or a single one: divisions | deaths | gillespie
```

The three scenarios are independent, so they can be launched as separate processes
to run concurrently.

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

```
data/test_square.hf5        # flat-sheet mesh (divisions / deaths)
data/crypt_cylinder.hf5     # crypt mesh (gillespie); both copied on first run
outputs/
  divisions/   divisions.gif   still_t*.png   # flat 2-D
  deaths/      deaths.gif      still_t*.png   # flat 2-D
  gillespie/   gillespie.gif   still_t*.png   # crypt 3-D
               cell_type_over_time.png / .csv
               cell_type_along_z.png   / .csv
               events_along_z.png      / .csv
               events.csv
```

Stills and analysis figures render at 300 dpi; GIFs at 120 dpi. Nothing under
`data/` or `outputs/` is tracked by git — re-running regenerates everything.
Tune rates, `tf`/`dt` and the z-binning via the constants at the top of
`discrete_events.py`.
