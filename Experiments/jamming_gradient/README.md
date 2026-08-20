# Cell-jamming & parameter-gradient migration experiments

Two single-cell migration scenarios on a flat epithelial square sheet, reproduced
exactly from `Notebooks/simulation_walkthrough.ipynb` (specs 05 and 06) — same
config, same `tf`/`dt`, **no parameter sweep**. One cell (face 96) is given an
`ActiveMigration` drive and migrates through the tissue while stochastic
line-tension fluctuations act on every edge.

- **Jamming** — `CellJamming` fires at `t = 300` (run to `t = 400`, `dt = 0.1`),
  ramping every cell's preferred perimeter down so the tissue solidifies and the
  migrating cell arrests.
- **Gradient** — `ParameterGradient` imposes a linear `prefered_perimeter`
  gradient along x (`m = -0.1`, `c = 4.6`; `dt = 0.05`, run to `t = 400` — the
  notebook's `t = 300` extended by 100), so the migrating cell moves through a
  stiffness gradient.

Both run on the **python** backend (`ActiveMigration` is not rust-supported).

## Workflow — simulate, then analyse

Simulation and analysis are split so you can **re-analyse without re-simulating**:

1. **`jamming_gradient.py`** runs both scenarios and archives each one's `History`
   to `outputs/<jamming|gradient>/history.hf5` (compressed HDF5). No figures.
2. **`jamming_gradient_analysis.ipynb`** reopens those archives
   (`HistoryHdf5.from_archive`) and produces all GIFs, stills, traces and plots.

## Outputs (git-ignored)

`history.hf5` is written by `jamming_gradient.py`; everything else by
`jamming_gradient_analysis.ipynb`.

```
outputs/
  jamming/
    history.hf5                    # <- simulation (archived History)
    jamming.gif                    # faces by prefered_perimeter, migrating cell highlighted
    still_t*.png                   # snapshots across the run
    migrating_trace.csv
    circularity_over_time.csv
    displacement_circularity.png   # displacement + mean cell circularity (4πA/P²) vs time,
                                   #   twin axes + dotted jamming-transition line
  gradient/
    history.hf5                    # <- simulation (archived History)
    gradient.gif
    still_t*.png
    migrating_trace.csv
    displacement_perimeter.png     # displacement + the migrating cell's preferred perimeter
                                   #   vs time, twin axes (perimeter skips t = 0, see below)
    migrating_velocity_vs_x.png    # instantaneous velocity vs x-position
    circularity_along_x.png        # mean cell circularity along x, pooled over all timepoints
    circularity_along_x.csv
  bigraphs/
    jamming_bigraph.png            # the composite wiring per scenario (processes in
    gradient_bigraph.png           #   peach/pink, stores in light blue)
```

Stills, analysis figures and bigraphs render at 300 dpi; GIFs at 110 dpi.

The gradient figure omits the preferred perimeter at `t = 0`: `ParameterGradient` is a
**step**, so the first recorded frame still holds the pre-gradient default (3.6) and
would render as a spurious jump before the gradient's ~4.31.

## Run

```bash
conda activate vivarium-tyssue
cd Experiments/jamming_gradient

python jamming_gradient.py                     # 1) simulate -> outputs/*/history.hf5
jupyter lab jamming_gradient_analysis.ipynb    # 2) analyse & visualise (Run All)
```

The notebook requires ImageMagick (`magick`) on PATH for GIF rendering. Nothing
under `data/` or `outputs/` is tracked by git — re-running regenerates everything.
`tf`/`dt` and the archive frame cap are set at the top of `jamming_gradient.py`
(fixed to match the notebook specs); plot styling lives in the analysis notebook.

Each scenario draws from its own seed (`SEEDS` in `jamming_gradient.py`), applied
immediately before that run, so re-running one scenario reproduces its archive
without depending on the other having consumed the random stream first.
