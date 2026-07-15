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

## Outputs (git-ignored)

```
outputs/
  jamming/
    jamming.gif                    # faces by prefered_perimeter, migrating cell highlighted
    still_t*.png                   # snapshots across the run
    history.hf5                    # archived (thinned) simulation data
    migrating_trace.csv
    migrating_displacement.png     # displacement vs time + dotted jamming-transition line
    circularity_over_time.png      # mean cell circularity (4πA/P²) over time + jamming line
    circularity_over_time.csv
  gradient/
    gradient.gif
    still_t*.png
    history.hf5
    migrating_trace.csv
    migrating_displacement.png     # displacement vs time
    migrating_velocity_vs_x.png    # instantaneous velocity vs x-position
    prefered_perimeter_vs_time.png # migrating cell's preferred perimeter over time
    circularity_along_x.png        # mean cell circularity along x, pooled over all timepoints
    circularity_along_x.csv
```

## Run

```bash
conda activate vivarium-tyssue
cd Experiments/jamming_gradient
python jamming_gradient.py
```

Requires ImageMagick (`magick`) on PATH for GIF rendering. Nothing under `data/`
or `outputs/` is tracked by git — re-running the script regenerates everything.
`tf`/`dt` and the other constants are fixed at the top of `jamming_gradient.py` to
match the notebook.
