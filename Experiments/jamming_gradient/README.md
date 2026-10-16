# Cell-jamming & parameter-gradient migration experiments

Two single-cell migration scenarios on a flat epithelial square sheet, reproduced
from `Notebooks/simulation_walkthrough.ipynb` (specs 05 and 06) — same config, same
`tf`/`dt`, **no parameter sweep**. One cell (face 96) is given an `ActiveMigration`
drive and migrates through the tissue while stochastic line-tension fluctuations act on
every edge.

The drive is `MIGRATION_STRENGTH` = 0.6, not the walkthrough's 0.1. `ActiveMigration`
now splits the cell's strength between its vertices (vivatyssue `9007de9`); before, every
vertex carried the full strength. Face 96 is a hexagon, so 0.6 gives each vertex the
force it had before. With 0.1 the cell barely moves (final displacement 0.77).

- **Jamming** — `CellJamming` fires at `t = 300` (run to `t = 400`, `dt = 0.1`),
  ramping every cell's preferred perimeter down so the tissue solidifies and the
  migrating cell arrests.
- **Gradient** — `ParameterGradient` imposes a linear `prefered_perimeter`
  gradient along x (`m = -0.1`, `c = 4.6`; `dt = 0.05`, run to `t = 600` — the
  notebook's `t = 300` extended), so the migrating cell moves through a
  stiffness gradient.

Both scenarios set explicit rosette detachment rates through `auto_reconnect`
(`JAM_DETACH` / `GRAD_DETACH` in `jamming_gradient.py`) — see **Detachment
rates** below.

Both run on the **python** backend (`ActiveMigration` is not rust-supported).

## Detachment rates

`auto_reconnect` is a map, not a boolean: `{enabled, p_4, p_5p}`. `p_4` (rank-4
vertices) and `p_5p` (rank 5+) are rosette detachment probabilities **per unit
time** — tyssue multiplies both by `settings["dt"]`. EulerSolver now sets that
`dt` to the solver interval; before commit `2dcb8c9` (2026-08-31) nothing set it
and tyssue defaulted to `1.0`, so detachments effectively fired `1/dt` times more
often and the migrating cell slipped through its neighbours far more easily.

Both scenarios therefore restore those rates via `_july_detach_rates(dt)`, which
returns tyssue's own defaults divided by `dt`:

| scenario | `dt` | `p_4` | `p_5p` | effective `p_4` / `p_5p` |
|---|---|---|---|---|
| jamming  | 0.10 | 1.0 | 0.1 | 0.1 / 0.01 |
| gradient | 0.05 | 2.0 | 0.2 | 0.1 / 0.01 |

The nominal values differ only because the two scenarios integrate at different
`dt`; the *effective* per-step rate is identical, and matches July's. On the
jamming run this took the final displacement from 2.630 back to 9.194, against the
July `run.log` figure of 9.854. The current run ends at **8.08** (8.94 at its peak,
just after jamming at `t = 300`); the gradient run ends at **12.76**.

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

The gradient figure omits the preferred perimeter at `t = 0`. That used to hide a
spurious jump: `ParameterGradient` is a **step**, and the first recorded frame held the
pre-gradient default (3.6). The EulerSolver now records each frame after applying the
behaviors it received, so the first frame already carries the gradient (4.38 for the
migrating cell's starting x); the omission is harmless.

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
