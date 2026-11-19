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

- **Jamming** — `CellJamming` fires at `t = 300` (run to `t = 400`, `dt = 0.1`) and
  lowers every cell's preferred perimeter by 0.5% per step (`rate = -0.05` per unit
  time, relative to the current value) until the mean reaches the lower limit, 3.0.
  The step that would cross the limit is shortened to land on it, so P₀ goes from 3.8
  to exactly 3.0 at `t = 304.8` and stays there. Runs before 2026-11-19 never stopped
  the ramp (its limit check was always true), so P₀ fell to 0.026 by `t = 400`.
- **Gradient** — `ParameterGradient` imposes a linear `prefered_perimeter`
  gradient along x (`m = -0.1`, `c = 4.6`; `dt = 0.05`, run to `t = 600` — the
  notebook's `t = 300` extended), so the migrating cell moves through a
  stiffness gradient. The tissue starts with the gradient in place: each cell's
  `prefered_perimeter` (and the gradient's share column) is set to `m*x + c` from its
  position in the input mesh, and the step then keeps it tied to the cell's current x.

Both scenarios set explicit rosette detachment rates through `auto_reconnect`
(`JAM_DETACH` / `GRAD_DETACH` in `jamming_gradient.py`) — see **Detachment
rates** below.

Both run on the **python** backend (`ActiveMigration` is not rust-supported).

## Initial relaxation

Each scenario starts from the input mesh **relaxed under its own initial parameters**,
so the run does not begin with a transient (for example in cell shape and circularity)
that comes from a mesh relaxed for different parameters. Before the run,
`relaxed_mesh()` in `jamming_gradient.py` minimises the energy with tyssue's
quasistatic solver (`QSSolver`, L-BFGS-B), using:

- **conservative forces only**: `LineTension`, `FaceAreaElasticity` and
  `PerimeterElasticity`. `ActiveMigration` is not conservative, and the OU tension
  noise has zero mean, so neither is included;
- **the run's own bound model** (`model_factory_bound`), so the 52 boundary vertices
  stay fixed;
- **no T1/T2 transitions**, so the topology is the input mesh's;
- **tight tolerances** (`gtol` 1e-8, `ftol` 1e-12). tyssue's defaults leave a
  residual force of about 1e-3.

For the **gradient**, P₀ = m·x + c depends on where the cells end up, so the
relaxation alternates relaxing and resetting P₀ at the new positions until P₀ stops
changing (6 rounds). The run then starts with P₀ = m·x + c at the relaxed positions.

| scenario | energy before → after | largest vertex shift | residual interior force |
|---|---|---|---|
| jamming (P₀ = 3.8) | 3.82 → 1.39 | 0.58 | < 1e-6 |
| gradient (P₀ = m·x + c) | 6.76 → 3.56 | 0.73 | < 1e-6 |

The relaxed meshes are cached as `data/relaxed_<scenario>.hf5`, with a JSON sidecar
holding the summary and a hash of the inputs (mesh contents, parameters, gradient,
solver settings). They are rebuilt automatically when any of those inputs change.

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
July `run.log` figure of 9.854. The current run, started from the relaxed mesh (see
**Initial relaxation**), ends at **7.82**. Jamming to P₀ = 3.0 slows the tissue
(10-unit ΔR_rms 0.131 before `t = 300`, 0.099 after) but does not arrest the migrating
cell, which is still moving at `t = 400`. With the unbounded ramp it ended at 6.44
(peak 7.11 just after `t = 300`). The gradient run ends at **14.09** (13.62 before `ParameterGradient` became
a Process on 2026-11-19: it now reads the start-of-tick mesh, like the other regulators,
instead of running as a Step after each solver tick). Started from the unrelaxed input
mesh, the same runs ended at 7.84 (peak 8.24) and 14.03.

These numbers come from the 2026-11-16 rerun, which uses per-junction line-tension
noise: one OU draw per junction, so each junction's tension has sd ≈ `SIGMA` (0.10
measured). Runs before 2026-10-27 drew one value per half-edge. That gave an effective
sd of `SIGMA/√2` (0.072 measured), and those runs ended at 8.08 (jamming) and 12.76
(gradient).

## Workflow — simulate, then analyse

Simulation and analysis are split so you can **re-analyse without re-simulating**:

1. **`jamming_gradient.py`** runs both scenarios and archives each one's `History`
   to `outputs/<jamming|gradient>/history.hf5` (compressed HDF5). No figures.
   `python jamming_gradient.py gradient` (or `jamming`) runs just that scenario.
2. **`gradient_seeds.py`** re-runs the gradient scenario for 10 seeds in parallel
   and keeps only the migrating cell's trace per seed
   (`outputs/gradient_seeds/seed_<seed>.csv`). The single archived run is one draw
   of the line-tension noise, so the migrating cell's circularity is reported as an
   ensemble mean instead (see below).
3. **`jamming_gradient_analysis.ipynb`** reopens the archives
   (`HistoryHdf5.from_archive`) and the seed traces, and produces all GIFs, stills,
   traces and plots.

## Outputs (git-ignored)

`history.hf5` is written by `jamming_gradient.py` and `gradient_seeds/` by
`gradient_seeds.py`; everything else by `jamming_gradient_analysis.ipynb`.

```
outputs/
  jamming/
    history.hf5                    # <- simulation (archived History)
    jamming.gif                    # faces by prefered_perimeter, migrating cell highlighted
    still_t*.png                   # snapshots across the run, with a P0 colorbar
    migrating_trace.csv
    circularity_over_time.csv
    displacement_circularity.png   # displacement + mean cell circularity (4πA/P²) vs time,
                                   #   twin axes + dotted jamming-transition line
    rms_cell_movement.png          # RMS cell-centre (area centroid) displacement of the 153
    rms_cell_movement.csv          #   interior cells: (a) over a 10-unit window vs time,
    rms_cell_movement_vs_lag.csv   #   (b) vs window length before / after jamming (caging)
  gradient/
    history.hf5                    # <- simulation (archived History)
    gradient.gif                   # P0 colour scale GRAD_PP_RANGE (2.8-4.6)
    still_t*.png                   # snapshots with a P0 colorbar
    migrating_trace.csv
    displacement_perimeter.png     # displacement + the migrating cell's preferred perimeter
                                   #   vs time, twin axes
    migrating_velocity_vs_x.png    # instantaneous velocity vs x-position
    circularity_along_x.png        # cell circularity along x, one-cell-wide lattice bins,
    circularity_along_x.csv        #   interior cells only (see below); per bin
    circularity_cells.csv          #   per cell: bin, starting x, time-mean circularity
    migrating_circularity_seeds.png  # migrating cell's circularity vs time: mean ± SD
    migrating_circularity_seeds.csv  #   (shaded) over the seed ensemble
  gradient_seeds/
    seed_<seed>.csv                # <- gradient_seeds.py: migrating-cell trace per seed
  bigraphs/
    jamming_bigraph.png            # the composite wiring per scenario (processes in
    gradient_bigraph.png           #   peach/pink, stores in light blue)
```

Stills, analysis figures and bigraphs render at 300 dpi; GIFs at 110 dpi.

**Circularity along x.** Cells are binned on the hexagonal lattice, not on a fixed
grid of x values. Rows are offset by half a cell, so cell centres sit on half-columns
0.538 apart. Each bin pairs two neighbouring half-columns: it is one cell wide (1.076),
holds one cell from each row, and its edges fall between half-columns, so no column is
split. Each cell is assigned once, by its column in the unrelaxed input mesh (the run
starts from a relaxed copy that is no longer exactly on the lattice), and contributes a
single value: its circularity averaged over the run. Each bin is plotted at its cells'
mean x at `t = 0`. Cells touching the tissue boundary at
`t = 0` are excluded, and so is the migrating cell, which has its own trace. That
leaves 14 bins of 10–11 cells.

## Run

```bash
conda activate vivarium-tyssue
cd Experiments/jamming_gradient

python jamming_gradient.py                     # 1) simulate -> outputs/*/history.hf5
python gradient_seeds.py                       # 2) gradient seed ensemble -> outputs/gradient_seeds/
jupyter lab jamming_gradient_analysis.ipynb    # 3) analyse & visualise (Run All)
```

The notebook requires ImageMagick (`magick`) on PATH for GIF rendering. Nothing
under `data/` or `outputs/` is tracked by git — re-running regenerates everything.
`tf`/`dt` and the archive frame cap are set at the top of `jamming_gradient.py`
(fixed to match the notebook specs); plot styling lives in the analysis notebook.

Each scenario draws from its own seed (`SEEDS` in `jamming_gradient.py`), applied
immediately before that run, so re-running one scenario reproduces its archive
without depending on the other having consumed the random stream first.
`gradient_seeds.py` uses seeds `SEEDS["gradient"] + k` for `k = 0..9` and pins the
BLAS threads to 1; without that pin, same-seed runs diverge through T1 flips, so a
single seed does not by itself reproduce a run.
