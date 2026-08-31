# Discrete-event experiments

Four discrete-event scenarios exercising the rate-based `CellDivisions` /
`CellDeaths` processes and the Gillespie model.

- **divisions** — a `CellDivisions` process fires cell divisions as a **Poisson
  process** (rate-based random times) on a plain **flat square sheet**
  (`test_square.hf5`, `SheetGeometry`) — the same tissue the other Experiments use.
  There are no biological cell types here: every cell starts `normal` (grey) and a
  cell being actively grown toward division is highlighted `dividing` (magenta).
  This simply demonstrates that random divisions work; it runs for `tf=50`.
  2-D GIF only (the animation is the result — no stills).
- **deaths** — a `CellDeaths` process fires apoptotic extrusions as a Poisson
  process on the same flat square, using the same `apoptosis_extrusion` behaviour
  the Gillespie process drives; a dying cell is highlighted `extruding` (black);
  it runs for `tf=50`. A cell is removed once it has shrunk past `DEATH_CRIT` = 0.2,
  a fifth of the preferred area — which needs two settings beyond the threshold
  itself, both per-commitment so the crypt is unaffected: `DEATH_FLOOR` = 0.01 (the
  default 0.5 floor on the shrinking target area otherwise removes cells while they
  are still near full size) and `DEATH_CONTRACT` (a cell holding its full preferred
  perimeter stalls near area 0.35 and never reaches the threshold at all). 2-D GIF
  only (no stills). Each flat-sheet GIF's legend lists only that scenario's own
  states, so `dividing` never appears on the deaths animation, nor `extruding` on
  the divisions one.
- **gillespie** — the full Gillespie biochemistry (`Gillespie` process) on the 3-D
  **crypt cylinder** (`crypt_cylinder.hf5`, `VesselGeometry`) as in `tests/tests.py`
  and `Notebooks/simulation_walkthrough.ipynb`, run to `tf=72` with a mechanics step
  of `GILL_SOLVER_DT` = 0.001 (see *Time steps* below). Cells are drawn with the
  `CELL_TYPE_COLORS` palette; two analyses (see below).
- **gillespie_restart** — the **same model with the same parameters and the same
  step**, started from the settled tissue the `gillespie` run *ends* on rather than
  from the stock mesh. The stock `crypt_cylinder.hf5` has its cell types assigned
  from a spatial prior rather than produced by the dynamics, so the first run spends
  much of its window relaxing into the crypt's stationary composition; the restart
  opens at that composition, which is what makes the stationary state itself (and
  its fluctuations) measurable. `gillespie` therefore checkpoints its final
  epithelium to `outputs/gillespie/stable_eptm.hf5` — a plain tyssue mesh — and
  `gillespie_restart` loads it as its `eptm`. Set `GILL_STABLE_TIME` to a float to
  checkpoint that recorded timepoint instead of the final state.

  A checkpoint cannot carry a *pending* commitment: the queue that ramps a
  committed cell's preferred area lives on the solver's `EventManager`, which is
  not persisted, so a cell left flagged `dividing` / `extruding` would keep that
  label forever and never be picked by the Gillespie again (neither label is a real
  `cell_type`). `settle_pending_commitments` therefore restores each such cell's
  real type from its `commit_type` column and clears the flags before the mesh is
  written; the partly grown/shrunk `prefered_area` needs no fixing, since
  `EulerSolver` re-applies the configured face parameters on load. (Measured: 24
  cells were mid-commitment at the `gillespie` checkpoint, 41 at the restart's.)

  `gillespie_restart` writes a checkpoint of its own, so the chain can be extended
  by pointing a further run at it. That is worth doing if you need a strictly
  stationary start: `gillespie`'s `pc` population is still climbing at `tf=72`
  (26 → 76 → 124 → 147 over the second half), so its checkpoint sits close to but
  not exactly on the stationary composition, and the restart spends its first ~20
  time units relaxing `pc` 160 → ~100 before going flat for the remaining ~50.

`CellDivisions` is the renamed, rate-based successor of the old `TestRegulations`
process; `CellDeaths` is its death counterpart. Both live in
`vivarium_tyssue/processes/regulations.py`.

## Time steps

The crypt runs on **two independent clocks**, and only one of them is the solver's:

- **Mechanics** — `GILL_SOLVER_DT` = 0.001 is the `EulerSolver`'s interval, i.e. the
  explicit-Euler step. It was lowered from 0.005 because an explicit-Euler
  displacement is linear in `dt`: at 0.005 the transient gradient that follows a
  division or extrusion could displace a single vertex far enough in one step to
  push it off the tube and let neighbouring faces overlap (visible as a spike at
  the crypt mouth in the old `t=72` still).
- **Events** — the `Gillespie` process overrides `calculate_timestep`, which
  `process_bigraph` consults before each of its updates, and returns a true SSA
  waiting time `-ln(u) / Σ rate_max` (≈0.004 here). Its spec `interval`
  (`GILL_EVENT_DT`) is only a placeholder the scheduler replaces immediately. So the
  event statistics come from the rates alone and **do not change with the solver
  step** — measured over the same window, 274 events at `dt`=0.005 vs 260 at
  `dt`=0.001, same mix.
- **The one coupling** is the `Gillespie`'s `global_interval`, handed to the
  division / extrusion behaviours as their per-step `dt`. The committed-cell grower
  runs once per *solver* step, so `global_interval` tracks `GILL_SOLVER_DT` and a
  committed cell's growth per unit time is unchanged.

The tyssue `History` snapshots the whole mesh on every solver step and holds it in
RAM, so the 5× finer step is paired with `GILL_RECORD_EVERY` = 5 (set on the
`History`'s `save_every`/`dt`): recording cadence and memory footprint stay exactly
what they were at `dt`=0.005.

## Mechanics, and the folding it does not fully cure

Some cells become genuinely self-intersecting ("bow-tie"): the boundary crosses
itself and never recovers. This is **not** visible in the usual diagnostics —
tyssue's `face_df.area` sums *absolute* sub-triangle areas
(`sub_area = ‖n‖/2`, `sheet_geometry.py:52-56`), so a folded polygon scores the
same as a clean one and `FaceAreaElasticity` feels no restoring force.

It is tempting to read this as the tube buckling under post-division crowding,
since `VesselSurfaceElasticity` pins the sheet radially at `prefered_radius` = 2.5.
**That is not the cause** — the same folding occurs in the flat 2-D
`tumor_coupling` sheet, which has no radial pin. `vessel_elasticity` helps (below)
by suppressing cell *elongation*, which is the actual precondition, not by
addressing crowding.

Two parameters were swept against it at `tf`=45, scoring *integrated folded
cell-frames* (`folded_frames` alone saturates once folding begins and hides the
differences):

| `viscosity` | `vessel_elasticity` | cell-frames | worst cells/frame |
|-------------|---------------------|-------------|-------------------|
| 0.05        | 1                   | 28.5k       | 61                |
| 5           | 1                   | 20.6k       | 55                |
| 5           | 5                   | 18.3k       | 45                |
| **0.05**    | **10**              | **7.2k**    | **14**            |
| 0.05        | 20                  | 6.9k        | 15                |

`GILL_VESSEL_ELASTICITY` = 10.0 is the setting kept: a 4× reduction, saturating
there, with divisions (93) and daughter relaxation to `prefered_area` (1.001)
unaffected — the constraint is stiffer, the tissue is not frozen. `GILL_VISCOSITY`
stays at 0.05 rather than the composite's 5.0, which measured *worse* at matched
`vessel_elasticity`: since `ṙ = −∇E / viscosity` while the event rates are fixed in
absolute time, raising it slows the relief of post-division crowding 100× and the
crowding folds the sheet instead of relaxing out of it.

**Root cause.** Nothing in the model resists a face passing through itself. The
area term is blind to it by construction (unsigned, above); `reconnect` only reacts
to *short edges* and never fires here (measured: 24/24 events had no edge below
`threshold_length`); tyssue's actual defence, `collisions/auto_collisions`, is wired
only into `QSSolver` — `EulerSolver` deprecates `with_t1`/`with_t3` to no-ops
(`solvers/viscous.py:72-76`) — and its CGAL backend `tyssue._collisions` is not
built here. `validate()` is purely combinatorial and a bow-tie passes it. The
precondition is cell **elongation**.

**How to measure it.** Do **not** use `|A_vec|/A_fan`: that is a *non-convexity*
screen, not a self-intersection test (verified — face uid 87 at `t`=10.96 scores
0.9802 with no crossing). Two traps make hand-rolled tests silently wrong:

* the ring must be chained `srce→trgt`; archived row order is **not** polygon order;
* the projection plane must be the ring's **best-fit** plane, not its Newell normal.
  A symmetric bow-tie's lobes cancel exactly, so its Newell vector is **zero** and
  the frame is undefined — blind on the very case that matters most.

Two equivalent criteria exist, and they name different participants: an exact
segment-crossing test names the cell that **folded**, while "a vertex not on this
face's ring lies inside it" names the neighbour it **invaded**. The sets are
disjoint by construction, and every intrusion measured (62/62) traced back to a
folded parent face.

**Run-to-run spread is large — single runs cannot resolve small differences.** Two
`gillespie` runs at identical settings ended with **2** and **10** self-intersecting
faces on the final mesh. The crypt is not reproducible even under `np.random.seed`
(the `EventManager` uses an unseeded `random.shuffle`), so compare aggregates over
repeats, never one run against another.

**Folding is reduced, not eliminated** — it persists, and a checkpoint carries the
folded geometry forward, so `gillespie_restart` starts already folded at `t`=0.

### Commitment ramp — why `GILL_GROWTH_RATE` is 0.25, not 0.045

At `GILL_GROWTH_RATE` = 0.045 both crypt runs produced a **division burst**: nothing
divided at all for the first 25 time units, then the whole backlog discharged as one
wave. The cause is a queue that swallows its own pool. Only `sc` and `pc` can emit a
division (`jump == cell_type`), which is 113 of 704 cells at `t`=0. A capable cell
waits ~9 `t` to be picked but then takes ~24 `t` to ramp `prefered_area` to
`GILL_DIVISION_CRIT`, so nearly all of them commit before any divides:

| | free sc+pc | `dividing` | % of the capable pool locked |
|---|---|---|---|
| `t` = 0 | 113 | 0 | 0% |
| `t` = 25 | 24 | **83** | **78%** |
| `t` = 40 | 143 | 15 | 9% |

Commits then stop for want of candidates — not because the rate falls. The
controlling number is (ramp) × (per-cell commit rate), ≈ 2.6 at 0.045; it wants to be
below ~0.3.

Swept 0.045 / 0.10 / 0.15 / 0.25 / 0.35 / 0.50 / 0.75 at `tf`=40, then **5 repeats
each at 0.25 and 0.50** — the crypt is not reproducible run to run, so single runs
cannot separate them. At 0.25 the burst is gone: first birth at `t`=5 instead of
`t`=25, peak `dividing` 28.4 ± 3.6 instead of 83, free `sc`+`pc` never below
82.8 ± 4.3 instead of 24. Throughput is unchanged (77.6 ± 3.0 births vs 72) — only
the timing moves — and there was **zero mesh corruption** in any of the 10 runs
(no open rings, orphan faces, sub-3-sided faces, degenerate areas, duplicated
`unique_id`s or non-finite coordinates).

0.25 and 0.50 are statistically **indistinguishable** on folding (peak
self-intersecting faces 2.0 ± 1.6 vs 4.2 ± 1.9; Welch *p*=0.085, Mann-Whitney
*p*=0.092), on rosettes, elongation and throughput. They separate only on queue
occupancy (peak `dividing` 28.4 vs 13.8, *p*<0.001). 0.25 keeps ~25% of the
division-capable pool in cycle, which reads as a crypt; 0.50 keeps ~12%. Faster is
not automatically better: 0.75 was as bad as the original (7 crossings, rank-7
rosettes).

Two things were tested and rejected. Making `prefered_perimeter` consistent with
`prefered_area` (3.5 → 3.7224, the regular-hexagon floor) barely touched the burst
and is **actively destructive** — retested at `tf`=72, it gave 67 self-intersecting
faces against 2, 406 invaded vertices against 2, and 387 overlapping face pairs
against 1. The sub-floor target is load-bearing: it acts as a constant inward
tension that both keeps cells compact (mean shape index 3.815 → 3.990, 7× as many
cells past 4.5) and throttles division via the ~0.884 equilibrium area (divisions
+61%; the restart leg ran away to 1674 events and was OOM-killed). See the comment
on `GILL_PREF_PERIMETER`. Starting from a stationary-composition mesh
(`crypt_cylinder_stable.hf5`, see `scripts/gen_crypt_stable.py`) fixes the initial
*composition* but not the transient, because the transient is the commitment queue,
not a composition mismatch. That mesh is still generated and available, but no
scenario uses it.

## Workflow — simulate, then analyse

Simulation and analysis are split into two steps so you can **re-analyse without
re-simulating** (and re-simulate without disturbing earlier analysis):

1. **`discrete_events.py`** runs the simulations and archives each scenario's
   `History` to `outputs/<scenario>/history.hf5` (compressed HDF5), plus — for the
   crypt scenarios — the emitted discrete events to `outputs/<scenario>/events.csv`
   (events come from the process emitter, not the History) and, for `gillespie`,
   the restart mesh `stable_eptm.hf5`. It produces **no** figures or videos.
2. **`discrete_events_analysis.ipynb`** reopens those archives
   (`HistoryHdf5.from_archive`) and produces all GIFs, the crypt stills and
   the analyses.

## Run

Use the repo's `vivarium-tyssue` conda env (the `.venv`/`uv` are broken). The
notebook needs ImageMagick (`magick`) on PATH for GIF rendering.

```bash
conda activate vivarium-tyssue
cd Experiments/discrete_events

# 1) simulate (writes outputs/<scenario>/history.hf5 [+ events.csv])
python discrete_events.py            # all four, in order
python discrete_events.py divisions  # or a single one:
                                     #   divisions | deaths | gillespie | gillespie_restart

# 2) analyse & visualise
jupyter lab discrete_events_analysis.ipynb   # then Run All

# ...or headless; MAKE_GIFS=0 skips the (slow) animations and keeps the analyses
MAKE_GIFS=0 jupyter nbconvert --to notebook --execute --inplace \
    discrete_events_analysis.ipynb
```

`divisions`, `deaths` and `gillespie` are independent and can be launched as
separate processes to run concurrently; `gillespie_restart` needs `gillespie`'s
checkpoint and errors out if it is missing. The notebook skips any scenario whose
`history.hf5` is missing, so it works on whichever subset you have simulated.

## Gillespie analyses

Produced for **both** crypt runs, each into its own `outputs/<run>/`, plus one
figure comparing the two.

1. **cell-type distribution over time** — stacked cell counts per type at every
   recorded timepoint (`cell_type_over_time.png` / `.csv`).
2. **spatial distribution along z** — one figure (`along_z.png`) with two panels on
   a shared crypt axis: on top, a line per cell type giving mean per-frame occupancy
   finely binned along z (`cell_type_along_z.csv`); underneath, division /
   differentiation / extrusion event counts binned along the same axis, each event
   placed at the mean z of its cell (`events_along_z.csv`, raw events in
   `events.csv`). Reading a vertical line through both panels shows the crypt's
   programme directly — divisions at the stem-cell base, differentiation through the
   progenitor zone, extrusion at the differentiated top. The panels keep separate y
   axes: occupancy is a per-frame mean, events are run totals.

3. **the two crypt runs side by side** — `runs_compared.png` (written under
   `outputs/gillespie_restart/`) puts each run's per-type cell counts against time
   on shared axes. Run 1's trajectories are dominated by the initial transient; run
   2's open flat, which is the signature of a genuinely settled starting state, so
   its drift is the stationary state's own fluctuation.

All analysis figures place their legends outside the plot area.

## Outputs (git-ignored)

Written by `discrete_events.py` (`history.hf5`, `events.csv`); everything else is
written by `discrete_events_analysis.ipynb`.

```
data/test_square.hf5        # flat-sheet mesh (divisions / deaths)
data/crypt_cylinder.hf5     # crypt mesh (gillespie); both copied on first run
outputs/
  divisions/          history.hf5                  # <- simulation (archived History)
                      divisions.gif                # <- analysis notebook (flat 2-D)
  deaths/             history.hf5
                      deaths.gif                   # <- analysis notebook (flat 2-D)
  gillespie/          history.hf5   events.csv     # <- simulation
                      stable_eptm.hf5              # <- simulation: the restart mesh
                      gillespie.gif  still_t*.png  # <- analysis notebook (crypt 3-D)
                      cell_type_over_time.png / .csv
                      along_z.png                  # cell types + events, shared z axis
                      cell_type_along_z.csv / events_along_z.csv
  gillespie_restart/  history.hf5   events.csv     # <- simulation (from stable_eptm)
                      stable_eptm.hf5              # <- simulation: chain a further run
                      gillespie_restart.gif  still_t*.png
                      cell_type_over_time.png / .csv
                      along_z.png
                      cell_type_along_z.csv / events_along_z.csv
                      runs_compared.png            # both runs' trajectories
  bigraphs/           divisions_bigraph.png        # <- analysis notebook: the composite
                      deaths_bigraph.png           #    wiring (processes peach/pink,
                      gillespie_bigraph.png        #    stores light blue)
```

`gillespie_restart` reuses the `gillespie` wiring exactly, so it gets no separate
bigraph. The crypt stills, the analysis figures and the bigraphs render at 300 dpi;
GIFs at 120 dpi. Nothing under
`data/` or `outputs/` is tracked by git — re-running regenerates everything.
Tune rates, `tf`/`dt`, the checkpoint time and archive frame caps via the constants
at the top of `discrete_events.py`; z-binning and plot styling live in the analysis
notebook.
