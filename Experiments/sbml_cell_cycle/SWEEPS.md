# Folding sweeps: what keeps the growing free sheet clean

The growing free sheet in `sbml_cell_cycle.py` eventually *folds*: a vertex
crosses an edge and faces overlap. The vertex model has no term that resists
self-intersection, and tyssue's area is unsigned, so the energy never sees the
fold. These sweeps (2026-10-01) asked which parameters delay the fold, without
adding any detection or correction.

> **Coupling used here:** the earlier one. A cell committed at mitotic exit, grew
> its target area to `crit_area` and then split (`grow_divide`). The demo now grows
> cells in G1/G2 and divides them on entering M (`grow_cells` / `split_cells`), so
> `crit_area` no longer exists and `growth` in `sweep_folding.py` now multiplies
> the G1/G2 growth rate. The mechanics findings (section 3) are about relaxation
> speed and should carry over. The division numbers (sections 1–2) are specific to
> the old coupling.

**Answer:** the ratio of the mechanics' relaxation rate to the rate of growth.
Area elasticity ÷ viscosity is the dial. Making the sheet relax 10× faster than
the demo lets it grow about 4× larger before the first fold, from ~400 cells to
~1600. No setting eliminated folding outright.

## Method

`sweep_folding.py` runs the demo's sheet (206 cells, free border, GF = 1, model
`BIOMD0000000941`) with one or more parameters changed.
- After every 1.0 sim time it checks the mesh with
  `find_intersections(strict=True)`.
- A run stops at the first fold or at the target cell count.
- **Metric:** the largest cell count with a clean mesh.
- **Seeds:** 3 per setting unless noted. Fold onset is chaotic: the same seed
  re-run varies by about ±10 cells, and settings differ by ±50–100 across seeds.
  So read the medians.

**Baseline** (the demo's settings before these sweeps; `BASELINE` in the script):
- cell cycle 20, division `growth_rate` 0.1, `crit_area` 2.0;
- `area_elasticity` K_A = 1, `prefered_perimeter` P0 = 3.5,
  `perimeter_elasticity` K_P = 0.5, `line_tension` 0;
- `viscosity` η = 0.2, T1 `threshold_length` 0.03, dt 0.02.

The baseline folds at **~360–400 cells (t ≈ 28)**.

The solver step is shortened in proportion to K/η, so explicit Euler stays
stable (`solver_dt`). A dt-only control, below, shows that this shortening is not
what helps.

## 1. Division rate: helps, but only up to a ceiling

Grid over cell cycle {20, 30, 40, 60, 80} × `growth_rate` {0.05, 0.1, 0.2} ×
`crit_area` {1.4, 1.7, 2.0}, with 3–9 seeds each.

| setting | median clean cells (range) |
|---|---|
| cycle 20 (demo) | ~370–430 |
| cycle 40 | 450–530 |
| cycle 60, growth 0.1, crit 1.4 (9 seeds, target 1600) | ~870 (501–977) |
| cycle 80, growth 0.1, crit 1.4 | 498–953, no better than 60 |

- Slower cycles and a smaller `crit_area` push the fold out, by about 2× at
  best.
- Every setting still folded.
- `growth_rate` mattered little on its own.

## 2. Perimeter target that grows with the area target: no gain

The division grower used to inflate a committed cell's `prefered_area` without
limit; `crit_area` only gates the split. In a crowded sheet a cell's target
reached ~2× its real area, and every fold had such a cell beside it.

`grow_divide()` (then named `division()`) now also grows the cell's `prefered_perimeter` as √(A₀), via
`scale_perimeter=True`, the default. Both halves get the pre-growth P0 back on
division.

Effect:
- The overgrowth ratio fell from ~1.9 to 1.3–1.5.
- Half as many cells sat in `dividing`.
- **Fold onset did not improve.** It stayed within noise, and the best
  division-rate setting fell from 800+ to ~540 cells.
- Folds then also appeared among cells that were not dividing at all. So the
  stuck dividing cells were a bystander, not the cause.

The rest of this file was run with the new perimeter scaling.

## 3. Mechanics, one parameter at a time

Each row changes one parameter from the baseline.

| parameter | value | K_A/η | median clean cells (range) |
|---|---|---|---|
| **area_elasticity** | 0.5 | 2.5 | 239 (238–252) |
| | 1 (baseline) | 5 | 361 (263–380) |
| | 2 | 10 | 624 (597–746) |
| | 5 | 25 | 820 (519–822), 2/3 reached 800 |
| | 10 | 50 | 813 (802–834), 3/3 reached 800 |
| **viscosity** | 0.4 | 2.5 | 260 (226–262) |
| | 0.1 | 10 | 683 (442–743) |
| | 0.05 | 20 | 812 (811–834), 3/3 reached 800 |
| prefered_perimeter | 3.2 / 3.3 / 3.4 | | 393–402 |
| | 3.6 / 3.7 | | 330 / 277, worse |
| perimeter_elasticity | 0.1 / 0.25 / 1 / 2 | | 368–404 |
| line_tension | 0.02 / 0.05 / 0.1 | | 390–401 |
| threshold_length (T1) | 0.01 / 0.06 / 0.1 | | 379 / 403 / 436 |
| **dt only** (dt × 0.5 / 0.25 / 0.1) | | 5 | 370 / 378 / 341, no effect |

- Only area elasticity and viscosity matter.
- The two act through one combination, the area term's relaxation rate K_A/η.
  Equal K_A/η gives the same result whether it comes from a stiffer area or a
  thinner viscosity.
- Raising P0 toward the hexagon value (3.72) makes things worse, as in the crypt.
- The perimeter, tension and T1 settings are all within noise.
- The dt-only control rules out integration error as the explanation.

## 4. Fast relaxation, out to 1600 cells

| setting | K_A/η | median clean cells (range) | reached 1600 |
|---|---|---|---|
| baseline | 5 | 399 (364–408) | 0/3 |
| η = 0.05 | 20 | 932 (766–1057) | 0/3 |
| K_A = 10 | 50 | 1548 (1250–1605) | 1/3 |
| η = 0.02 | 50 | 1605 (1455–1638) | 2/3 |

The clean size grows roughly in proportion to K_A/η. At K_A/η = 50 the sheet
reaches ~1500–1600 cells, about 3 doublings.

## Interpretation and cost

- **"Divisions happen too often" holds, as a ratio.** They are too frequent *relative to
  how fast the sheet relaxes*. Lowering η by a factor k is the same as slowing the
  cell cycle **and** division growth by k at fixed mechanics: time just rescales.
  - That's why slowing the cycle alone (section 1) hit a ceiling: growth inside
    each division stayed just as fast.
  - Folds come from vertices being pushed faster than the sheet can
    rearrange, i.e. a sheet far from mechanical equilibrium.
- **There's no free lunch on compute.** Either way the solver has to resolve the
  mechanical relaxation time, so K_A/η = 50 costs ~10× the steps per unit of
  biological time. A run to 1600 cells took ~9 min on one core.
- **Folding is delayed, not removed.** At K_A/η = 50, one seed in three still
  folded before 1600 cells. Growing indefinitely needs either relaxation that
  keeps pace with an ever larger sheet, or a self-intersection term in the energy.

**Adopted in the demo:** η = 0.02, with everything else unchanged and dt =
0.002. The run stops at 1200 cells, which was clean on all 6 K_A/η = 50 runs. The
demo then reached 1200 cells at t = 58.5 with every 0.25-unit frame clean. With
the new phase-driven coupling it reaches 1206 cells at t = 50.25, also clean on
every frame.
K_A = 10 works too, but it also makes cells stiffer and less deformable, whereas
lowering η changes only the timescale.

## Reproducing

```bash
cd Experiments/sbml_cell_cycle
python sweep_folding.py --oat --vary area_elasticity=0.5,2,5,10 --vary viscosity=0.05,0.1,0.4
python sweep_folding.py --oat --vary dt_factor=0.5,0.25,0.1
python sweep_folding.py --oat --vary viscosity=0.02 --vary area_elasticity=10 --target 1600
python sweep_folding.py --vary cycle=40,60 --vary growth=0.5,1 --seed 1 2 3   # new coupling
```

Results are printed, not saved. Each run uses one core, and the pool uses
every core but one.
