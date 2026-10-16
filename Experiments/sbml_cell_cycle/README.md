# SBML cell cycle in every cell

A demonstration of two things:

1. **SBML integration with tyssue.** A BioModels SBML file is read directly and
   compiled once into a vectorised numpy right-hand side
   (`vivarium_tyssue/sbml/vectorized.py`, `VectorizedSBML`). There is no
   COPASI/roadrunner call per cell. The whole population's ODEs are one
   `(n_species, n_cells)` array operation.
2. **A subcellular model running in every cell.** The `SBMLCellCycle` process
   (`vivarium_tyssue/processes/cell_cycle.py`) gives each face its own copy of the
   model. Each cell reads its phase from its own cyclins: it grows in G1 and G2,
   holds its size in S and M, and divides on entering M. A daughter inherits her
   mother's molecular state.

The full write-up, with single-cell analysis and figures, is in
`sbml_cell_cycle_analysis.ipynb`.

## The model — Gérard & Goldbeter 2011 skeleton ([BIOMD0000000941](https://www.ebi.ac.uk/biomodels/BIOMD0000000941))

C. Gérard & A. Goldbeter, *A skeleton model for the network of cyclin-dependent
kinases driving the mammalian cell cycle*, Interface Focus 1:24–35 (2011).

This is a single-cell, deterministic ODE model of the mammalian Cdk network. It is
the stripped-down backbone of the authors' 39-variable model and keeps only the
feedback loops that make the cycle oscillate and switch. The SBML has six
variables:

| symbol | SBML species | meaning |
|---|---|---|
| $M_d$ | `cyclin_D_Cdk4_6` | cyclin D/Cdk4–6 (G1) |
| $E2F$ | `transcription_factor_E2F_active` | active E2F |
| $M_e$ | `cyclin_E_Cdk2` | cyclin E/Cdk2 (G1/S) |
| $M_a$ | `cyclin_A_Cdk2` | cyclin A/Cdk2 (S/G2) |
| $M_b$ | `cyclin_B_Cdk1` | cyclin B/Cdk1 (G2/M) |
| $Cdc20$ | `Cdc20_active` | active Cdc20 (APC/C, mitotic exit) |

$$
\begin{aligned}
\frac{dM_d}{dt} &= v_{sd}\,\frac{GF}{K_{gf}+GF} - V_{dd}\,\frac{M_d}{K_{dd}+M_d}\\
\frac{dE2F}{dt} &= V_{1e2f}\,\frac{E2F_{tot}-E2F}{K_{1e2f}+E2F_{tot}-E2F}\,(M_d+M_e) - V_{2e2f}\,\frac{E2F}{K_{2e2f}+E2F}\,M_a\\
\frac{dM_e}{dt} &= v_{se}\,E2F - V_{de}\,M_a\,\frac{M_e}{K_{de}+M_e}\\
\frac{dM_a}{dt} &= v_{sa}\,E2F - V_{da}\,Cdc20\,\frac{M_a}{K_{da}+M_a}\\
\frac{dM_b}{dt} &= v_{sb}\,M_a - V_{db}\,Cdc20\,\frac{M_b}{K_{db}+M_b}\\
\frac{dCdc20}{dt} &= V_{1cdc20}\,M_b\,\frac{Cdc20_{tot}-Cdc20}{K_{1cdc20}+Cdc20_{tot}-Cdc20} - V_{2cdc20}\,\frac{Cdc20}{K_{2cdc20}+Cdc20}
\end{aligned}
$$

$E2F_{tot}=3$ and $Cdc20_{tot}=5$ are conserved totals. The only external input is
the growth factor $GF$.

**The loop:**
1. GF drives cyclin D.
2. Cyclin D, and then cyclin E, flip E2F on. Cyclin E feeds back positively, which
   makes this the G1/S switch.
3. E2F makes cyclins E and A.
4. Cyclin A switches E2F off and removes cyclin E (negative feedback), and it drives
   cyclin B.
5. Cyclin B activates Cdc20, which destroys cyclins A and B. That is mitotic exit,
   and the loop resets.

**Behaviour** (measured with this engine):
- Units (Gérard & Goldbeter 2011, Table 2): concentrations in µM ("tentatively"),
  time in hours. The SBML file declares COPASI's mmol / mL / h defaults instead.
- The period is **23.6 h** at GF = 1, and it barely depends on GF
  above threshold.
- A full cycle needs **GF > 0.072**. Below that the cell sits in a stable G0 steady
  state, so GF acts as an on/off switch rather than a rate dial.
- The decision about the next cycle is made ~3 h before the previous
  mitotic exit. A cell that loses GF after that point divides at the end of its
  current cycle, and its two daughters divide once more. Arrest therefore lags by
  one to two rounds, up to 4× the cells.

## Coupling to the vertex model

- **Growth factor.** GF = 1 in every cell, so every cell cycles.
  - The process can also drive any SBML constant per cell from a `face_df` column,
    or from a ratio such as `"area/prefered_area"` (`input_columns` /
    `input_functions` / `input_args`).
  - It's unused here. The model's late commitment (below) makes such couplings lag
    by one to two division rounds.
- **Model → mesh: phases decide growth and division.** Every step each cell's phase
  is read from its own state by ordered rules (`phases` in the process config; first
  match wins, else G1):

  | phase | rule | share of cycle | the cell |
  |---|---|---|---|
  | S | cyclin E > 1 | ~37% | holds its size |
  | M | cyclin B > 1, cyclin E ≤ 1, cyclin A < 2 | ~19% | divides on entry, then holds |
  | G2 | cyclin B > 1, cyclin E ≤ 1 | ~19% | grows |
  | G1 | otherwise | ~25% | grows |

  - **Growth.** G1 and G2 cells get the `grow_cells` behavior each step.
    `prefered_area` grows at `growth_rate` = ln 2 / (0.44 × cycle), so it doubles
    over the G1 + G2 time, and `prefered_perimeter` grows as its square root.
  - **Division.** A cell that has just entered M gets `split_cells`. It divides at
    once, and both halves restart from target area 1 and the matching perimeter.
  - **Initial sizes.** The initial cells start at the target size their phase
    implies, so a cell seeded late in G2 doesn't split at birth size.
  - Neither uses `grow_divide`, which grows a cell to `crit_area` and then splits it.
- **Inheritance.**
  - Every step, each state variable is written to a `face_df` column `cc_<species>`,
    plus `cc_phase` and `cc_last_division`.
  - The mother keeps her `unique_id`. `split_cells` stamps the daughter's
    `mother_uid`, and the process starts the daughter from her mother's live state.
  - It can't use the copied `cc_*` row. tyssue's EventManager shuffles queued
    behaviors, so the split can run before that step's state write. The daughter
    would then start in G2, re-enter M and split again at half size.
- **Heterogeneity.**
  - Initial cells get random phases along the limit cycle.
  - Each cell gets its own lognormal factor (CV 0.1) on `vsa`, `vse` and `Vde`, the
    rates the period is most sensitive to. Sister cells therefore drift apart.
- **Time.** `time_scale = 1.18` h of model time per sim unit, so one cycle is ~20 sim
  time units.

## The run: a growing sheet

The flat square sheet (`test_square.hf5`, 206 cells) runs with a **free border**:
`model_factory`, not `model_factory_bound`, which zeroes the force on every
`boundary` vertex. The sheet spreads as it grows.

| | |
|---|---|
| GF | 1 in every cell |
| mechanics | viscosity 0.02 (dt 0.002), `prefered_perimeter` 3.5, `perimeter_elasticity` 0.5, area elasticity 1 |
| rosettes | 4-fold vertices resolve after a 10 min mean wait (`p_4` 7.08, `p_5p` 0.708 per sim unit); tyssue's defaults gave ~12 h |
| cell cycle | ≈ 20 time units (measured 20.1); grow in G1/G2 at 0.079, divide on entering M |
| run | to 1200 cells (t = 50, 999 divisions, 206 → 1205 cells), snapshot every 0.25 |
| final mesh | 0 folded, 0 invading vertices, 0 overlapping pairs (`find_intersections(strict=True)`, checked on every frame) |

**Why it stays clean, and why it stops at 1200 cells.**
- A freely growing vertex sheet eventually folds, because nothing in the model
  resists self-intersection.
- How large it gets first is set by how fast the mechanics relax compared with
  growth, i.e. area elasticity ÷ viscosity. With the earlier grow-then-split
  coupling, the first fold came at ~400 cells at viscosity 0.2 and at ~1500–1600 at
  0.02, so the run stops at 1200.
- The parameter sweeps behind this are in `SWEEPS.md`. Division rate, perimeter
  target, line tension, the T1 threshold and the rosette detachment rate matter far
  less. About 1 seed in 10 folds before 1200 cells at any detachment rate tried.
- The low viscosity needs a 10× smaller time step, so the run takes ~10 minutes.
  The solver's per-step History would hold ~30,000 frames, so `run()` advances
  one 0.25-unit frame at a time and snapshots the mesh itself.

**Checked in the run's snapshots:**
- all 999 splits happen with the mother in M;
- target area grows only in G1 and G2, and is constant in S and M;
- division-to-division time is 20.1 on average (minimum 15.5, maximum 25.3; 793 complete cycles);
- no cell divides twice in a row.

**The notebook also makes two GIFs** (`outputs/figures/growth_phase.gif` and
`growth_cyclinB.gif`).
- The axes are fixed to the union of all frames' extents, so the growing tissue
  stays in view.
- `growth_phase.gif` colours cells by their phase (G1 / S / G2 / M), read from each
  cell's own cyclins. `growth_cyclinB.gif` colours them by cyclin B.
- Cells that divided within the last time unit have a magenta outline.

## Running

```bash
conda activate vivarium-tyssue
cd Experiments/sbml_cell_cycle
python sbml_cell_cycle.py
jupyter nbconvert --to notebook --execute --inplace sbml_cell_cycle_analysis.ipynb
```

- `outputs/growth/history.hf5` holds vert/edge/face snapshots every 0.25 time
  units, including every cell's `cc_*` state.
- `python sweep_folding.py ...` reruns the folding sweeps; see `SWEEPS.md`.
- `outputs/growth/summary.csv` holds the per-time summary.
- The figures and GIFs are in `outputs/figures/`.
- Both `data/` and `outputs/` are git-ignored.

## Limitations

- **No size checkpoint.** The cycle runs on its own clock. Growth is prescribed by
  phase, and a cell divides on entering M whatever size it has actually reached.
- **No CKIs (p21/p27) or pRB**, so there is no DNA-damage checkpoint.
- **Commitment is set before the previous mitotic exit.** Any signal coupled to GF
  acts one to two division rounds late. Models with a G1 restriction point
  (BIOMD0000000730, Conradie 2010 BIOMD0000000265) are drop-in replacements through
  `model_source`.
- **Deterministic** apart from the random initial phases and per-cell parameter
  noise. Runs are reproducible: `seed_all()` seeds numpy and Python's `random`, which
  tyssue's EventManager uses, and BLAS is pinned to one thread.
- **The vertex model has no anti-self-intersection term.** A free, growing sheet
  folds eventually: at ~1500–1600 cells with these mechanics. That is what bounds
  the run length.
