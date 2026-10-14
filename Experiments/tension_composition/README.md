# Tension composition — stochastic line tension on top of directional and adhesion tension

Two tension processes driving the **same junctions at once**:

| scenario | deterministic process | parent experiment |
|---|---|---|
| `ce` — convergent extension | `DirectionalLineTension` (polarity along x, Λ 0 → 0.15, cos² law) | `Experiments/directional_tension` |
| `sort` — cell sorting | `DifferentialAdhesion` (Λ_homo 0.05, Λ_het 0.45) | `Experiments/differential_adhesion` |

Both get `StochasticLineTension` on top: an Ornstein–Uhlenbeck share per junction with
persistence τ = 1 and amplitude σ ∈ {0.025, 0.05, 0.1}. Each scenario also has a no-noise
run (the parent's main run) and a noise-only control at σ = 0.1. For `ce` the control has
no directional tension; for `sort` it has uniform Λ_homo, which is the parent's control.
Mechanics, meshes, T1 settings and seeds are copied from the parents.

## How two processes share one parameter

Parameter-setting processes ship **deltas**, not new values. Each one moves only its own
share of `line_tension`, mirrored in its own `edge_df` column, and the `update_parameter`
behavior (`parameter="line_tension"`, `dataframe="edge"`) adds the delta in place:

```
line_tension = directional_tension + stochastic_tension      (ce)
line_tension = adhesion_tension    + stochastic_tension      (sort)
```

Each process remembers the share it last emitted rather than reading it back from the
mesh. The solver executes a behavior two steps after the process emits it, so the mesh a
process sees does not yet hold its latest deltas. A row cloned mid-run (division, vertex
split) takes its starting share from the copied column. The shares add up to
`line_tension` to within 4e-15 in every frame of every run.

## Results

| ce run | CE index at t = 240 | completed T1s | cell aspect ratio |
|---|---|---|---|
| no noise | 1.96 | 30 | 2.21 |
| σ = 0.025 | 1.96 | 197 | 1.84 |
| σ = 0.05 | 2.04 | 411 | 1.52 |
| σ = 0.1 | **2.25** | 686 | 1.58 |
| noise only, σ = 0.1 | 0.98 | 652 | 1.54 |

| sort run | sorting index at t = 200 | A\|B interface length |
|---|---|---|
| no noise | 0.55 | 108 |
| σ = 0.025 | 0.58 | 101 |
| σ = 0.05 | 0.68 | 81 |
| σ = 0.1 | **0.79** | 59 |
| noise only, σ = 0.1 | 0.49 | 197 |

- **Noise helps both mechanisms, and does neither on its own.** The noise-only controls
  stay at CE ≈ 1 and at the random-labelling sorting index 0.49.
- **Convergent extension changes character.** Without noise the tissue extends mostly by
  stretching its cells (aspect ratio 2.2, 30 T1s). With noise the cells stay rounder
  (≈ 1.55) and the tissue extends further by rearranging (≈ 700 T1s). The noise unjams the
  tissue, and the directional tension biases which rearrangements win.
- **Sorting stops being trapped.** Without noise sorting stalls early (S = 0.55): the
  heterotypic interface shrinks fast and then stops changing. Noise lets junctions keep
  remodelling, so like cells keep gathering (S = 0.79 at σ = 0.1).
- With σ ≥ 0.05 about 10–25 % of junctions carry negative tension at any moment. The OU
  share is centred on 0, as in `Experiments/stochastic_tension`. Every final mesh passes
  `Sheet.validate()`.

## Running

```bash
cd Experiments/tension_composition
python tension_composition.py                       # 10 runs in parallel, ~4 min
jupyter lab tension_composition_analysis.ipynb      # figures (Run All)
```

`python tension_composition.py --only ce` (or `sort`) runs one scenario. `--jobs` sets the
number of parallel workers.

## Outputs (git-ignored)

```
outputs/
  metrics_ce.csv, metrics_sort.csv     # per-frame metrics, from every recorded frame
  runs_ce.json, runs_sort.json         # per-run summary (time, validity, max residual)
  <scenario>/<run>/history.hf5         # history thinned to 241 frames
  bigraphs/{ce,sort}_bigraph.png
  figures/
    tension_decomposition.png          # total vs shares, for σ = 0.05
    ce_over_time.png, ce_start_end.png
    sorting_over_time.png, sort_end.png
    outcome_vs_sigma.png
    ce_s0.05.gif, sort_s0.05.gif
```

## Notes

- `auto_reconnect` must be a map (`{"enabled": True}`). The solver's `"map"` schema turns a
  bare `True` into `{}`, which silently switches T1 transitions off.
- The CE metrics are those of `directional_tension` (`run_metrics`); the sorting metrics
  those of `differential_adhesion` (`interface_metrics`).
