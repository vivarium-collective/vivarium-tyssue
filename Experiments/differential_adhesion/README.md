# Differential-adhesion cell sorting on a closed spherical epithelium

A 2.5-D **closed** epithelial sheet (`ClosedSheetGeometry`) of 200 cells wrapped onto a
sphere, carrying two cell types (`A` / `B`) mixed 50:50 at random. Junction tension is
set by the *identity of the two cells the junction separates* — heterotypic (`A|B`)
interfaces are held at a higher line tension than homotypic (`A|A`, `B|B`) ones — so the
tissue lowers its energy by shrinking the mixed interface, and the two populations sort
into domains.

Energy: `FaceAreaElasticity` + `PerimeterElasticity` + `LineTension` +
`LumenVolumeElasticity` + `SurfaceElasticity`, integrated by the stock `EulerSolver`
with `auto_reconnect` (T1 transitions) on.

Everything — mesh generation, config, spec, bigraph, run, movie, analysis — lives in the
single notebook

    differential_adhesion.ipynb

Run it from the repo's **`vivarium-tyssue` conda env** (the `.venv`/`uv` setup is broken)
with this directory as the working directory. The GIFs need ImageMagick's `magick` on
PATH. End to end it takes about 10 minutes: two 2400-step runs (the sorting run and a
uniform-tension control), two GIFs and the metrics.

Headline result: the sorting index — homotypic junctions / all junctions — rises from
0.49 (random 50:50) to ~0.70 over 120 time units, while the control stays flat at 0.49.
The total `A|B` interface length, which is what the energy actually penalises, falls from
189 to ~72 (the control ends at 192). Exact numbers move by a percentage point between runs: tyssue's
`EventManager` shuffles its queue with the stdlib RNG, which nothing seeds.

## What drives it

| piece | where it lives |
|---|---|
| `DifferentialAdhesion` process | `vivarium_tyssue/processes/regulations.py` |
| `differential_adhesion` behavior | `vivarium_tyssue/behaviors/behaviors.py` |
| behavior registration | `vivarium_tyssue/maps/behavior_maps.py` |
| process registration | `vivarium_tyssue/processes/__init__.py` |
| vertex-deviation geometry | `vivarium_tyssue/geometry/vert_deviation.py` |
| geometry registration | `vivarium_tyssue/core_maps.py` |
| tests | `tests/test_differential_adhesion.py`, `tests/test_vert_deviation_geometry.py` |

The process regulates the `EulerSolver` **only** through the `behaviors` port, exactly
like `StochasticLineTension` / `AnisotropicTension`; `EulerSolver` itself is untouched.

The split between the two is deliberate. The process ships only the two tension *values*;
the behavior does the classification, on the live epithelium, inside the solver's
`EventManager`. A junction rewired by a T1 separates a different pair of cells than it did
a step earlier, so a tension table computed once at t = 0 and keyed by edge id would put
homotypic tension on a freshly-made heterotypic interface. Re-deriving the classification
from the current `srce`/`trgt`/`face` topology every step is what keeps that from
happening.

## The surface smoother

`SurfaceElasticity` is a *vertex* effector: it penalises how far each vertex sits from
the centroid of its neighbours (`vert_df["dev_length"]`), which damps a cell everting
into a spike. It only **reads** that column — nothing in stock tyssue writes it for a
sheet, only the cylinder geometry does — so this repo adds
`ClosedSheetVertDeviationGeometry`: `ClosedSheetGeometry` plus a vectorised one-ring
deviation update. It averages every neighbour rather than the first three the edge table
happens to list, which matters because a T1 reorders that table.

It is a **trade-off, not a free improvement**. Over a full run, raising $K_S$ buys
smoothness and pays in sorting, monotonically: $K_S = 0$ leaves 19 everted vertices at
S = 0.71, $K_S = 0.5$ removes eversion entirely but freezes the mesh and sorting stops
(S = 0.50). `dev_length` mixes out-of-plane bulge with the in-plane one-ring irregularity
that sorting legitimately produces, so a symmetric quadratic on it cannot target one
without the other — and re-centring the well on the sorted state's roughness makes the
eversion *worse*, not better. The notebook has the full sweep and the reasoning. The
default is $K_S = 0.05$; set it to 0 to recover the original run.

## Outputs (git-ignored)

    data/sphere_200.hf5                 the generated mesh
    outputs/cell_sorting.gif            the movie
    outputs/cell_sorting_control.gif    the uniform-tension control
    outputs/sorting_index.csv           S(t) and the counts behind it
    outputs/sorting_index_control.csv
    outputs/sorting_index.png           S(t), both runs
    outputs/heterotypic_length.png      total A|B interface length, both runs
    outputs/roughness.png               mean one-ring deviation over time, both runs
    outputs/initial_condition.png       the 50:50 starting mix
    outputs/start_end.png               start vs end vs control
    outputs/differential_adhesion_bigraph.png
