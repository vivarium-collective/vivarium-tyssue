# Convergent extension from directional (planar-polarised) line tension

A flat **2D** epithelial sheet — the repo's shared `workspace/datasets/test_square.hf5`,
206 cells with **trimmed borders** and a free boundary, run as a true 2D sheet
(`PlanarGeometry`, `dim == 2`, no `z` column) — in which junction line tension is graded by
the angle each junction makes with a **user-supplied polarity vector**. Junctions lying
*along* the polarity axis are the most contractile, junctions *across* it the least, so
the sheet narrows along the polarity direction and lengthens across it: **convergent
extension**.

The mechanism follows the planar-polarised-tension picture of vertex-model convergent
extension (Biophysical Journal, 2021,
[S0006349521008845](https://www.sciencedirect.com/science/article/pii/S0006349521008845)),
with two deliberate differences: the polarity axis is **any vector the user supplies**
(normalised by the process, so `(3, 3)`, `(1, 1)` and `(-1, -1)` are all the same axis),
and the angle that sets the tension is the **acute** angle between that vector and the
junction itself.

## The law

For a junction with unit direction $\hat d$ and a normalised polarity vector $\hat p$,

$$\theta = \arccos\big(|\hat d \cdot \hat p|\big) \in [0, \pi/2], \qquad
  \Lambda(\theta) = \Lambda_{\min} + (\Lambda_{\max} - \Lambda_{\min})\, a(\theta)^{\,n}$$

The absolute value makes $\theta$ acute, which is what makes the response invariant both
to the sign of the polarity vector and to which endpoint the mesh lists as `srce`. The
alignment $a(\theta) \in [0,1]$ is 1 for a parallel junction and 0 for a perpendicular
one, and `sharpness` ($n$) narrows or broadens the high-tension cone without moving
either extreme:

| `profile` | $a(\theta)$ | |
|---|---|---|
| `cos2` (default) | $\cos^2\theta = \tfrac12(1 + \cos 2\theta)$ | with $n=1$ this is the classical nematic law $\Lambda = \bar\Lambda(1 + \alpha\cos 2\theta)$ |
| `abs_cos` | $\lvert\cos\theta\rvert$ | broader high-tension cone |
| `linear` | $1 - 2\theta/\pi$ | linear in the angle itself |

## The mesh

The same sheet the **stochastic-tension** experiment runs on, copied into `data/` on first
use exactly as that experiment does, so the two flat-sheet experiments share one substrate.
It arrives already `sanitize`d with **trimmed borders**: the Voronoi border cells are
clipped rather than left ragged, so 43 of the 206 cells are partial (four- and five-sided)
and the sheet has a genuine free boundary.

One conversion is applied: the stored mesh carries an all-zero `z` column, which makes
tyssue infer a 3-D sheet. Dropping it gives a true 2D sheet so `PlanarGeometry` applies and
nothing is quietly 2.5-D; topology and vertex positions are untouched. (`Epithelium.__init__`
re-materialises any column its specs declare, so the sheet is rebuilt on `planar_spec()` and
the drop repeated afterwards — otherwise `z` comes straight back at its default.)

Energy: `FaceAreaElasticity` + `PerimeterElasticity` + `LineTension`, integrated by the
stock `EulerSolver` with `auto_reconnect` (T1 transitions) on. $A_0$ and $P_0$ are the
sheet's mean cell area and perimeter, so the interior starts very close to its rest state
and every deformation is paid for by the tension term. The clipped border cells start at
about half an interior cell's area and relax outward over the first few time units — a
transient the isotropic control measures too, which is why the CE index is normalised per
run at $t = 0$.

## The runs

Three runs, differing only in the polarity law:

| run | `polarity` | $\Lambda_{\min}$ | $\Lambda_{\max}$ |
|---|---|---|---|
| `px` | $(1, 0)$ | 0.0 | 0.15 |
| `pxy` | $(1, 1)$ — not normalised, not a lattice axis | 0.0 | 0.15 |
| `control` | $(1, 0)$ | 0.075 | 0.075 |

The control runs the same process on the same wiring; equal extremes make
$\Lambda(\theta)$ constant, so the polarity vector becomes irrelevant. That isolates
*anisotropy* as the cause — not the presence of line tension, and not the total
contractility, which is matched at $\bar\Lambda = 0.075$.

### Choosing $\Lambda_{\max}$

This is the parameter to be careful with: it competes directly with the area term
($K_A = 1$, $A_0 = 1$), and a tension that wins that competition does not deform the
tissue, it *degrades* it. Measured at $t = 240$ with everything else fixed:

| $\Lambda_{\max}$ | CE index | area kept | smallest cell | mean sides | verdict |
|---|---|---|---|---|---|
| 0.05 | 1.29 | 96 % | 1.01 | 5.72 | pristine, modest signal |
| 0.10 | 1.62 | 93 % | 0.97 | 5.71 | very clean |
| **0.15** | **1.96** | **90 %** | **0.92** | **5.70** | **chosen** |
| 0.20 | 2.36 | 87 % | 0.81 | 5.68 | intact, cells noticeably squashed |
| 0.30 | 3.27 | 81 % | 0.68 | 5.56 | starting to degrade |

0.15 gives a clear signal with the area loss saturating early (most of it by
$t \approx 40$) rather than running away, no cell ending below 0.92 of the sheet's rest
area, and the polygon distribution essentially unchanged. This sheet tolerates tension
noticeably better than a perfectly regular hexagonal patch does — the trimmed border gives
it somewhere to yield.

## Everything lives in one notebook

    directional_tension.ipynb

Run it from the repo's **`vivarium-tyssue` conda env** (the `.venv`/`uv` setup is broken)
with this directory as the working directory:

```bash
conda activate vivarium-tyssue
cd Experiments/directional_tension
jupyter lab directional_tension.ipynb      # Run All
```

End to end it takes about fifteen minutes: three 2400-step runs, three GIFs and the metrics.
The GIFs are written with Pillow, so **no ImageMagick is needed**.

## What drives it

| piece | where it lives |
|---|---|
| `DirectionalLineTension` process | `vivarium_tyssue/processes/regulations.py` |
| `update_tension`, `apply_gradient` behaviors | `vivarium_tyssue/behaviors/behaviors.py` |
| process registration | `vivarium_tyssue/processes/__init__.py` |
| tests | `tests/test_directional_tension.py` |

The process regulates the `EulerSolver` **only** through the `behaviors` port, exactly
like `StochasticLineTension` / `AnisotropicTension`; `EulerSolver` itself is untouched.
The tensions are computed **in the process** — from `edge_df`'s `dx`/`dy` — and shipped
as a `unique_id -> tension` map through the `update_tension` behavior.

That split is the opposite of the one `DifferentialAdhesion` makes, and deliberately so.
There, whether a junction is heterotypic depends on the *pair of cells it separates*,
which a T1 rewires, so the classification must happen on the live mesh inside the
solver's `EventManager`. Here a junction's tension depends only on **its own geometry**,
and the map is keyed by `unique_id`, so an entry for a junction a T1 has since removed
simply matches nothing. The map is recomputed every step, so a junction that rotates
towards the polarity axis is retensioned as it rotates and a junction born of a T1 picks
up its tension on the next step.

A second behavior, the repo's generic `apply_gradient`, writes the per-edge alignment
$a(\theta)$ into `edge_df["polar_alignment"]` so the polarity read-out lands in the
solver's `History` next to the tension. **No new behavior was added for this
experiment**, and the column has to be declared in the solver's `parameters` (`edge_df`)
so that it exists when the `History` is built — a column that does not exist at build
time is never recorded, however faithfully it is written later.

## The movie axes

Tyssue's own `create_gif` takes its axis limits from **frame 0**. That is wrong here: the
tissue grows along the extension axis as it runs, so frame-0 limits crop the end of the
movie and rescale the extension away.

The notebook's `frame_bounds` instead scans **every** frame of **every** run, takes the
union of the bounding boxes, and squares that box up to the figure's aspect ratio (only
ever adding room, never cropping). With a fixed `figsize`, a fixed `dpi` and *no*
`bbox_inches="tight"` — which would re-crop each frame independently — one unit of $x$ is
the same number of pixels as one unit of $y$ in every frame of all three movies, so the
shape change on screen is the real shape change and the three runs are directly
comparable.

## Analysis

Every metric is recomputed from the recorded mesh (positions + topology), never read back
off a column the process wrote, so it is an independent check on the model.

- **Tissue shape** — twice the standard deviation of the vertex coordinates projected onto
  $\hat p$ and $\hat p^{\perp}$ ($L_{\parallel}$, $L_{\perp}$). Second moments rather than
  a bounding box: one vertex flicking out on the free boundary moves the box and barely
  moves the moment. The **CE index** is $(L_{\perp}/L_{\parallel})$ normalised to 1 at
  $t = 0$.
- **Cell shape** — the same second-moment construction per cell gives an elongation and a
  long axis; averaged nematically against $\hat p$ that is $S_{\text{cell}}$. Comparing the
  cells' own aspect-ratio change $C$ with the tissue's $\mathrm{CE}$ gives a
  **rearrangement share** $R = 1 - \ln C / \ln \mathrm{CE}$: 1 for convergent extension
  carried entirely by T1s with cells keeping their shape, 0 for a purely affine deformation.
- **T1 transitions** — counted as *completed* rearrangements: a pair of cells that becomes
  adjacent having never been adjacent before (tracked by face `unique_id`, since the row
  index is renumbered by every `reset_index`). A junction that shrinks and re-grows between
  the same two cells exchanged no neighbours and is not counted.
- **The law, read back** — recomputing $\theta$ from the recorded positions and plotting it
  against the recorded `line_tension` reproduces the Step-2 curve, on the starting mesh and
  on one rewired by hundreds of T1s. The points hug the curve rather than sitting exactly on
  it, which is the pipeline showing through: a frame's tension was computed from the
  *previous* frame's geometry, so each junction is scored at an angle one $\mathrm{d}t$ stale.

## What the runs actually show

Over $t = 0 \to 240$ (2400 steps), at $\Lambda_{\max} = 0.15$:

| | `px` | `pxy` | `control` |
|---|---|---|---|
| CE index | **1.96** | **1.82** | 0.95 |
| $L_{\parallel}$ | $\times 0.66$ | $\times 0.71$ | $\times 0.98$ |
| $L_{\perp}$ | $\times 1.30$ | $\times 1.29$ | $\times 0.93$ |
| completed T1s | 22 | **247** | **0** |
| cell aspect change $C$ | $\times 1.90$ | $\times 1.25$ | $\times 0.95$ |
| rearrangement share $R$ | $0.04$ | $\mathbf{0.63}$ | — |
| total area | $228 \to 203$ | $228 \to 201$ | $228 \to 215$ |
| cells lost | 0 | 0 | 0 |

Convergent extension about the polarity axis, rotating with the polarity vector, and an
isotropic control that does neither and fires **no** T1s at all. Two things worth stating
rather than glossing:

* **How much of it is rearrangement depends on the polarity direction**, not just on the
  tension law — the most interesting result here. Along $(1,0)$ the deformation is
  essentially affine: $R \approx 0.04$, only 22 T1s, and the cells stretch by as much as the
  tissue does. Along $(1,1)$, nearly two thirds of the tissue's shape change is carried by
  cells changing neighbours — $R \approx 0.63$, 247 T1s, and the tissue reaches
  $\mathrm{CE} = 1.82$ while its cells elongate by only $\times 1.25$. That is convergent
  extension by intercalation, the textbook mechanism.

  The cause is the lattice. The hexagonal tiling has its junctions in essentially three
  orientation families (30°, 90°, 150°). With $\hat p$ along $x$, the 90° family sits at
  $\theta = 90°$ where $\Lambda = \Lambda_{\min} = 0$: those junctions are not merely weak,
  they are *free*, so the sheet's cheapest response is to stretch them and almost nothing
  ever shrinks enough to fire a T1. Rotate $\hat p$ to 45° and the three families land at
  $\theta = 15°, 45°, 75°$ — none free — so junctions do collapse and the tissue rearranges.
  Raising $\Lambda_{\min}$ off zero is the knob that removes the free family.

* **The sheet shrinks somewhat** ($228 \to \sim 203$ in total area, against 215 for the
  control): line tension has no opposing term at short junction lengths, so the sheet
  settles smaller than its rest area. The loss **saturates** (most of it by $t \approx 40$),
  no cell ends below 0.92 of the sheet's rest area, and the polygon distribution is
  essentially unchanged.

Exact numbers move slightly between runs — tyssue's `EventManager` shuffles its queue with
the stdlib RNG, which nothing seeds.

## Outputs (git-ignored)

    data/test_square.hf5                the shared mesh, copied from workspace/datasets
    data/flat_sheet_2d.hf5              the same mesh with z dropped (what the solver loads)
    outputs/px.gif                      polarity (1,0)      — junctions coloured by live tension
    outputs/pxy.gif                     polarity (1,1)
    outputs/control.gif                 isotropic control
    outputs/initial_condition.png       the starting sheet + its junction-orientation families
    outputs/tension_law.png             Lambda(theta) for each profile + the mesh coloured by it
    outputs/start_end.png               start vs end, all three runs, on the movies' axes
    outputs/convergent_extension.png    L_par / L_perp, the CE index, and the T1 count
    outputs/cell_shape.png              cell vs. tissue shape, R, S_cell, length-weighted alignment
    outputs/tension_vs_angle.png        the law read back off the run
    outputs/metrics.csv                 every per-frame metric, one row per sampled frame
    outputs/directional_tension_bigraph.png

Nothing under `data/` or `outputs/` is tracked by git — the experiment is fully
regenerated by re-running the notebook. To change the polarity, edit `POLARITIES` in
Step 2; to change the law, edit `LAMBDA_MIN` / `LAMBDA_MAX` / `PROFILE` / `SHARPNESS` in
the same cell.
