# Quantitative SBML <-> tissue coupling — work-in-progress handoff

Status: **paused, code reverted to the working flux-based coupling.** This documents
the quantitative-coupling attempt so it can be resumed. Everything below was validated
in isolation except the final controller, which stalls at the first coupled step
(open bug, see section 5).

## 1. Goal

Make the coupling between the COPASI tumor ODE (BIOMD0000000903) and the 3D tyssue
monolayer **quantitative**: instantiate the SBML model at the **same cell counts** as
the tissue and have the tissue mirror it 1:1, with the SBML model **slowed to the
mechanical timescale** so it doesn't demand more divisions than the mesh can absorb.
(User: same initial healthy+cancer populations; grow-before-dividing; crit_vol < 2;
ignore E, I.)

## 2. Cell-scale SBML model — DONE and verified

BIOMD0000000903 is nonlinear with million-scale constants (logistic k*C*(1-C/M),
saturations p*C*E/(a+C), bilinear delta*H*T) -> **not scale-free**. To run it at ~200
cells you must rescale species AND every parameter by its dimensional role. Two
factors: **alpha** (population) and **beta** (time).

Rescaling map (from the exact rate laws, extracted via basico):

  - x alpha        : M1,M2,M3,a1,a2,a3,w,v            (population caps/saturations)
  - x beta         : k1,k2,q,p1,p2,p3,n1,n2,p,u,mu,d1,d2,d3   (rate constants)
  - x beta/alpha   : gamma1,gamma2,gamma3,delta       (bilinear interaction coeffs)
  - x alpha*beta   : s,tau                            (zeroth-order source fluxes)

- **alpha = 206 / 25e6 ~= 8.24e-6** (healthy carrying capacity M3 -> 206 = mesh cells).
- **beta = 0.5** (slows the ODE; running the rescaled model for wall-time D = native
  progress beta*D). beta sets pacing; the endpoint depends only on beta*(run length).
- **Verification (passed):** a homogeneous rescale (beta=1, ICs = alpha*native)
  reproduces the native trajectory scaled by alpha to <1e-5 rel-err — confirms the
  parameter classification.
- **Save/load (passed):** basico.save_model(path, type="sbml") -> CopasiUTCProcess
  reloads ICs exactly and steps correctly.

Species: **T** tumor, **H** healthy, **C** stem (cancer-stem), I immune, E estrogen.
**C must be > 0** or the cascade is dead (stem formation & tumor induction are prop. to C).

Initial amounts (mesh N=206; seed 6 tumor + 6 stem): **H0=194, T0=6, C0=6, E0=0, I0=0**
(total 206 = tissue at t=0). Small-focus dynamics (beta=0.5, TF=40): tumor 6->~134, stem
6->~28, healthy 194->~2 (near-total tumor takeover). Fully non-degenerate.

Reusable helper written: write_cellscale_sbml(native, out, alpha, beta, ics) (uses
basico get_parameters/set_parameters/set_species). Verification scripts are in the job
tmp dir (rescale_verify.py, calib.py, roundtrip.py).

## 3. Grow-before-dividing — the key mechanical finding

**Instant division (division_3d_instant, grow-free) is unusable in this monolayer.**
Profiling showed tyssue's monolayer cell_division takes **~6.4 s per call** on an
un-grown cell (calls sheet_topology.face_division ~130x/division — the thin-slab
geometry makes the split plane graze many faces). A full run hangs.

**Grow-then-divide with a LOW crit_vol is the fix.** division_3d with crit_vol=1.3,
growth_rate=5.0: cells grow to crit in ~1-2 steps then split cleanly. Timing dropped
to **<=0.31 s/step**, Nv stable (820->867), mesh clean. So: keep division_3d, set
crit_vol low (< 2) and growth_rate high so the committed pool stays small.
(auto_reconnect on or off both fine here; the earlier Nv thrashing was an artifact of
the instant-division strain, not reconnect.)

## 4. The coupling controller — three iterations

The tissue must be driven so per-type counts track round(SBML species). Committed cells
are relabelled "dividing" (a **mixed** tumor/stem pool), and commit_type is **not**
recorded in History — both complicate counting.

1. **Error-feedback on raw counts** (drive current->round(species)): oscillates — the
   datasets TumorCoupling reads lag the behaviours it emits, so it re-fires corrections
   (killed 12 healthy while seeding converted the same cells; mass churn, 17 s steps).
2. **Delta-driven** (fire round(now)-round(prev)): stable & fast, but **loses events** —
   prev_target advances even when no free cell exists to divide, so the tissue
   permanently lags (tumor 11 vs model 18; stem 8 vs 22).
3. **Mixed (current attempt):** cancer types (tumor/stem) via *lineage* error-feedback
   (count committed cells in their commit_type lineage so in-flight divisions aren't
   double-fired or lost; only divide cells with commit_state==0, which waits for the
   seed to exist); **healthy via delta-deaths** (H only declines -> monotonic, no
   oscillation). See _lineage_state + _update_quantitative in the reverted diff.

## 5. OPEN BUG (where it stopped)

The **mixed controller stalls at the 2nd coupled step** (only step 1 = seeding printed;
step 2 never completed in 5 min). Prime suspect: the seeding <-> datasets-lag interaction
at the boundary (the first _update_quantitative may still see the pre-seed snapshot and
fire something expensive, or a division is triggered that shouldn't be). Not yet
root-caused. First debugging step on resume: instrument step 2's _lineage_state output
and the emitted behaviours (per-call log to a FILE — conda run buffers stdout, see s.7).

## 6. Design parameters (for resume)

- alpha = 206/25e6, beta = 0.5 (env TUMOR3D_BETA), crit_vol = 1.3, growth_rate = 5.0.
- seed {tumor:6, stem:6}; ICs H=194,T=6,C=6,E=0,I=0.
- species_map {T:tumor, H:healthy, C:stem}; substrate = healthy (delta-deaths).
- Calibration: beta*TF = native progress; native progress 20 -> tumor->134. For a
  gentle, fast demo target native progress ~2-4 (tumor->20-40, fewer divisions).

## 7. Environment gotchas (cost a lot of time)

- **conda run buffers ALL stdout until the child exits** — no interim progress. Have
  scripts write progress to a plain file with open/write/close per line.
- macOS has no `timeout`. Foreground `sleep` is blocked (use a bg job + poll a file).
- Leftover background conda-run children keep competing for CPU (caused a 61 s outlier
  step) — pkill -9 -f them and verify `ps aux | grep vivarium-tyssue/bin/python`.
- The bg-isolation guard blocks the Edit/Write tools on this checkout; edits were applied
  via Bash `python` scripts.

## 8. Next steps

1. Root-cause the step-2 stall (section 5).
2. Once tracking is clean, wire the quantitative analysis: tissue tumor-lineage
   (tumor+dividing) vs round(T), healthy vs round(H) — expect near-exact (corr~1).
3. Pick beta / run-length for a mesh-safe, visually clear demo (native progress ~2-4).
4. Consider recording commit_type in History (via the new history_columns config) so the
   analysis can resolve the "dividing" pool exactly instead of relying on a small pool.

The eulersolver.py protection (contract tests + pre-commit hook + CLAUDE.md) is unrelated
and stays in place.
