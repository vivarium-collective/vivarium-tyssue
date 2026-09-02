"""Calibrate the tumor <-> tyssue timescale coupling (diagnostic; writes nothing).

The overlap problem in the tumor experiment is a timescale mismatch: division
events fire faster than the vertex-model sheet can mechanically relax around them.
This script measures the two clocks and recommends a coupling that separates them:

  1. tau_mech  — mechanical relaxation time of the sheet. We equilibrate the flat
     sheet, inflate ONE central cell's prefered_area (the perturbation a division
     applies), then step the mechanics alone and watch the max vertex speed decay.
     tau_mech = time for that speed to fall to 1/e of its post-perturbation peak.

  2. division rate — how many division EVENTS the COPASI coupling fires per unit
     tyssue time (at the current alpha=1, scales). Measured by running the real
     coupling briefly and counting `division` behaviors.

From these it prints:
  * the current dimensionless crowding number  rate * tau_mech  (want << 1),
  * tau_grow before/after the `dt=interval` inflation fix,
  * a recommended alpha (copasi_time), births-scale factor, and tf.

Run from the repo's conda env:
    conda activate vivarium-tyssue
    cd Experiments/tumor_coupling
    python calibrate_timescale.py
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

# Reuse the experiment's dataset resolver and spec builders.
import tumor_coupling as EXP  # noqa: E402  (same-dir module)

# Targets (dimensionless).
TARGET_CROWDING = 0.2   # desired division rate * tau_mech (events per relaxation time)
# Desired tau_grow / tau_mech for the per-cell inflation. There is a sweet spot:
# below ~1 the cell balloons faster than the sheet relaxes (overlap); too high and
# the tumor grows so slowly it stalls (each cell is locked mid-division, starving
# flux-driven births) — a compact multi-cell seed pushes that ceiling up. ~4x tau_mech
# with a 6-cell seed is overlap-free (min face area holds above its t=0 value) and
# still grows steadily.
GROW_SEPARATION = 4.0


# ---------------------------------------------------------------------------
# 1. Mechanical relaxation time tau_mech
# ---------------------------------------------------------------------------
def measure_tau_mech(core, mesh_path, warmup=3.0, relax=15.0, dt=0.01, bump_to=2.0):
    """Inflate one central cell and time the sheet's relaxation to a new
    equilibrium via the decay of the max vertex speed."""
    from process_bigraph import Composite

    cfg = EXP.tumor_config(mesh_path)
    cfg["backend"] = "python"          # mid-run parameter change: avoid rust geom cache
    cfg["record_history"] = False
    spec = {
        "Tyssue": {
            "_type": "process", "address": "local:EulerSolver", "config": cfg,
            "inputs": {"behaviors": ["Behaviors"], "global_time": ["global_time"]},
            "outputs": {"datasets": ["Tissue State"], "network_changed": ["Network Changed"],
                        "behaviors_update": ["Behaviors"]},
            "interval": dt,
        },
        "Network Changed": False, "Behaviors": {},
    }
    sim = Composite({"state": spec}, core=core)
    eptm = sim.state["Tyssue"]["instance"].eptm
    coords = eptm.coords

    with contextlib.redirect_stdout(io.StringIO()):
        sim.run(warmup)   # settle any residual motion from loading

    # Perturb: inflate the cell nearest the sheet centroid (as a division would).
    fx, fy = eptm.face_df["x"], eptm.face_df["y"]
    cx, cy = fx.mean(), fy.mean()
    central = ((fx - cx) ** 2 + (fy - cy) ** 2).idxmin()
    eptm.face_df.loc[central, "prefered_area"] = bump_to

    prev = eptm.vert_df[coords].to_numpy(dtype=float).copy()
    times, speeds = [], []
    n = int(round(relax / dt))
    for i in range(n):
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run(dt)
        pos = eptm.vert_df[coords].to_numpy(dtype=float)
        speeds.append(float(np.abs(pos - prev).max()) / dt)
        times.append((i + 1) * dt)
        prev = pos.copy()

    speeds = np.array(speeds); times = np.array(times)
    peak_i = int(speeds.argmax())
    peak = speeds[peak_i]
    thresh = peak / math.e
    after = np.where(speeds[peak_i:] <= thresh)[0]
    tau = float(times[peak_i + after[0]] - times[peak_i]) if len(after) else float("nan")
    return tau, times, speeds


# ---------------------------------------------------------------------------
# 2. Current division-event rate (COPASI coupling, alpha=1)
# ---------------------------------------------------------------------------
def measure_division_rate(core, mesh_path, model_path, tf=30.0, window=10.0):
    """Run the real coupling and count `division` events per unit tyssue time.
    Returns (mean_rate, late_rate) where late_rate is over the final `window`."""
    from process_bigraph import Composite
    from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

    spec = EXP.build_tumor_spec(mesh_path, model_path)
    spec["emitter"] = emitter_from_wires({
        "global_time": ["global_time"], "behaviors": ["Behaviors"],
    })
    sim = Composite({"state": spec}, core=core)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run(tf)
    frames = gather_emitter_results(sim)[("emitter",)]

    seen, events = set(), []   # dedupe (time, cell_uid) division commits
    for fr in frames:
        t = float(fr.get("global_time", 0.0))
        beh = fr.get("behaviors", [])
        beh = list(beh.values()) if isinstance(beh, dict) else (beh or [])
        for b in beh:
            if isinstance(b, dict) and b.get("func") == "division":
                key = (round(t, 4), b.get("cell_uid"))
                if key not in seen:
                    seen.add(key); events.append(t)
    events = np.array(events)
    mean_rate = len(events) / tf if tf else float("nan")
    late_rate = int((events >= tf - window).sum()) / window if len(events) else 0.0
    return mean_rate, late_rate, len(events)


def main():
    from vivarium_tyssue.core import build_core

    mesh = EXP.ensure_dataset(EXP.FLAT_DATASET)
    model = EXP.ensure_dataset(EXP.SBML_MODEL)
    core = build_core()

    print("Measuring tau_mech (inflate one cell, watch the sheet relax) ...", flush=True)
    tau_mech, _, speeds = measure_tau_mech(core, mesh)
    print(f"  tau_mech ~= {tau_mech:.2f} tyssue-time units "
          f"(peak vertex speed {speeds.max():.3f})", flush=True)

    print("Measuring current division-event rate (alpha=1) ...", flush=True)
    mean_rate, late_rate, n_div = measure_division_rate(core, mesh, model)
    print(f"  {n_div} divisions over the probe run: mean {mean_rate:.2f}/unit, "
          f"late {late_rate:.2f}/unit", flush=True)

    # --- Recommendations -----------------------------------------------------
    ref_rate = max(late_rate, mean_rate)
    crowding = ref_rate * tau_mech
    # Primary lever: the inflation time. Target tau_grow ~= GROW_SEPARATION * tau_mech,
    # gradual enough that the sheet accommodates each division without overlap but
    # fast enough that the clone still establishes. With growth integrated at the real
    # solver step, tau_grow = ln2/growth_rate, so:
    growth_rate = math.log(2) / (GROW_SEPARATION * tau_mech)
    tau_grow_old = math.log(2) / (100 * math.log(1 + 1.0 * 0.1))   # old: growth_rate 0.1, dt=1.0
    tau_grow_new = math.log(2) / growth_rate

    print("\n" + "=" * 70)
    print("RECOMMENDED TIMESCALE COUPLING")
    print("=" * 70)
    print(f"  tau_mech (mechanical relaxation)  ~= {tau_mech:.2f}")
    print(f"  current division rate              = {ref_rate:.2f}/unit  "
          f"(crowding rate*tau = {crowding:.2f})")
    print(f"  tau_grow (inflation) before        = {tau_grow_old:.2f}  "
          f"(growth_rate 0.1, dt=1.0 -> ~{tau_mech/max(tau_grow_old,1e-9):.0f}x too fast)")
    print(f"  -> growth_rate                     = {growth_rate:.2f}  "
          f"(dt=interval; tau_grow ~= {tau_grow_new:.2f} = {GROW_SEPARATION:.0f}x tau_mech)")
    print(f"  -> seed a COMPACT focus (>=6 cells) so gradual-growth divisions always")
    print(f"     find a free tumor cell (a single seed stalls / dies out)")
    print(f"  -> halve ALL scales (births AND deaths together) for ~half the event")
    print(f"     rate while keeping the birth:death balance; extend tf to compensate")
    print(f"  NOTE: copasi_time (alpha) < 1 slows the tumor clock but DELAYS the")
    print(f"        induction flux and can starve the seed -> keep alpha = 1.0")
    print("=" * 70)


if __name__ == "__main__":
    main()
