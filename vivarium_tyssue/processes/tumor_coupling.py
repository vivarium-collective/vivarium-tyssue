"""TumorCoupling — drive discrete tyssue cell events from COPASI reaction fluxes.

"Gillespie, but driven by COPASI fluxes instead of internal rates." Each step:
read the SBML model's per-reaction birth/death fluxes, scale each into a
fractional event accumulator per cell type, fire floor(accumulator) discrete
behaviors on selected mesh cells, and emit scalar observables for the timeseries
charts. See docs/superpowers/specs/2026-06-12-tumor-tyssue-coupling-design.md.
"""
from __future__ import annotations

import math


def fractional_events(
    fluxes: dict, scales: dict, accumulators: dict, *, dt: float,
) -> tuple[dict, dict]:
    """Convert scaled fluxes into integer event counts via fractional accumulators.

    For each key in ``scales``: add ``max(flux, 0) * scale * dt`` to the running
    accumulator, fire ``floor(accumulator)`` events, and carry the remainder. A
    missing or negative flux contributes zero (no event is lost to rounding —
    the fractional remainder persists across steps).

    Returns ``(counts, new_accumulators)``.
    """
    counts: dict = {}
    new_acc = dict(accumulators)
    for key, scale in scales.items():
        flux = fluxes.get(key, 0.0)
        increment = max(float(flux), 0.0) * float(scale) * float(dt)
        total = new_acc.get(key, 0.0) + increment
        fired = int(math.floor(total))
        counts[key] = fired
        new_acc[key] = total - fired
    return counts, new_acc
