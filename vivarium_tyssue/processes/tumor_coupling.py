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


def _rows(face_df) -> list[tuple[int, str]]:
    """[(unique_id, cell_type)] from a face_df (dict-of-lists, pandas DataFrame, or empty)."""
    uids = face_df.get("unique_id")
    if uids is None:
        return []
    types = face_df.get("cell_type")
    uids_list = list(uids)  # works for list, np.array, pd.Series
    types_list = list(types) if types is not None else ["?"] * len(uids_list)
    return [(int(u), str(t)) for u, t in zip(uids_list, types_list)]


def select_uids(face_df: dict, cell_type: str, n: int, *, exclude: set, rng_pick) -> list:
    """Choose up to ``n`` unique_ids of cells whose type is ``cell_type``,
    skipping any in ``exclude``. Caps to availability. ``rng_pick(items, k)``
    selects k items from a list (injected for deterministic tests; the process
    passes a numpy-random sampler)."""
    candidates = [u for (u, t) in _rows(face_df) if t == cell_type and u not in exclude]
    if n <= 0 or not candidates:
        return []
    k = min(n, len(candidates))
    return list(rng_pick(candidates, k))


import numpy as np
from process_bigraph import Process

_EVENT_KEYS = [
    "tumor_births", "tumor_deaths", "healthy_births", "healthy_deaths",
    "stem_births", "stem_deaths",
]
_TYPES = ["tumor", "healthy", "stem"]


class TumorCoupling(Process):
    """Drive tyssue cell events from COPASI reaction fluxes (see module docstring)."""

    config_schema = {
        "birth_fluxes": "map[string]",   # cell_type -> exact COPASI flux key
        "death_fluxes": "map[string]",   # cell_type -> exact COPASI flux key
        "scales": "map[float]",          # event key (e.g. 'tumor_births') -> scale
        "geom": "string",
        "dt": "float",
        "growth_rate": "float",
        "shrink_rate": "float",
        "division_crit": "float",
        "apoptosis_crit": "float",
        "seed": "map[integer]",          # {'tumor': n, 'stem': m} initial focus
    }

    def initialize(self, config):
        self.birth_fluxes = config["birth_fluxes"]
        self.death_fluxes = config["death_fluxes"]
        self.scales = config["scales"]
        self.geom = config["geom"]
        self.dt = config["dt"]
        self.growth_rate = config["growth_rate"]
        self.shrink_rate = config["shrink_rate"]
        self.division_crit = config["division_crit"]
        self.apoptosis_crit = config["apoptosis_crit"]
        self.seed = config.get("seed", {}) or {}
        self._birth_acc = {t: 0.0 for t in _TYPES}
        self._death_acc = {t: 0.0 for t in _TYPES}
        self._seeded = False

    def inputs(self):
        return {"fluxes": "map[float]", "datasets": "tyssue_data", "global_time": "float"}

    def outputs(self):
        out = {"behaviors": "list[node]"}
        for k in _EVENT_KEYS:
            out[k] = "float"
        for t in _TYPES:
            out[f"{t}_count"] = "float"
        return out

    # -- behavior dict builders (shapes match gillespie.py / behaviors.py) --
    def _divide(self, uid, cell_type):
        return {"func": "divide_crypt", "geom": self.geom, "cell_uid": int(uid),
                "dt": self.dt, "cell_type": cell_type, "crit_area": self.division_crit,
                "growth_rate": self.growth_rate}

    def _kill(self, uid):
        return {"func": "apoptosis_extrusion", "geom": self.geom, "cell_uid": int(uid),
                "dt": self.dt, "crit_area": self.apoptosis_crit, "shrink_rate": self.shrink_rate}

    def _differentiate(self, uid, new_type):
        return {"func": "differentiation", "cell_uid": int(uid), "new_type": new_type}

    @staticmethod
    def _pick(items, k):
        idx = np.random.choice(len(items), size=k, replace=False)
        return [items[i] for i in idx]

    def _counts(self, face_df):
        rows = _rows(face_df)
        return {t: float(sum(1 for (_, ct) in rows if ct == t)) for t in _TYPES}

    def update(self, inputs, interval):
        face_df = inputs["datasets"]["face_df"]
        behaviors = []
        used: set = set()
        fired = {k: 0 for k in _EVENT_KEYS}

        # --- Seeding step: convert the initial focus, then return. ---
        if not self._seeded:
            self._seeded = True
            for healthy_uid in select_uids(face_df, "healthy", int(self.seed.get("tumor", 0)),
                                           exclude=used, rng_pick=self._pick):
                behaviors.append(self._differentiate(healthy_uid, "tumor")); used.add(healthy_uid)
            for healthy_uid in select_uids(face_df, "healthy", int(self.seed.get("stem", 0)),
                                           exclude=used, rng_pick=self._pick):
                behaviors.append(self._differentiate(healthy_uid, "stem")); used.add(healthy_uid)
            counts = self._counts(face_df)
            return self._result(behaviors, fired, counts)

        # --- Flux-driven events. ---
        births, self._birth_acc = fractional_events(
            {t: inputs["fluxes"].get(self.birth_fluxes.get(t, ""), 0.0) for t in _TYPES},
            {t: self.scales.get(f"{t}_births", 0.0) for t in _TYPES},
            self._birth_acc, dt=self.dt)
        deaths, self._death_acc = fractional_events(
            {t: inputs["fluxes"].get(self.death_fluxes.get(t, ""), 0.0) for t in _TYPES},
            {t: self.scales.get(f"{t}_deaths", 0.0) for t in _TYPES},
            self._death_acc, dt=self.dt)

        # Births. Tumor birth prefers differentiating a free stem cell (C->T).
        for t in _TYPES:
            for _ in range(births[t]):
                if t == "tumor":
                    stem = select_uids(face_df, "stem", 1, exclude=used, rng_pick=self._pick)
                    if stem:
                        behaviors.append(self._differentiate(stem[0], "tumor")); used.add(stem[0])
                        fired["tumor_births"] += 1; continue
                pick = select_uids(face_df, t, 1, exclude=used, rng_pick=self._pick)
                if pick:
                    behaviors.append(self._divide(pick[0], t)); used.add(pick[0])
                    fired[f"{t}_births"] += 1

        # Deaths.
        for t in _TYPES:
            for _ in range(deaths[t]):
                pick = select_uids(face_df, t, 1, exclude=used, rng_pick=self._pick)
                if pick:
                    behaviors.append(self._kill(pick[0])); used.add(pick[0])
                    fired[f"{t}_deaths"] += 1

        return self._result(behaviors, fired, self._counts(face_df))

    def _result(self, behaviors, fired, counts):
        result = {"behaviors": behaviors}
        for k in _EVENT_KEYS:
            result[k] = float(fired[k])
        for t in _TYPES:
            result[f"{t}_count"] = counts[t]
        return result
