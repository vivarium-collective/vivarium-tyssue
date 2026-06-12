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
        # COPASI is owned internally rather than wired as a separate process:
        # process-bigraph's generic map[float] stores drop unknown keys and
        # accumulate additively, so passing per-step fluxes through a shared store
        # is unreliable. With model_source set, this process steps the SBML model
        # itself each update and reads fluxes directly. Leave empty in unit tests
        # and inject fluxes via the `fluxes` input port instead.
        "model_source": "string",
        "copasi_time": "float",
        "copasi_intervals": "integer",
        # topology_ops=true uses real tyssue cell_division / remove_face behaviors
        # (divide_crypt / apoptosis_extrusion). Those crash under the pandas-3.0 /
        # numpy-2.2 that the COPASI stack forces (tyssue 1.1.0 topology is
        # incompatible). Default false: drive cell FATE on a fixed-topology mesh —
        # birth relabels a source cell into the target type, death relabels a cell
        # to "dead". Composition still evolves under SBML control; flip to true once
        # a pandas-3-compatible tyssue is available. See the design doc.
        "topology_ops": "boolean",
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
        self.topology_ops = bool(config.get("topology_ops", False))
        self._birth_acc = {t: 0.0 for t in _TYPES}
        self._death_acc = {t: 0.0 for t in _TYPES}
        self._seeded = False
        # Internal COPASI process (optional — only when model_source is given).
        self._copasi = None
        model_source = config.get("model_source", "")
        if model_source:
            from pbg_copasi.processes import CopasiUTCProcess
            core = getattr(self, "core", None)
            if core is None:
                from vivarium_tyssue.core import build_core
                core = build_core()
            self._copasi = CopasiUTCProcess(
                config={
                    "model_source": model_source,
                    "time": float(config.get("copasi_time", 1.0) or 1.0),
                    "intervals": int(config.get("copasi_intervals", 10) or 10),
                },
                core=core,
            )

    def _read_fluxes(self, inputs, interval):
        """Fluxes for this step: an injected `fluxes` input (unit tests) takes
        precedence; otherwise step the internal COPASI model and read its fluxes
        directly (the model carries its own state forward via update_model)."""
        injected = inputs.get("fluxes")
        if injected:
            return injected
        if self._copasi is not None:
            out = self._copasi.update(
                {"species_concentrations": {}, "species_counts": {}}, interval
            )
            return out.get("fluxes", {})
        return {}

    def inputs(self):
        return {"fluxes": "map[float]", "datasets": "tyssue_data", "global_time": "float"}

    def outputs(self):
        out = {"behaviors": "list[node]"}
        for k in _EVENT_KEYS:
            out[k] = "float"
        for t in _TYPES + ["dead"]:
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
        return {t: float(sum(1 for (_, ct) in rows if ct == t)) for t in _TYPES + ["dead"]}

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
        fluxes = self._read_fluxes(inputs, interval)
        births, self._birth_acc = fractional_events(
            {t: fluxes.get(self.birth_fluxes.get(t, ""), 0.0) for t in _TYPES},
            {t: self.scales.get(f"{t}_births", 0.0) for t in _TYPES},
            self._birth_acc, dt=interval)
        deaths, self._death_acc = fractional_events(
            {t: fluxes.get(self.death_fluxes.get(t, ""), 0.0) for t in _TYPES},
            {t: self.scales.get(f"{t}_deaths", 0.0) for t in _TYPES},
            self._death_acc, dt=interval)

        if self.topology_ops:
            self._emit_topology(face_df, births, deaths, behaviors, used, fired)
        else:
            self._emit_relabel(face_df, births, deaths, behaviors, used, fired)

        return self._result(behaviors, fired, self._counts(face_df))

    # -- birth source for a target type (relabel mode): stem|healthy -> tumor,
    #    healthy -> stem, dead -> healthy (regeneration). --
    _BIRTH_SOURCES = {"tumor": ("stem", "healthy"), "stem": ("healthy",), "healthy": ("dead",)}

    def _emit_relabel(self, face_df, births, deaths, behaviors, used, fired):
        """Fixed-topology fate model (default): COPASI drives cell-type changes.
        Birth relabels a source cell into the target type; death relabels a cell
        of the dying type to 'dead'. No tyssue topology surgery."""
        for t in _TYPES:
            for _ in range(births[t]):
                src = []
                for source_type in self._BIRTH_SOURCES[t]:
                    src = select_uids(face_df, source_type, 1, exclude=used, rng_pick=self._pick)
                    if src:
                        break
                if src:
                    behaviors.append(self._differentiate(src[0], t)); used.add(src[0])
                    fired[f"{t}_births"] += 1
        for t in _TYPES:
            for _ in range(deaths[t]):
                pick = select_uids(face_df, t, 1, exclude=used, rng_pick=self._pick)
                if pick:
                    behaviors.append(self._differentiate(pick[0], "dead")); used.add(pick[0])
                    fired[f"{t}_deaths"] += 1

    def _emit_topology(self, face_df, births, deaths, behaviors, used, fired):
        """Real vertex-model ops (topology_ops=true): births split a cell
        (cell_division), deaths extrude it (remove_face). Tumor birth prefers
        differentiating a free stem cell (C->T). Requires a pandas-3-compatible
        tyssue."""
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
        for t in _TYPES:
            for _ in range(deaths[t]):
                pick = select_uids(face_df, t, 1, exclude=used, rng_pick=self._pick)
                if pick:
                    behaviors.append(self._kill(pick[0])); used.add(pick[0])
                    fired[f"{t}_deaths"] += 1

    def _result(self, behaviors, fired, counts):
        result = {"behaviors": behaviors}
        for k in _EVENT_KEYS:
            result[k] = float(fired[k])
        for t in _TYPES + ["dead"]:
            result[f"{t}_count"] = counts[t]
        return result
