"""A subcellular SBML model running inside every cell of the vertex model.

:class:`SBMLCellCycle` compiles a single-cell SBML ODE model once
(:class:`vivarium_tyssue.sbml.VectorizedSBML`) and integrates one copy of it per
living face, all cells at once. Its only coupling to the tissue is the one every
other regulation process uses: it reads ``datasets`` and writes ``behaviors``. The
EulerSolver is untouched.

Coupling, mesh -> model
    Any constant of the SBML model can be driven per cell from a ``face_df``
    column, or from the ratio of two (``"area/prefered_area"``, a cell's areal
    strain). ``input_columns`` names the column, ``input_functions`` picks a
    transform from :data:`~vivarium_tyssue.processes.regulations.GRADIENT_MAP`
    (linear / exponential / hill), and ``input_args`` supplies its arguments. For
    example ``GF <- hill(area / prefered_area)`` gives stretched cells growth
    factor and starves relaxed or compressed ones, so density-dependent arrest
    emerges from the mechanics. Strain rather than raw area matters: a growing
    cell has an inflated ``prefered_area``, so it does not read its own growth as
    stretch.

Coupling, model -> mesh
    Each step every cell is assigned a cell-cycle phase from its own state by
    ``phases``: ordered rules ``"NAME: cond & cond & ..."``, each condition a
    comparison of a state variable with a number, first match wins, otherwise
    ``default_phase``. Then
      * cells in a ``growth_phases`` phase grow their target area (and, with
        ``scale_perimeter``, their target perimeter as its square root) at
        ``growth_rate`` -- the ``grow_cells`` behavior;
      * a cell that has just entered ``division_phase`` is divided at once -- the
        ``split_cells`` behavior; both halves restart from ``birth_area``.
    Cells in the other phases keep their target shape. Cells seeded from the limit
    cycle (``initial_state="limit_cycle"``) also start at the target size they
    would have reached by their phase: a cell seeded late in G2 has already grown
    for most of its G1 + G2 time, so it does not enter M (and split) at birth size.
    With the Gérard & Goldbeter
    phases (G1, S, G2, M), cells grow in G1 and G2, hold in S and M, and divide on
    entering M.

Per-cell state and inheritance
    The process owns the authoritative per-cell state, keyed by ``unique_id``.
    Every update it also writes each state variable to a ``face_df`` column
    ``<record_prefix><species>`` through the ``apply_gradient`` behavior. Those
    columns land in the solver's History. The mother keeps its ``unique_id``, so a
    uid that appears after the first update is a daughter:
      * born by ``split_cells``, it carries ``mother_uid`` and starts from the
        mother's own state, phase and division time;
      * born any other way (e.g. ``grow_divide``), it starts from the state
        columns tyssue's ``cell_division`` copied from the mother's face row. Those
        can be a step old: the EventManager shuffles queued behaviors, so the split
        may run before that step's ``apply_gradient``.
    The columns must be declared in the EulerSolver ``parameters['face_df']`` or
    the History won't record them.
    ``<record_prefix>phase`` holds the phase as an index into :attr:`phase_names`
    and ``<record_prefix>last_division`` the sim time of the cell's latest
    division, for quiescence read-outs.

Time
    ``time_scale`` is model time units per sim time unit. With the Gérard &
    Goldbeter skeleton (period ~23.6 model units), ``time_scale=2.36`` gives a
    ~10 sim-time-unit cycle.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from process_bigraph import Process

from vivarium_tyssue.processes.regulations import GRADIENT_MAP
from vivarium_tyssue.sbml import VectorizedSBML


class SBMLCellCycle(Process):
    """Per-cell SBML ODE model whose cell-cycle phase decides growth and division."""

    config_schema = {
        "model_source": "string",                 # path to the SBML file
        "time_scale": {"_type": "float", "_default": 1.0},   # model time per sim time
        "max_step": {"_type": "float", "_default": 0.01},    # Euler substep (model time)
        # Constant overrides of SBML constants, applied to every cell.
        "parameters": "map[float]",
        # Per-cell inputs: SBML constant -> face_df column (or "col_a/col_b" ratio),
        # transform, transform args. With no function the column value is used as is.
        "input_columns": "map[string]",
        "input_functions": "map[string]",
        "input_args": "map[map[float]]",
        # SBML constant -> coefficient of variation of a per-cell lognormal factor
        # (mean 1). Drawn when a uid is first seen, so daughters get fresh draws.
        "parameter_noise": "map[float]",
        # Phase rules, checked in order: "NAME: cond & cond", each cond a
        # comparison of a state variable with a number. No match -> default_phase.
        "phases": {"_type": "list[string]", "_default": []},
        "default_phase": {"_type": "string", "_default": "G1"},
        # Phases in which cells grow (grow_cells), and the phase whose entry
        # divides a cell (split_cells).
        "growth_phases": {"_type": "list[string]", "_default": ["G1", "G2"]},
        "division_phase": {"_type": "string", "_default": "M"},
        "growth_rate": {"_type": "float", "_default": 0.1},   # per unit sim time
        "scale_perimeter": {"_type": "boolean", "_default": True},
        "birth_area": {"_type": "float", "_default": 1.0},    # target area of each half
        # Cell types that neither grow nor divide (the model still runs), e.g. a
        # fixed frame of cells anchoring a pinned border.
        "no_division_types": {"_type": "list[string]", "_default": []},
        "geom": {"_type": "string", "_default": "SheetGeometry"},
        # "limit_cycle": random phases along a burnt-in single-cell trajectory, each
        # cell's target area scaled to the growth its phase implies;
        # "sbml": every cell starts at the SBML initial values and its own targets.
        "initial_state": {"_type": "string", "_default": "limit_cycle"},
        "burn_in": {"_type": "float", "_default": 200.0},
        "sample_window": {"_type": "float", "_default": 100.0},
        "record_prefix": {"_type": "string", "_default": "cc_"},
    }

    def initialize(self, config):
        self.model = VectorizedSBML(config["model_source"])
        self.time_scale = float(config["time_scale"])
        self.max_step = float(config["max_step"])
        self.base_params = self.model.params(config.get("parameters") or {})

        self.input_columns = dict(config.get("input_columns") or {})
        self.input_functions = dict(config.get("input_functions") or {})
        self.input_args = {k: dict(v) for k, v in (config.get("input_args") or {}).items()}
        for name in self.input_columns:
            if name not in self.base_params:
                raise KeyError(f"input {name!r} is not a constant of the SBML model")
        for name, func in self.input_functions.items():
            if name not in self.input_columns:
                raise KeyError(f"input_functions[{name!r}] has no input_columns entry")
            if func not in GRADIENT_MAP:
                raise KeyError(f"unknown input function {func!r}; expected one of {sorted(GRADIENT_MAP)}")

        self.noise = dict(config.get("parameter_noise") or {})
        for name in self.noise:
            if name not in self.base_params:
                raise KeyError(f"parameter_noise {name!r} is not a constant of the SBML model")

        self.phase_names, self._phase_rules = self._compile_phases(
            config.get("phases") or [], config.get("default_phase", "G1"))
        division_phase = config.get("division_phase", "M")
        growth_phases = config.get("growth_phases", ["G1", "G2"])
        for name in [division_phase, *growth_phases]:
            if name not in self.phase_names:
                raise ValueError(f"phase {name!r} is not one of {self.phase_names}")
        self.division_code = self.phase_names.index(division_phase)
        self.growth_codes = [self.phase_names.index(n) for n in growth_phases]
        self.growth_rate = float(config.get("growth_rate", 0.1))
        self.scale_perimeter = bool(config.get("scale_perimeter", True))
        self.birth_area = float(config.get("birth_area", 1.0))
        self.geom = config.get("geom") or "SheetGeometry"

        self.initial_mode = config["initial_state"]
        if self.initial_mode not in ("limit_cycle", "sbml"):
            raise ValueError("initial_state must be 'limit_cycle' or 'sbml'")
        self.skip_types = {"dividing", "extruding", *(config.get("no_division_types") or [])}
        self.prefix = config["record_prefix"]
        self.columns = [self.prefix + sid for sid in self.model.state_ids]
        self.phase_column = self.prefix + "phase"
        self.division_column = self.prefix + "last_division"

        # Per-cell state, aligned by position with self.uids.
        self.uids = np.empty(0, dtype=np.int64)
        self.Y = np.empty((self.model.n_state, 0))
        self.phase = np.empty(0, dtype=np.int64)
        self.last_division = np.empty(0)
        self.factors = {name: np.empty(0) for name in self.noise}
        # uid -> target-area factor still to apply to a freshly seeded cell.
        self._initial_size = {}
        self._started = False
        self._warned_orphans = False
        self.n_divisions = 0
        self.n_skipped = 0

    def inputs(self):
        return {
            "global_time": "float",
            "datasets": {"_type": "tyssue_data"},
        }

    def outputs(self):
        return {"behaviors": "list[node]"}

    # ------------------------------------------------------------------ phases
    def _compile_phases(self, rules, default):
        """``"NAME: cond & cond"`` rules -> (phase names, [(code, compiled), ...]).
        Each condition is parenthesised before joining, so ``a > 1 & b < 2`` means
        ``(a > 1) & (b < 2)`` rather than Python's ``a > (1 & b) < 2``."""
        names, compiled = [default], []
        for rule in rules:
            name, sep, expr = rule.partition(":")
            name = name.strip()
            if not sep or not name or not expr.strip():
                raise ValueError(f"phase rule {rule!r} is not 'NAME: condition'")
            if name in names:
                raise ValueError(f"phase {name!r} is defined twice")
            conds = [c.strip() for c in expr.split("&")]
            code = compile(" & ".join(f"({c})" for c in conds), f"<phase {name}>", "eval")
            unknown = set(code.co_names) - set(self.model.state_ids)
            if unknown:
                raise KeyError(f"phase {name!r} uses {sorted(unknown)}, which are not "
                               f"state variables of the SBML model")
            names.append(name)
            compiled.append((len(names) - 1, code))
        return names, compiled

    def classify(self, Y):
        """Phase index (into :attr:`phase_names`) of each column of ``Y``."""
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        env = {sid: Y[i] for i, sid in enumerate(self.model.state_ids)}
        phase = np.zeros(Y.shape[1], dtype=np.int64)
        undecided = np.ones(Y.shape[1], dtype=bool)
        for code, expr in self._phase_rules:
            hit = undecided & np.broadcast_to(eval(expr, {"__builtins__": {}}, env),
                                              undecided.shape)
            phase[hit] = code
            undecided &= ~hit
        return phase

    # ------------------------------------------------------------- per-cell setup
    def _draw_factors(self, n):
        out = {}
        for name, cv in self.noise.items():
            if cv <= 0:
                out[name] = np.ones(n)
                continue
            sigma2 = np.log1p(cv ** 2)
            out[name] = np.random.lognormal(-0.5 * sigma2, np.sqrt(sigma2), size=n)
        return out

    def _fresh_states(self, n, faces_rows, factors):
        """States for ``n`` cells with no inherited state, and the factor by which
        each cell's target area has grown since its last division."""
        if self.initial_mode == "sbml":
            return np.tile(self.model.initial_state()[:, None], (1, n)), np.ones(n)
        # One reference trajectory at the new cells' mean inputs (not one per cell).
        P = self._params(faces_rows, factors) if n else self.base_params
        P_ref = {k: (float(np.mean(v)) if np.ndim(v) else v) for k, v in P.items()}
        y = self.model.integrate(self.model.initial_state().reshape(-1, 1), P_ref,
                                 self.config["burn_in"], self.max_step)[:, 0]
        t, states = self.model.trajectory(y, P_ref, self.config["sample_window"], self.max_step)
        # Model time spent in growth phases since the last entry into the division
        # phase, at every point of the trajectory (NaN before the first entry).
        phase = self.classify(states.T)
        h = t[1] - t[0]
        grown = np.full(len(t), np.nan)
        g = np.nan
        for k in range(len(t)):
            if phase[k] == self.division_code and (k == 0 or phase[k - 1] != self.division_code):
                g = 0.0
            elif phase[k] in self.growth_codes and not np.isnan(g):
                g += h
            grown[k] = g
        valid = np.flatnonzero(~np.isnan(grown))
        if len(valid) == 0:          # no division in the window: a resting cell
            valid, grown = np.arange(len(t)), np.zeros(len(t))
        # Draw from numpy's global stream so np.random.seed makes runs reproducible.
        rng = np.random.default_rng(np.random.randint(2**31 - 1))
        picks = valid[rng.integers(0, len(valid), size=n)]
        size = np.exp(self.growth_rate * grown[picks] / self.time_scale)
        return states[picks].T.copy(), size

    def _inherited_states(self, rows):
        """State columns copied onto daughters from their mothers, or None where the
        face row carries no usable state (columns missing or never written)."""
        if not all(c in rows.columns for c in self.columns):
            return None, np.zeros(len(rows), dtype=bool)
        Y = rows[self.columns].to_numpy(dtype=float).T
        ok = np.all(np.isfinite(Y), axis=0) & (np.abs(Y).sum(axis=0) > 0)
        return Y, ok

    def _sync(self, faces, time):
        """Align internal per-cell state with the living faces (in face order)."""
        uids = faces["unique_id"].to_numpy(dtype=np.int64)
        pos = pd.Index(self.uids).get_indexer(uids)
        known = pos >= 0
        n_new = int((~known).sum())

        Y = np.empty((self.model.n_state, len(uids)))
        phase = np.zeros(len(uids), dtype=np.int64)
        last_division = np.empty(len(uids))
        factors = {k: np.empty(len(uids)) for k in self.noise}
        Y[:, known] = self.Y[:, pos[known]]
        phase[known] = self.phase[pos[known]]
        last_division[known] = self.last_division[pos[known]]
        for k in self.noise:
            factors[k][known] = self.factors[k][pos[known]]

        if n_new:
            new = ~known
            rows = faces.loc[new]
            new_factors = self._draw_factors(n_new)
            for k in self.noise:
                factors[k][new] = new_factors[k]
            fresh = np.ones(n_new, dtype=bool)
            Y_new = np.empty((self.model.n_state, n_new))
            div_new = np.full(n_new, time)
            phase_new = None
            if self._started:
                inherited, ok = self._inherited_states(rows)
                if inherited is not None:
                    Y_new[:, ok] = inherited[:, ok]
                    fresh = ~ok
                    if self.division_column in rows.columns:
                        div_new[ok] = rows[self.division_column].to_numpy(dtype=float)[ok]
                # A daughter of split_cells takes its mother's live state instead.
                if "mother_uid" in rows.columns:
                    mpos = pd.Index(self.uids).get_indexer(
                        rows["mother_uid"].fillna(-1).to_numpy(dtype=np.int64))
                    born = mpos >= 0
                    Y_new[:, born] = self.Y[:, mpos[born]]
                    div_new[born] = self.last_division[mpos[born]]
                    phase_new = np.where(born, self.phase[np.maximum(mpos, 0)], -1)
                    fresh &= ~born
                if fresh.any() and not self._warned_orphans:
                    warnings.warn(
                        f"SBMLCellCycle: {int(fresh.sum())} new cell(s) carry no inherited "
                        f"state; declare {self.columns} in the EulerSolver face_df "
                        "parameters. Seeding them from the limit cycle.",
                        stacklevel=2,
                    )
                    self._warned_orphans = True
            if fresh.any():
                sub = {k: v[fresh] for k, v in new_factors.items()}
                Y_new[:, fresh], size = self._fresh_states(int(fresh.sum()), rows.loc[fresh], sub)
                uid_new = uids[new][fresh]
                self._initial_size.update(zip(uid_new.tolist(), size.tolist()))
            Y[:, new] = Y_new
            last_division[new] = div_new
            # A new cell starts in the phase its state is in (a split_cells daughter
            # in its mother's), so a daughter born in the division phase is not
            # divided again on its first step.
            classified = self.classify(Y_new)
            if phase_new is not None:
                classified = np.where(phase_new >= 0, phase_new, classified)
            phase[new] = classified

        self.uids, self.Y, self.phase, self.last_division, self.factors = (
            uids, Y, phase, last_division, factors,
        )
        self._started = True

    @staticmethod
    def _column(faces, column):
        """A face_df column, or the ratio of two (``"area/prefered_area"``)."""
        if "/" in column:
            num, den = (c.strip() for c in column.split("/", 1))
            return faces[num].to_numpy(dtype=float) / faces[den].to_numpy(dtype=float)
        return faces[column].to_numpy(dtype=float)

    def _params(self, faces, factors):
        P = dict(self.base_params)
        for name, column in self.input_columns.items():
            x = self._column(faces, column)
            func = self.input_functions.get(name)
            P[name] = GRADIENT_MAP[func](x, **self.input_args.get(name, {})) if func else x
        for name, f in factors.items():
            P[name] = P[name] * f
        return P

    def _size_updates(self, faces, updates):
        """Scale freshly seeded cells' targets to the growth their phase implies."""
        size = {u: f for u, f in self._initial_size.items() if f != 1.0}
        self._initial_size = {}
        if not size or "prefered_area" not in faces.columns:
            return
        rows = faces[faces["unique_id"].isin(size)]
        f = rows["unique_id"].map(size).to_numpy(dtype=float)
        uid = rows["unique_id"].tolist()
        updates["prefered_area"] = {
            "dataframe": "face",
            "update": dict(zip(uid, (rows["prefered_area"].to_numpy(dtype=float) * f).tolist())),
        }
        if self.scale_perimeter and "prefered_perimeter" in faces.columns:
            perim = rows["prefered_perimeter"].to_numpy(dtype=float) * np.sqrt(f)
            updates["prefered_perimeter"] = {"dataframe": "face",
                                             "update": dict(zip(uid, perim.tolist()))}

    # ------------------------------------------------------------------- update
    def update(self, inputs, interval):
        faces = inputs["datasets"]["face_df"]
        if len(faces) == 0:
            return {"behaviors": []}
        if "is_alive" in faces.columns:
            faces = faces[faces["is_alive"] > 0]
        time = float(inputs.get("global_time", 0.0) or 0.0)

        self._sync(faces, time)
        P = self._params(faces, self.factors)
        self.Y = self.model.integrate(self.Y, P, interval * self.time_scale, self.max_step)

        phase = self.classify(self.Y)
        entered = (phase == self.division_code) & (self.phase != self.division_code)
        growing = np.isin(phase, self.growth_codes)
        self.phase = phase

        if "cell_type" in faces.columns:
            blocked = faces["cell_type"].isin(self.skip_types).to_numpy()
        else:
            blocked = np.zeros(len(faces), dtype=bool)
        divide = entered & ~blocked
        growing &= ~blocked
        self.n_skipped += int((entered & blocked).sum())
        self.n_divisions += int(divide.sum())
        self.last_division[divide] = time + interval

        behaviors = []
        if growing.any():
            behaviors.append({
                "func": "grow_cells",
                "cell_uids": self.uids[growing].tolist(),
                "growth_rate": self.growth_rate,
                "dt": interval,
                "scale_perimeter": self.scale_perimeter,
            })
        if divide.any():
            behaviors.append({
                "func": "split_cells",
                "geom": self.geom,
                "cell_uids": self.uids[divide].tolist(),
                "prefered_area": self.birth_area,
            })

        uids = self.uids.tolist()
        updates = {
            col: {"dataframe": "face", "update": dict(zip(uids, self.Y[j].tolist()))}
            for j, col in enumerate(self.columns)
        }
        updates[self.phase_column] = {
            "dataframe": "face", "update": dict(zip(uids, self.phase.astype(float).tolist())),
        }
        updates[self.division_column] = {
            "dataframe": "face", "update": dict(zip(uids, self.last_division.tolist())),
        }
        self._size_updates(faces, updates)
        behaviors.append({"func": "apply_gradient", "parameter_updates": updates})
        return {"behaviors": behaviors}
