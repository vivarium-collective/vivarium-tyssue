"""Compile an SBML ODE model into one vectorised numpy right-hand side.

Running COPASI or roadrunner once per cell is far too slow for a vertex model with
hundreds to thousands of cells. Instead the SBML is parsed **once**: every kinetic
law, assignment rule and rate rule is translated from its MathML AST into Python
source, and the result is compiled into a single function

    rhs(Y, P, t=0.0) -> dY

where ``Y`` has shape ``(n_state, n_cells)`` — one column per cell — and each
``P[name]`` is either a scalar or an ``(n_cells,)`` array. Per-cell inputs (a
growth factor read from the mesh) and per-cell parameter noise therefore broadcast
through the model for free, and all cells are integrated together.

Supported: species (concentration or amount semantics), global and
reaction-local parameters, constant compartments, function definitions and
initial assignments (expanded by libsbml's converters), assignment rules, rate
rules, and the usual MathML operators including ``piecewise``.

Unsupported constructs raise :class:`NotImplementedError` naming the element:
events, algebraic rules, ``delay``, non-constant compartments and
``stoichiometryMath``. The engine fails loudly rather than silently simulating
something other than the SBML.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import libsbml
except ImportError as exc:  # pragma: no cover - surfaced on first use
    libsbml = None
    _LIBSBML_ERROR = exc


def _require_libsbml():
    if libsbml is None:
        raise ImportError(
            "vivarium_tyssue.sbml needs python-libsbml (pip install python-libsbml)"
        ) from _LIBSBML_ERROR


class VectorizedSBML:
    """An SBML ODE model compiled to a numpy right-hand side over many cells.

    Parameters
    ----------
    source: str or Path
        Path to an SBML file, or an SBML document as a string.
    """

    def __init__(self, source):
        _require_libsbml()
        doc = self._read(source)
        self._convert(doc, "expandFunctionDefinitions")
        self._convert(doc, "expandInitialAssignments")
        model = doc.getModel()
        if model is None:
            raise ValueError("SBML document has no model")
        self.model_id = model.getId()
        self.model_name = model.getName() or model.getId()
        self._check_supported(model)

        # ---- symbols -------------------------------------------------------
        self.compartments = {c.getId(): c.getSize() for c in model.getListOfCompartments()}
        rate_rule_vars = {
            r.getVariable() for r in model.getListOfRules() if r.isRate()
        }
        assigned_vars = {
            r.getVariable() for r in model.getListOfRules() if r.isAssignment()
        }
        changed_by_reactions = set()
        for rxn in model.getListOfReactions():
            for ref in list(rxn.getListOfReactants()) + list(rxn.getListOfProducts()):
                changed_by_reactions.add(ref.getSpecies())

        self._species_meta = {}
        state_ids, constants = [], {}
        for sp in model.getListOfSpecies():
            sid = sp.getId()
            value = self._species_initial_value(sp)
            self._species_meta[sid] = sp
            is_state = (
                not sp.getConstant()
                and not sp.getBoundaryCondition()
                and (sid in changed_by_reactions or sid in rate_rule_vars)
            ) or (sid in rate_rule_vars)
            if sid in assigned_vars:
                continue
            if is_state:
                state_ids.append(sid)
                constants.setdefault("__init__", {})[sid] = value
            else:
                # Never changed by any reaction or rule (e.g. a conserved total that
                # only appears as a modifier) -> a constant, overridable through P.
                constants[sid] = value
        init = constants.pop("__init__", {})

        for p in model.getListOfParameters():
            pid = p.getId()
            if pid in assigned_vars:
                continue
            if pid in rate_rule_vars:
                state_ids.append(pid)
                init[pid] = p.getValue()
            else:
                constants[pid] = p.getValue()
        for cid, size in self.compartments.items():
            if cid not in assigned_vars:
                constants[cid] = size

        self.state_ids = state_ids
        self._state_index = {sid: i for i, sid in enumerate(state_ids)}
        self._init = np.array([init[sid] for sid in state_ids], dtype=float)
        # Clip species at zero during integration; rate-rule parameters may go negative.
        self._nonneg = np.array(
            [sid in self._species_meta for sid in state_ids], dtype=bool
        )

        # ---- reactions (collect local parameters before compiling) -----------
        self._reactions = []
        for rxn in model.getListOfReactions():
            kl = rxn.getKineticLaw()
            if kl is None or kl.getMath() is None:
                raise NotImplementedError(f"reaction {rxn.getId()!r} has no kinetic law")
            locals_ = {}
            local_params = (
                kl.getListOfLocalParameters() if model.getLevel() >= 3
                else kl.getListOfParameters()
            )
            for lp in local_params:
                key = f"{rxn.getId()}__{lp.getId()}"
                constants[key] = lp.getValue()
                locals_[lp.getId()] = key
            stoich = {}
            for sign, refs in ((-1.0, rxn.getListOfReactants()), (1.0, rxn.getListOfProducts())):
                for ref in refs:
                    if ref.isSetStoichiometryMath():
                        raise NotImplementedError(
                            f"reaction {rxn.getId()!r}: stoichiometryMath is not supported"
                        )
                    sid = ref.getSpecies()
                    stoich[sid] = stoich.get(sid, 0.0) + sign * ref.getStoichiometry()
            self._reactions.append((rxn.getId(), kl.getMath(), locals_, stoich))

        self._defaults = constants
        self._assignment_rules = self._order_assignment_rules(model)
        self._rate_rules = [
            (r.getVariable(), r.getMath()) for r in model.getListOfRules() if r.isRate()
        ]
        self.source = self._generate_source()
        namespace = {"np": np, "math": math}
        exec(compile(self.source, f"<sbml:{self.model_id}>", "exec"), namespace)
        self._rhs = namespace["_rhs"]

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _read(source):
        text = str(source)
        if text.lstrip().startswith("<"):
            doc = libsbml.readSBMLFromString(text)
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"SBML file not found: {path}")
            doc = libsbml.readSBMLFromFile(str(path))
        n_err = doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) + doc.getNumErrors(
            libsbml.LIBSBML_SEV_FATAL
        )
        if n_err:
            raise ValueError(f"SBML read errors:\n{doc.getErrorLog().toString()}")
        return doc

    @staticmethod
    def _convert(doc, option):
        props = libsbml.ConversionProperties()
        props.addOption(option, True)
        if doc.convert(props) != libsbml.LIBSBML_OPERATION_SUCCESS:
            raise ValueError(f"libsbml conversion {option!r} failed")

    @staticmethod
    def _check_supported(model):
        if model.getNumEvents():
            ids = [e.getId() or f"event#{i}" for i, e in enumerate(model.getListOfEvents())]
            raise NotImplementedError(f"SBML events are not supported: {ids}")
        for rule in model.getListOfRules():
            if rule.isAlgebraic():
                raise NotImplementedError("SBML algebraic rules are not supported")
        for comp in model.getListOfCompartments():
            if not comp.getConstant():
                raise NotImplementedError(
                    f"non-constant compartment {comp.getId()!r} is not supported"
                )

    def _species_initial_value(self, sp):
        """Initial value in the units the species symbol denotes in math."""
        volume = self.compartments.get(sp.getCompartment(), 1.0)
        if sp.getHasOnlySubstanceUnits():
            if sp.isSetInitialAmount():
                return sp.getInitialAmount()
            return sp.getInitialConcentration() * volume
        if sp.isSetInitialConcentration():
            return sp.getInitialConcentration()
        if sp.isSetInitialAmount():
            return sp.getInitialAmount() / volume
        return 0.0

    def _order_assignment_rules(self, model):
        """Assignment rules in dependency order (SBML allows any document order)."""
        rules = {
            r.getVariable(): r.getMath() for r in model.getListOfRules() if r.isAssignment()
        }
        ordered, done = [], set()

        def names(node, out):
            if node is None:
                return out
            if node.getType() == libsbml.AST_NAME:
                out.add(node.getName())
            for i in range(node.getNumChildren()):
                names(node.getChild(i), out)
            return out

        def visit(var, stack):
            if var in done:
                return
            if var in stack:
                raise ValueError(f"circular assignment rules through {var!r}")
            for dep in names(rules[var], set()):
                if dep in rules:
                    visit(dep, stack | {var})
            done.add(var)
            ordered.append((var, rules[var]))

        for var in rules:
            visit(var, frozenset())
        return ordered

    # --------------------------------------------------------------- codegen
    def _symbol(self, name, locals_):
        if name in locals_:
            return f"P[{locals_[name]!r}]"
        if name in self._state_index:
            return f"Y[{self._state_index[name]}]"
        if name in self._assigned:
            return f"_a_{self._assigned[name]}"
        if name in self._defaults:
            return f"P[{name!r}]"
        raise KeyError(f"unknown SBML symbol {name!r}")

    def _expr(self, node, locals_=None):
        locals_ = locals_ or {}
        t = node.getType()
        kids = [node.getChild(i) for i in range(node.getNumChildren())]
        e = lambda n: self._expr(n, locals_)  # noqa: E731

        if t in (libsbml.AST_INTEGER,):
            return repr(float(node.getInteger()))
        if t in (libsbml.AST_REAL, libsbml.AST_REAL_E, libsbml.AST_RATIONAL):
            return repr(float(node.getReal()))
        if t == libsbml.AST_NAME:
            return self._symbol(node.getName(), locals_)
        if t == libsbml.AST_NAME_TIME:
            return "t"
        if t == libsbml.AST_NAME_AVOGADRO:
            return repr(6.02214076e23)
        if t == libsbml.AST_CONSTANT_PI:
            return repr(math.pi)
        if t == libsbml.AST_CONSTANT_E:
            return repr(math.e)
        if t == libsbml.AST_CONSTANT_TRUE:
            return "True"
        if t == libsbml.AST_CONSTANT_FALSE:
            return "False"

        if t == libsbml.AST_PLUS:
            return "(" + " + ".join(e(k) for k in kids) + ")" if kids else "0.0"
        if t == libsbml.AST_MINUS:
            if len(kids) == 1:
                return f"(-{e(kids[0])})"
            return f"({e(kids[0])} - {e(kids[1])})"
        if t == libsbml.AST_TIMES:
            return "(" + " * ".join(e(k) for k in kids) + ")" if kids else "1.0"
        if t == libsbml.AST_DIVIDE:
            return f"({e(kids[0])} / {e(kids[1])})"
        if t in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
            return f"({e(kids[0])} ** {e(kids[1])})"

        unary = {
            libsbml.AST_FUNCTION_EXP: "np.exp",
            libsbml.AST_FUNCTION_LN: "np.log",
            libsbml.AST_FUNCTION_ABS: "np.abs",
            libsbml.AST_FUNCTION_FLOOR: "np.floor",
            libsbml.AST_FUNCTION_CEILING: "np.ceil",
            libsbml.AST_FUNCTION_SIN: "np.sin",
            libsbml.AST_FUNCTION_COS: "np.cos",
            libsbml.AST_FUNCTION_TAN: "np.tan",
            libsbml.AST_FUNCTION_TANH: "np.tanh",
            libsbml.AST_FUNCTION_ARCTAN: "np.arctan",
        }
        if t in unary:
            return f"{unary[t]}({e(kids[0])})"
        if t == libsbml.AST_FUNCTION_LOG:
            # log(x) is base 10; log(base, x) has the base as the first child.
            if len(kids) == 1:
                return f"np.log10({e(kids[0])})"
            return f"(np.log({e(kids[1])}) / np.log({e(kids[0])}))"
        if t == libsbml.AST_FUNCTION_ROOT:
            if len(kids) == 1:
                return f"np.sqrt({e(kids[0])})"
            return f"({e(kids[1])} ** (1.0 / {e(kids[0])}))"
        if t == getattr(libsbml, "AST_FUNCTION_MAX", None):
            return "np.maximum.reduce([" + ", ".join(e(k) for k in kids) + "])"
        if t == getattr(libsbml, "AST_FUNCTION_MIN", None):
            return "np.minimum.reduce([" + ", ".join(e(k) for k in kids) + "])"

        if t == libsbml.AST_FUNCTION_PIECEWISE:
            # (value, condition) pairs, then an optional otherwise value.
            otherwise = e(kids[-1]) if len(kids) % 2 else "np.nan"
            expr = otherwise
            for i in range(len(kids) // 2 * 2 - 2, -1, -2):
                expr = f"np.where({e(kids[i + 1])}, {e(kids[i])}, {expr})"
            return expr

        relational = {
            libsbml.AST_RELATIONAL_EQ: "==", libsbml.AST_RELATIONAL_NEQ: "!=",
            libsbml.AST_RELATIONAL_GT: ">", libsbml.AST_RELATIONAL_GEQ: ">=",
            libsbml.AST_RELATIONAL_LT: "<", libsbml.AST_RELATIONAL_LEQ: "<=",
        }
        if t in relational:
            op = relational[t]
            pairs = [f"({e(a)} {op} {e(b)})" for a, b in zip(kids, kids[1:])]
            return pairs[0] if len(pairs) == 1 else "np.logical_and.reduce([" + ", ".join(pairs) + "])"
        logical = {
            libsbml.AST_LOGICAL_AND: "np.logical_and.reduce",
            libsbml.AST_LOGICAL_OR: "np.logical_or.reduce",
            libsbml.AST_LOGICAL_XOR: "np.logical_xor.reduce",
        }
        if t in logical:
            return f"{logical[t]}([" + ", ".join(e(k) for k in kids) + "])"
        if t == libsbml.AST_LOGICAL_NOT:
            return f"np.logical_not({e(kids[0])})"

        if t == libsbml.AST_FUNCTION_DELAY:
            raise NotImplementedError("SBML delay() is not supported")
        raise NotImplementedError(
            f"unsupported SBML math: {libsbml.formulaToL3String(node)!r} (AST type {t})"
        )

    def _generate_source(self):
        self._assigned = {var: i for i, (var, _) in enumerate(self._assignment_rules)}
        lines = ["def _rhs(Y, P, t=0.0):"]
        for var, math_ in self._assignment_rules:
            lines.append(f"    _a_{self._assigned[var]} = {self._expr(math_)}  # {var}")
        lines.append("    dY = np.zeros(Y.shape, dtype=float)")
        for i, (rid, math_, locals_, stoich) in enumerate(self._reactions):
            lines.append(f"    _r{i} = {self._expr(math_, locals_)}  # {rid}")
            for sid, coef in stoich.items():
                if sid not in self._state_index or coef == 0.0:
                    continue
                scale = coef
                sp = self._species_meta[sid]
                if not sp.getHasOnlySubstanceUnits():
                    # Reaction rates are amount/time; convert to concentration/time.
                    lines.append(
                        f"    dY[{self._state_index[sid]}] += {scale!r} * _r{i} / P[{sp.getCompartment()!r}]"
                    )
                else:
                    lines.append(f"    dY[{self._state_index[sid]}] += {scale!r} * _r{i}")
        for var, math_ in self._rate_rules:
            lines.append(f"    dY[{self._state_index[var]}] = {self._expr(math_)}  # rate rule {var}")
        lines.append("    return dY")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------- API
    @property
    def n_state(self):
        return len(self.state_ids)

    def index(self, state_id):
        """Row of ``state_id`` in ``Y``."""
        return self._state_index[state_id]

    def initial_state(self):
        """The SBML initial values, shape ``(n_state,)``."""
        return self._init.copy()

    def default_params(self):
        """Every constant the RHS reads (parameters, compartments, constant species,
        ``<reaction>__<local>`` parameters) at its SBML value."""
        return dict(self._defaults)

    def params(self, overrides=None):
        """``default_params()`` updated with ``overrides`` (unknown names raise)."""
        p = self.default_params()
        for key, value in (overrides or {}).items():
            if key not in p:
                raise KeyError(f"{key!r} is not a constant of SBML model {self.model_id!r}")
            p[key] = value
        return p

    def rhs(self, Y, P, t=0.0):
        """Time derivative of ``Y`` (shape ``(n_state, n_cells)``)."""
        return self._rhs(Y, P, t)

    def integrate(self, Y, P, duration, max_step=0.01, t0=0.0):
        """Advance ``Y`` by ``duration`` model time units with clipped explicit Euler.

        ``ceil(duration / max_step)`` equal substeps; species are clipped at zero
        after each (cell-cycle models drive some species to ~1e-5, where an
        unclipped step can overshoot slightly negative). Returns a new array.
        """
        Y = np.array(Y, dtype=float, copy=True)
        if duration <= 0:
            return Y
        n = max(1, int(math.ceil(duration / max_step - 1e-9)))
        h = duration / n
        nonneg = self._nonneg
        for k in range(n):
            Y += h * self._rhs(Y, P, t0 + k * h)
            Y[nonneg] = np.maximum(Y[nonneg], 0.0)
        if not np.all(np.isfinite(Y)):
            raise FloatingPointError(
                f"SBML model {self.model_id!r} diverged (max_step={max_step}); "
                "reduce max_step"
            )
        return Y

    def trajectory(self, y0, P, duration, max_step=0.01):
        """Single-cell trajectory: ``(times, states)`` with states ``(n_steps+1, n_state)``."""
        n = max(1, int(math.ceil(duration / max_step - 1e-9)))
        h = duration / n
        Y = np.array(y0, dtype=float).reshape(-1, 1)
        out = np.empty((n + 1, self.n_state))
        out[0] = Y[:, 0]
        for k in range(n):
            Y += h * self._rhs(Y, P, k * h)
            Y[self._nonneg] = np.maximum(Y[self._nonneg], 0.0)
            out[k + 1] = Y[:, 0]
        return np.linspace(0.0, duration, n + 1), out

    def limit_cycle_samples(self, P, n, burn_in=200.0, window=100.0, max_step=0.01, rng=None):
        """``n`` states sampled at random times along one cell's long-time trajectory.

        Burns in from the SBML initial state for ``burn_in`` model time units, then
        samples uniformly over the next ``window``. For an oscillator (window longer
        than a period) this gives uniformly random phases on the limit cycle; for a
        model at a steady state it returns copies of that state. Shape ``(n_state, n)``.
        """
        rng = np.random.default_rng() if rng is None else rng
        y = self.integrate(self.initial_state().reshape(-1, 1), P, burn_in, max_step)[:, 0]
        _, states = self.trajectory(y, P, window, max_step)
        picks = rng.integers(0, len(states), size=n)
        return states[picks].T.copy()
