import math
import numpy as np

from process_bigraph import Process, Step

from vivarium_tyssue.maps import *
from vivarium_tyssue.behaviors.behaviors import DEATH_FLOOR

from tyssue import config


def linear_gradient(x, m, c):
    """Simple linear gradient function"""
    return m * x + c

def exponential_gradient(x, a, c):
    """Simple exponential gradient function"""
    return a ** x + c

def hill_gradient(x, vmax, hmax, n=1):
    """Simple hill-equation gradient function"""
    return (vmax * x**n)/(hmax**n + x**n)

GRADIENT_MAP = {
    "linear": linear_gradient,
    "exponential": exponential_gradient,
    "hill": hill_gradient,
}

def _poisson_event_count(rate, interval, n_eligible):
    """Number of events firing in ``interval`` for a Poisson process of the given
    ``rate`` (events per unit time), capped at the number of eligible cells."""
    if rate <= 0 or interval <= 0 or n_eligible == 0:
        return 0
    return int(min(np.random.poisson(rate * interval), n_eligible))


class CellDivisions(Process):
    """Randomly triggers cell divisions as a Poisson process at a fixed ``rate``.

    Formerly ``TestRegulations``, which fired divisions on a deterministic period.
    Each update draws the number of division events in the elapsed ``interval``
    from ``Poisson(rate * interval)``; that many distinct alive cells are chosen
    uniformly at random and given the ``division`` behavior (grow the preferred
    area at ``growth_rate`` until ``crit_area``, then split). When the tissue
    tracks ``cell_type`` (e.g. the crypt), the chosen cell is flagged
    ``"dividing"`` while it grows and restored to its own type on division, so it
    colour-codes exactly like the Gillespie crypt.
    """

    config_schema = {
        "rate": "float",         # mean division events per unit time (Poisson)
        "geom": "string",
        "growth_rate": "float",
        "crit_area": "float",
    }

    def initialize(self, config):
        self.rate = self.config["rate"]

    def inputs(self):
        return {
            "global_time": "float",
            "datasets": {
                "_type": "tyssue_data",
            },
        }

    def outputs(self):
        return {
            "behaviors": "list[node]"
        }

    def update(self, inputs, interval):
        faces = inputs["datasets"]["face_df"]
        if len(faces) == 0:
            return {"behaviors": []}

        eligible = faces
        if "is_alive" in faces.columns:
            eligible = eligible[eligible["is_alive"] > 0]
        has_type = "cell_type" in faces.columns
        if has_type:
            # Don't re-pick a cell that is already mid-event.
            eligible = eligible[~eligible["cell_type"].isin(["dividing", "extruding"])]

        n_events = _poisson_event_count(self.rate, interval, len(eligible))
        if n_events == 0:
            return {"behaviors": []}

        chosen = np.random.choice(eligible.index.to_numpy(), size=n_events, replace=False)
        base = {
            "func": "division",
            "geom": self.config["geom"],
            "crit_area": self.config["crit_area"],
            "growth_rate": self.config["growth_rate"],
            "dt": interval,
        }
        update = [
            {
                **base,
                "cell_uid": int(faces.loc[idx, "unique_id"]),
                "cell_type": faces.loc[idx, "cell_type"] if has_type else None,
            }
            for idx in chosen
        ]
        return {"behaviors": update}


class CellDeaths(Process):
    """Randomly triggers cell deaths (apoptotic extrusion) as a Poisson process.

    The death counterpart of :class:`CellDivisions`, using the same
    ``apoptosis_extrusion`` behavior the Gillespie process drives: each update
    draws ``Poisson(rate * interval)`` deaths, chooses that many distinct alive
    cells at random, and flags each ``"extruding"`` (shrinking its preferred area
    at ``shrink_rate`` each step until it collapses and is removed). Extruding
    cells colour-code as in the Gillespie crypt.
    """

    config_schema = {
        "rate": "float",         # mean death events per unit time (Poisson)
        "geom": "string",
        "shrink_rate": "float",
        "crit_area": "float",
        # Floor on the shrinking prefered_area that removes a cell regardless of
        # crit_area (see behaviors.DEATH_FLOOR). On a relaxed sheet the default
        # fires first; lower it below crit_area to let crit_area govern.
        "death_floor": {"_type": "float", "default": DEATH_FLOOR},
        # Constrict a dying cell's prefered_perimeter along with its prefered_area.
        # Off by default; required for a small crit_area to be reachable at all.
        "contract_perimeter": {"_type": "boolean", "default": False},
    }

    def initialize(self, config):
        self.rate = self.config["rate"]

    def inputs(self):
        return {
            "global_time": "float",
            "datasets": {
                "_type": "tyssue_data",
            },
        }

    def outputs(self):
        return {
            "behaviors": "list[node]"
        }

    def update(self, inputs, interval):
        faces = inputs["datasets"]["face_df"]
        if len(faces) == 0:
            return {"behaviors": []}

        eligible = faces
        if "is_alive" in faces.columns:
            eligible = eligible[eligible["is_alive"] > 0]
        if "cell_type" in faces.columns:
            eligible = eligible[eligible["cell_type"] != "extruding"]

        n_events = _poisson_event_count(self.rate, interval, len(eligible))
        if n_events == 0:
            return {"behaviors": []}

        chosen = np.random.choice(eligible.index.to_numpy(), size=n_events, replace=False)
        base = {
            "func": "apoptosis_extrusion",
            "geom": self.config["geom"],
            "crit_area": self.config["crit_area"],
            "shrink_rate": self.config["shrink_rate"],
            "death_floor": self.config["death_floor"],
            "contract_perimeter": self.config["contract_perimeter"],
            "dt": interval,
        }
        update = [{**base, "cell_uid": int(faces.loc[idx, "unique_id"])} for idx in chosen]
        return {"behaviors": update}

class StochasticLineTension(Process):

    config_schema = {
        "tau": {
            "_type": "float",
            "default": 1.0,
        },
        "sigma": "float",
    }

    def initialize(self, config):
        self.tau = self.config["tau"]
        self.sigma = self.config["sigma"]

    def inputs(self):
        return {
            "datasets": {
                "_type": "tyssue_data",
            },
        }

    def outputs(self):
        return {
            "behaviors": "list[node]"
        }

    def update(self, inputs, interval):
        if len(inputs["datasets"]["edge_df"]) > 0:
            tension = np.array(inputs["datasets"]["edge_df"]["line_tension"])
            unique_ids = np.array(inputs["datasets"]["edge_df"]["unique_id"])
            decay = np.exp(-interval / self.tau)
            noise_scale = self.sigma * np.sqrt(1 - np.exp(-2 * interval / self.tau))
            new_tension = list(decay * tension + noise_scale * np.random.randn(len(tension)))
            tension_update = {unique_id:tension_v for unique_id, tension_v in zip(unique_ids, new_tension)}

            update = [{
                "func": "update_tension",
                "tension_update": tension_update,
            }]

        else:
            update = []
        return {"behaviors": update}


class DifferentialAdhesion(Process):
    """Differential-adhesion cell sorting: drives the ``differential_adhesion``
    behavior, which retensions every junction from the identity of the two cells
    it separates.

    Sibling of :class:`StochasticLineTension` and :class:`AnisotropicTension` —
    it regulates the ``EulerSolver`` purely through the ``behaviors`` port and
    never touches the mesh itself.

    Where those two processes compute the new tensions in the process and ship
    them as a ``unique_id -> tension`` map, this one ships only the two tension
    VALUES and lets the behavior do the classification on the live epithelium.
    That split is deliberate: which junctions are heterotypic is a property of the
    topology as it stands at the moment the tension is applied. A T1 transition
    rewires a junction so that it separates a different pair of cells, so a table
    computed once, keyed by edge id, would put homotypic tension on a
    freshly-created heterotypic interface and stay wrong for the rest of the run.

    The behavior is emitted on every update, so the classification is redone from
    the current ``srce``/``trgt``/``face`` topology every step. Note that tyssue's
    ``EventManager.update`` shuffles its queue, so within a step this behavior runs
    either just before or just after ``reconnect``'s T1 surgery; a junction created
    by a T1 that lands after it therefore picks up its tension on the next step.
    The tensions are never more than one step behind the mesh.
    """

    config_schema = {
        # Tension on an interface between two cells of the same type (adhesive).
        "homotypic_tension": "float",
        # Tension on an interface between cells of different types. Sorting is
        # driven by heterotypic > homotypic.
        "heterotypic_tension": "float",
        # face_df column holding the cell type.
        "type_column": {"_type": "string", "default": "cell_type"},
        # Tension on half-edges with no opposite (the free border of an open
        # sheet). Applied only when apply_boundary_tension is true; a closed
        # sheet has no such edges.
        "boundary_tension": {"_type": "float", "default": 0.0},
        "apply_boundary_tension": {"_type": "boolean", "default": False},
        # edge_df column the behavior stamps with the 0/1 heterotypic flag, so the
        # classification lands in the solver's History next to the tension. It must
        # already exist when the solver builds its History — declare it in the
        # EulerSolver ``parameters`` (edge_df) — or nothing is recorded.
        "record_column": {"_type": "string", "default": "heterotypic"},
    }

    def initialize(self, config):
        self.homotypic_tension = config["homotypic_tension"]
        self.heterotypic_tension = config["heterotypic_tension"]
        self.type_column = config["type_column"]
        self.boundary_tension = (
            config["boundary_tension"] if config["apply_boundary_tension"] else None
        )
        self.record_column = config["record_column"]

    def inputs(self):
        return {
            "datasets": {
                "_type": "tyssue_data",
            },
        }

    def outputs(self):
        return {
            "behaviors": "list[node]"
        }

    def update(self, inputs, interval):
        if len(inputs["datasets"]["edge_df"]) == 0:
            return {"behaviors": []}
        return {
            "behaviors": [{
                "func": "differential_adhesion",
                "homotypic_tension": self.homotypic_tension,
                "heterotypic_tension": self.heterotypic_tension,
                "boundary_tension": self.boundary_tension,
                "type_column": self.type_column,
                "record_column": self.record_column,
            }]
        }


class CellJamming(Process):
    config_schema = {
        "trigger_time": "float",
        "rate": "float",
        "limits": "list[float]",
    }

    def initialize(self, config):
        self.trigger_time = self.config["trigger_time"]
        self.rate = self.config["rate"]
        self.limits = self.config["limits"]

    def inputs(self):
        return {
            "global_time": "float",
            "datasets": {
                "_type": "tyssue_data",
            },
        }

    def outputs(self):
        return {
            "behaviors": "list[node]"
        }

    def update(self, inputs, interval):
        if math.isclose(inputs["global_time"], self.trigger_time):
            update = [{
                "func": "cell_jamming",
                "rate": self.rate,
                "limits": self.limits,
                "dt": interval,
            }]
        else:
            update = []
        return {"behaviors": update}


class ParameterGradient(Step):
    """Creates a 1D gradient of chemical or mechanical signal"""
    config_schema = {
        "gradient_type": "string", #gradient function key for gradient function
        "axis": "string", #direction axis for gradient
        "args": "map[float]", #parameters for chosen gradient equation
        "model_parameters": "map[string]", #map of parameter names (keys) and dataframe the parameter is found in (values)
    }
    def initialize(self, config):
        self.gradient = GRADIENT_MAP[config["gradient_type"]]
        self.args = config["args"]
        self.axis = config["axis"]
        self.model_parameters = config["model_parameters"]

    def inputs(self):
        return {
            "datasets": "tyssue_data",
        }

    def outputs(self):
        return {
            "behaviors": "list[node]"
        }

    def update(self, inputs):
        parameter_updates = {}
        for parameter, df in self.model_parameters.items():
            if len(inputs["datasets"][df+"_df"]) > 0:
                if df == "edge":
                    positions = np.array((
                            inputs["datasets"]["edge_df"][f"s{self.axis}"] +
                            inputs["datasets"]["edge_df"][f"t{self.axis}"]
                        )/2
                    )
                else:
                    positions = np.array(inputs["datasets"][df+"_df"][self.axis])
                unique_ids = np.array(inputs["datasets"][df+"_df"]["unique_id"])

                new_parameter = self.gradient(x=positions, **self.args)
                parameter_update = {unique_id:parameter for unique_id, parameter in zip(unique_ids, new_parameter)}
                parameter_updates[parameter] = {
                    "dataframe" : df,
                    "update" : parameter_update,
                }
            update = [{
                "func": "apply_gradient",
                "parameter_updates": parameter_updates,
            }]
            return {"behaviors": update}

class AnisotropicTension(Step):
    config_schema = {
        "axes" : "list[string]", # list of axis labels (first axis label will be the axis of higher tension)
        "tension_values": "list[float]", #low and high value of
    }

    def initialize(self, config):

        self.axes = config["axes"]
        self.tension_values = config["tension_values"]

    def inputs(self):
        return {
            "datasets": "tyssue_data",
        }

    def outputs(self):
        return {
            "behaviors": "list[node]"
        }

    def update(self, inputs):
        edge_df = inputs["datasets"]["edge_df"]

        if edge_df.empty:
            return {"behaviors": {}}

        d1 = edge_df[f"d{self.axes[1]}"].to_numpy()
        d2 = edge_df[f"d{self.axes[0]}"].to_numpy()
        unique_ids = edge_df["unique_id"].to_numpy()

        angles = np.abs(np.arctan2(d1, d2))
        angles = np.minimum(angles, np.pi - angles)

        tensions = np.where(
            angles > np.pi / 4,
            self.tension_values[0],
            self.tension_values[1],
        )

        tension_update = dict(zip(unique_ids, tensions))

        return {
            "behaviors": [{
                "func": "update_tension",
                "tension_update": tension_update,
            }]
        }

# ---------------------------------------------------------------------------
# Directional (planar-polarised) line tension
# ---------------------------------------------------------------------------

def _alignment_cos2(theta):
    """cos^2(theta) — equivalently (1 + cos 2theta) / 2."""
    return np.cos(theta) ** 2

def _alignment_abs_cos(theta):
    """|cos(theta)| — falls off more slowly than ``cos2``, so a broader cone
    of junctions sits close to the maximum tension."""
    return np.abs(np.cos(theta))

def _alignment_linear(theta):
    """1 - 2*theta/pi — linear in the acute angle itself."""
    return 1.0 - 2.0 * theta / np.pi

ALIGNMENT_MAP = {
    "cos2": _alignment_cos2,
    "abs_cos": _alignment_abs_cos,
    "linear": _alignment_linear,
}


class DirectionalLineTension(Process):
    r"""Planar-polarised junction tension: line tension graded by the angle
    between each junction and a user-supplied polarity vector.

    This is the ingredient that drives convergent extension in a vertex model.
    Junctions lying **along** the polarity axis carry the highest tension and
    contract; junctions **perpendicular** to it carry the lowest and are free to
    grow. Repeated over the sheet the tissue narrows (converges) along the
    polarity direction and lengthens (extends) across it.

    The angle
    ---------
    Half-edges are directed but junctions are not, so the quantity that matters
    is the **acute** angle between the edge and the polarity axis,

    .. math::

        \theta = \arccos\!\big(|\hat{d} \cdot \hat{p}|\big) \in [0, \pi/2]

    with :math:`\hat d` the unit edge vector and :math:`\hat p` the unit polarity
    vector (the configured ``polarity`` is normalised on initialisation, so it
    need not be a unit vector). Taking the absolute value of the dot product is
    what makes :math:`\theta` acute and the response invariant to the sign of
    both the edge and the polarity vector — flipping ``polarity`` to ``-p`` gives
    exactly the same tensions.

    The tension
    -----------
    From :math:`\theta` an *alignment* :math:`a(\theta) \in [0, 1]` is formed —
    1 for an edge parallel to the polarity axis, 0 for a perpendicular one — and
    the tension interpolates between the two configured extremes:

    .. math::

        \Lambda(\theta) = \Lambda_{\min}
            + (\Lambda_{\max} - \Lambda_{\min})\, a(\theta)^{\,n}

    ``profile`` selects :math:`a`:

    ==============  ===========================================================
    ``cos2``        :math:`\cos^2\theta = (1 + \cos 2\theta)/2` (default). With
                    ``sharpness`` 1 this is the classical nematic form
                    :math:`\Lambda = \bar\Lambda\,(1 + \alpha \cos 2\theta)`,
                    with :math:`\bar\Lambda` the mean of the two extremes and
                    :math:`\alpha` the anisotropy
                    :math:`(\Lambda_{\max}-\Lambda_{\min})/(\Lambda_{\max}+\Lambda_{\min})`.
    ``abs_cos``     :math:`|\cos\theta|` — falls off more slowly near
                    :math:`\theta = 0`, so a broader cone of edges is close to
                    :math:`\Lambda_{\max}`.
    ``linear``      :math:`1 - 2\theta/\pi` — linear in the angle itself.
    ==============  ===========================================================

    ``sharpness`` (:math:`n`) then narrows (``n > 1``) or broadens (``n < 1``) the
    high-tension cone without changing either extreme.

    Where the work happens
    ----------------------
    Sibling of :class:`StochasticLineTension` and :class:`AnisotropicTension`, and
    like them it computes the tensions **in the process** and ships a
    ``unique_id -> tension`` map through the ``update_tension`` behavior; the
    ``EulerSolver`` itself is untouched. That is the right split here (unlike
    :class:`DifferentialAdhesion`, which must classify on the live mesh): an
    edge's tension depends only on its own geometry, which the process reads
    straight out of ``edge_df``, and edges are addressed by ``unique_id``, so a
    stale entry for an edge a T1 has since removed simply matches nothing.

    Because the sheet keeps deforming, the map is recomputed **every** update:
    an edge that rotates towards the polarity axis is retensioned accordingly,
    and an edge born of a T1 picks up its tension on the next step.
    """

    config_schema = {
        # Polarity axis. Normalised on initialisation, so any non-zero vector
        # works; its length sets the dimensionality (2 -> x/y, 3 -> x/y/z) unless
        # ``coords`` says otherwise. Sign is irrelevant (the angle is acute).
        "polarity": "list[float]",
        # Tension on a junction perpendicular to the polarity axis.
        "tension_min": {"_type": "float", "_default": 0.0},
        # Tension on a junction parallel to the polarity axis. Convergent
        # extension needs tension_max > tension_min.
        "tension_max": "float",
        # Which a(theta) to use — see ALIGNMENT_MAP / the class docstring.
        "profile": {"_type": "string", "_default": "cos2"},
        # Exponent n on the alignment. > 1 narrows the high-tension cone.
        "sharpness": {"_type": "float", "_default": 1.0},
        # Coordinate labels the edge vector is read from ("dx", "dy", ...).
        # Empty (the default) means infer from len(polarity).
        # NB: a non-empty list _default is applied twice by bigraph_schema and
        # comes back concatenated with itself, so [] is the only safe default.
        "coords": {"_type": "list[string]", "_default": []},
        # edge_df column the per-edge alignment a(theta) is written to, so the
        # polarity read-out lands in the solver's History next to the tension.
        # It must already exist when the solver builds its History — declare it
        # in the EulerSolver ``parameters`` (edge_df) — or nothing is recorded.
        # Empty disables the read-out.
        "record_column": {"_type": "string", "_default": "polar_alignment"},
    }

    def initialize(self, config):
        polarity = np.asarray(config["polarity"], dtype=float)
        norm = np.linalg.norm(polarity)
        if polarity.size == 0 or not np.isfinite(norm) or norm == 0.0:
            raise ValueError(
                f"DirectionalLineTension: 'polarity' must be a non-zero finite "
                f"vector, got {config['polarity']!r}"
            )
        self.polarity = polarity / norm

        coords = list(config["coords"]) or ["x", "y", "z"][: self.polarity.size]
        if len(coords) != self.polarity.size:
            raise ValueError(
                f"DirectionalLineTension: len(coords)={len(coords)} does not match "
                f"len(polarity)={self.polarity.size}"
            )
        self.coords = coords

        profile = config["profile"]
        if profile not in ALIGNMENT_MAP:
            raise ValueError(
                f"DirectionalLineTension: unknown profile {profile!r}; "
                f"expected one of {sorted(ALIGNMENT_MAP)}"
            )
        self.alignment = ALIGNMENT_MAP[profile]
        self.profile = profile

        self.tension_min = config["tension_min"]
        self.tension_max = config["tension_max"]
        self.sharpness = config["sharpness"]
        self.record_column = config["record_column"]

    def inputs(self):
        return {
            "datasets": {
                "_type": "tyssue_data",
            },
        }

    def outputs(self):
        return {
            "behaviors": "list[node]"
        }

    def edge_alignment(self, edge_df):
        """``a(theta)`` for every half-edge, from the ``d<coord>`` columns.

        Zero-length edges (a junction mid-collapse) have no direction; they are
        given alignment 0, i.e. the minimum tension, rather than a NaN that would
        poison the solver.
        """
        deltas = np.column_stack([
            edge_df[f"d{c}"].to_numpy(dtype=float) for c in self.coords
        ])
        lengths = np.linalg.norm(deltas, axis=1)
        good = lengths > 0

        # |cos(theta)| of the acute angle, clipped against float error before arccos.
        cos_theta = np.zeros_like(lengths)
        cos_theta[good] = np.abs(deltas[good] @ self.polarity) / lengths[good]
        np.clip(cos_theta, 0.0, 1.0, out=cos_theta)

        alignment = np.zeros_like(lengths)
        alignment[good] = self.alignment(np.arccos(cos_theta[good]))
        return np.clip(alignment, 0.0, 1.0)

    def update(self, inputs, interval):
        edge_df = inputs["datasets"]["edge_df"]
        if len(edge_df) == 0:
            return {"behaviors": []}

        alignment = self.edge_alignment(edge_df)
        tensions = self.tension_min + (
            self.tension_max - self.tension_min
        ) * alignment ** self.sharpness

        unique_ids = edge_df["unique_id"].to_numpy()
        behaviors = [{
            "func": "update_tension",
            "tension_update": dict(zip(unique_ids, tensions)),
        }]
        if self.record_column:
            # `apply_gradient` is the repo's generic "write this column of this
            # element dataframe, keyed by unique_id" behavior — exactly what the
            # read-out needs, so no new behavior is introduced for it.
            behaviors.append({
                "func": "apply_gradient",
                "parameter_updates": {
                    self.record_column: {
                        "dataframe": "edge",
                        "update": dict(zip(unique_ids, alignment)),
                    },
                },
            })
        return {"behaviors": behaviors}
