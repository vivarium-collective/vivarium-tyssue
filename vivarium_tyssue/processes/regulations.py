import math
import numpy as np

from process_bigraph import Process, Step

from vivarium_tyssue.maps import *

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