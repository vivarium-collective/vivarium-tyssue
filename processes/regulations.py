import random
import math
import numpy as np

from process_bigraph import Process, Step

from vivarium_tyssue.maps import *

from tyssue import config
from tyssue.draw import create_gif


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

class TestRegulations(Process):

    config_schema = {
        "period": "float",
        "geom": "string",
        "growth_rate": "float",
        "crit_area": "float",
        "double": "boolean",
    }

    def initialize(self, config):
        self.period = self.config["period"]

    def inputs(self):
        return {
            "global_time": "float",
            "datasets": {
            "_type": "tyssue_data",
            },
        }

    def outputs(self):
        return {
            "behaviors": "behaviors"
        }

    def update(self, inputs, interval):
        print(inputs["global_time"])
        t = inputs["global_time"]
        on_period = (math.isclose(t % self.period, 0) or math.isclose(t % self.period, self.period)) and not math.isclose(t, 0)
        if on_period:
            faces = inputs["datasets"]["face_df"]
            positive_y = faces.loc[(faces["y"] > 1) & (faces["z"] > -7) & (faces["z"] < 7)]
            def pick():
                return random.choice(positive_y.index)

            base = {
                "geom": self.config["geom"],
                "crit_area": self.config["crit_area"],
                "growth_rate": self.config["growth_rate"],
                "dt": interval,
                "func": "division",
            }

            if self.config["double"]:
                update = {
                    "division1": {**base, "cell_id": pick()},
                    "division2": {**base, "cell_id": pick()},
                }
            else:
                update = {
                    "division": {**base, "cell_id": pick()}
                }
        else:
            update = {}

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
            "behaviors": "behaviors"
        }

    def update(self, inputs, interval):
        if len(inputs["datasets"]["edge_df"]) > 0:
            tension = np.array(inputs["datasets"]["edge_df"]["line_tension"])
            unique_ids = np.array(inputs["datasets"]["edge_df"]["unique_id"])
            decay = np.exp(-interval / self.tau)
            noise_scale = self.sigma * np.sqrt(1 - np.exp(-2 * interval / self.tau))
            new_tension = list(decay * tension + noise_scale * np.random.randn(len(tension)))
            tension_update = {unique_id:tension_v for unique_id, tension_v in zip(unique_ids, new_tension)}

            behavior = {
                "func": "update_tension",
                "tension_update": tension_update,
            }

            update = {"update_tension": behavior}
        else:
            update = {}
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
            "behaviors": "behaviors"
        }

    def update(self, inputs, interval):
        if math.isclose(inputs["global_time"], self.trigger_time):
            behavior = {
                "func": "cell_jamming",
                "rate": self.rate,
                "limits": self.limits,
                "dt": interval,
            }
            update = {"cell_jamming": behavior}
        else:
            update = {}
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
            "behaviors": "behaviors"
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
            behavior = {
                "func": "apply_gradient",
                "parameter_updates": parameter_updates,
            }
            update = {"apply_gradient": behavior}
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
            "behaviors": "behaviors"
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
            "behaviors": {
                "update_tension": {
                    "func": "update_tension",
                    "tension_update": tension_update,
                }
            }
        }

