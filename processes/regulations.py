import random
import time
import math
import numpy as np
from pprint import pprint
import cProfile
import pstats
import io

from bigraph_schema import allocate_core
from process_bigraph import Process, Composite
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from vivarium_tyssue.maps import *
from vivarium_tyssue.core_maps import GEOMETRY_MAP
from vivarium_tyssue.processes.eulersolver import EulerSolver, get_test_spec, run_test_solver, get_test_config_flat, get_test_config

from tyssue import config
from tyssue.draw import create_gif

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
                "func": "stochastic_tension",
                "tension_update": tension_update,
            }

            update = {"stochastic_tension": behavior}
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

# =====
# TESTS
# =====

def get_test_regulation_spec(interval=0.1, config=None, double=False):
    spec = get_test_spec(interval=interval, config=config)
    spec["Regulation"] = {
        "_type": "process",
        "address": "local:TestRegulations",
        "config": {
            "period": 5,
            "geom": "VesselGeometry",
            "crit_area": 2,
            "growth_rate": 0.2,
            "double": False,
        },
        "inputs": {
            "global_time": ["global_time"],
            "datasets": ["Datasets"],
        },
        "outputs": {
            "behaviors": ["Behaviors"],
        },
        "interval": interval,
    }
    if double:
        spec["Regulation"]["config"]["double"] = True

    return spec

def get_test_stochastic_spec(interval=0.1, config = None, tau=1.0, sigma=1.0):
    if callable(config):
        spec = get_test_spec(interval=interval, config=config())
    else:
        spec = get_test_spec(interval=interval, config=config)
    spec["Stochastic"] = {
        "_type": "process",
        "address": "local:StochasticLineTension",
        "config": {
            "tau": tau,
            "sigma": sigma,
        },
        "inputs": {
            "datasets": ["Datasets"],
        },
        "outputs": {
            "behaviors": ["Behaviors"],
        },
        "interval": interval,
    }
    return spec

def get_test_jamming_spec(interval = 0.1, config=None, tau=1.0, sigma=1.0):
    if callable(config):
        spec = get_test_stochastic_spec(interval = interval, config=config(), tau=tau, sigma=sigma)
    else:
        spec = get_test_stochastic_spec(interval=interval, config=config, tau=tau, sigma=sigma)

    spec["Jamming"] = {
        "_type": "process",
        "address": "local:CellJamming",
        "config": {
            "trigger_time": 100,
            "rate": -0.05,
            "limits": [3.0, 4.2],
        },
        "inputs": {
            "global_time": ["global_time"],
            "datasets": ["Datasets"],
        },
        "outputs": {
            "behaviors": ["Behaviors"],
        },
        "interval": interval,
    }
    return spec

def run_test_regulation(core, double = False, tf=20, dt=0.1):
    if double:
        spec = get_test_regulation_spec(interval=dt, double=True)
    else:
        spec = get_test_regulation_spec(interval=dt)
    spec["emitter"] = emitter_from_wires({
        "global_time": ["global_time"],
        "face_df": ["Datasets", "face_df"],
        "edge_df": ["Datasets", "edge_df"],
        "vert_df": ["Datasets", "vert_df"],
        "behaviors": ["Behaviors"],
    })
    sim = Composite(
        {
            "state": spec,
        },
        core=core,
    )
    sim.run(tf)
    results = gather_emitter_results(sim)[("emitter",)]
    return results, sim

def run_test_stochastic(core, config = None, tf=20, dt=0.1, jamming=False):

    if jamming:
        spec = get_test_jamming_spec(interval=dt, config=config, tau=0.2, sigma=0.1)
    else:
        spec = get_test_stochastic_spec(interval=dt, config=config, tau=0.2, sigma=0.1)

    spec["emitter"] = emitter_from_wires({
        "global_time": ["global_time"],
        "face_df": ["Datasets", "face_df"],
        "edge_df": ["Datasets", "edge_df"],
        "vert_df": ["Datasets", "vert_df"],
        "behaviors": ["Behaviors"],
    })
    sim = Composite(
        {
            "state": spec,
        },
        core=core,
    )
    sim.run(tf)
    results = gather_emitter_results(sim)[("emitter",)]
    return results, sim


if __name__ == "__main__":
    from vivarium_tyssue.data_types import register_types
    import pandas as pd
    from bigraph_viz import plot_bigraph
    from vivarium_tyssue.processes import register_processes
    from matplotlib import pyplot as plt

    profiler = cProfile.Profile()
    profiler1 = cProfile.Profile()
    # create the core object
    core = allocate_core()
    # register processes
    core = register_types(core)
    core.register_link("EulerSolver", EulerSolver)
    core = register_processes(core)

    # profiler.enable()
    # results, sim = run_test_regulation(core, double=False)
    # profiler.disable()
    # profiler.dump_stats("regulations.prof")
    # history = sim.state["Tyssue"]["instance"].history
    # history.update_datasets()
    # draw_specs = config.draw.sheet_spec()
    # draw_specs["face"]["visible"] = True
    # draw_specs["face"]["visible"] = True
    # draw_specs["face"]["alpha"] = 1
    # draw_specs["face"]["color"] = "blue"
    # draw_specs["edge"]["color"] = "black"
    # create_gif(history, "test_division.gif", coords = ["x", "z"], **draw_specs)
    # df = pd.DataFrame(results[10]["face_df"]).set_index("face")

    for jamming in [True]:
        if jamming:
            file_name = "test_jamming_flat.gif"
        else:
            file_name = "test_stochastic_migration.gif"
        start = time.time()
        # profiler1.enable()
        results1, sim1 = run_test_stochastic(core, config=get_test_config_flat(), tf=200, dt = 0.01, jamming=jamming)
        # profiler1.disable()
        # profiler1.dump_stats("regulations1.prof")
        print(f"{time.time() - start} seconds")
        history = sim1.state["Tyssue"]["instance"].history
        history.update_datasets()
        draw_specs = config.draw.sheet_spec()
        cmap = plt.get_cmap("autumn")
        color_map = cmap([1 if i == 33 else 0.0 for i in range(206)])
        draw_specs["face"]["visible"] = True
        draw_specs["face"]["alpha"] = 0.5
        draw_specs["face"]["color"] = color_map
        draw_specs["edge"]["color"] = "black"
        draw_specs["edge"]["color"] = "black"
        create_gif(history, output=file_name, coords = ["x", "y"], **draw_specs, num_frames=200)
        # df1 = pd.DataFrame.from_records(results1[10]["face_df"], index="face")