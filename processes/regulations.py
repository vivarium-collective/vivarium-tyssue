import random
import time
import numpy as np
from pprint import pprint
import cProfile
import pstats
import io

from process_bigraph import Process, Composite, ProcessTypes
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from vivarium_tyssue.maps import *
import vivarium_tyssue.models.crypt_gillespie.crypt_params as crypt_params
from vivarium_tyssue.processes.eulersolver import EulerSolver, get_test_spec, run_test_solver

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
            "datasets": "map[tyssue_dset]",
        }

    def outputs(self):
        return {
            "behaviors": "map"
        }

    def update(self, inputs, interval):
        print(inputs["global_time"])
        t = inputs["global_time"]
        on_period = (math.isclose(t % self.period, 0) or math.isclose(t % self.period, self.period)) and not math.isclose(t, 0)
        if on_period:
            faces = inputs["datasets"]["Face"]
            positive_y = [row for row in faces if row["y"] > 1]

            def pick():
                return random.choice(positive_y)["face"]

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
            "dataset": "tyssue_dset", #path to edge dataframe
        }

    def outputs(self):
        return {
            "behaviors": "map"
        }

    def update(self, inputs, interval):
        tension = np.array([edge["line_tension"] for edge in inputs["dataset"]])
        unique_ids = [edge["unique_id"] for edge in inputs["dataset"]]
        decay = np.exp(-interval / self.tau)
        noise_scale = self.sigma * np.sqrt(1 - np.exp(-2 * interval / self.tau))
        new_tension = list(decay * tension + noise_scale * np.random.randn(len(tension)))
        tension_update = {unique_id:tension_v for unique_id, tension_v in zip(unique_ids, new_tension)}

        behavior = {
            "func": "stochastic_tension",
            "tension_update": tension_update,
        }

        return {"behaviors": {"stochastic_tension": behavior}}

# =====
# TESTS
# =====

def get_test_regulation_spec(interval=0.1, double=False):
    spec = get_test_spec(interval=interval)
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

def get_test_stochastic_spec(interval=0.1, tau=1.0, sigma=1.0):
    spec = get_test_spec(interval=interval)
    spec["Stochastic"] = {
        "_type": "process",
        "address": "local:StochasticLineTension",
        "config": {
            "tau": tau,
            "sigma": sigma,
        },
        "inputs": {
            "dataset": ["Datasets", "Edge"],
        },
        "outputs": {
            "behaviors": ["Behaviors"],
        },
        "interval": interval,
    }
    return spec

def run_test_regulation(core, double = False):
    if double:
        spec = get_test_regulation_spec(interval=0.1, double=True)
    else:
        spec = get_test_regulation_spec(interval=0.1)
    spec["emitter"] = emitter_from_wires({
        "global_time": ["global_time"],
        "face_df": ["Datasets", "Face"],
        "edge_df": ["Datasets", "Edge"],
        "vert_df": ["Datasets", "Vert"],
        "behaviors": ["Behaviors"],
    })
    sim = Composite(
        {
            "state": spec,
        },
        core=core,
    )
    sim.run(20)
    results = gather_emitter_results(sim)[("emitter",)]
    return results, sim

def run_test_stochastic(core):
    spec = get_test_stochastic_spec(interval=0.1, tau=1.0, sigma=0.1)
    spec["emitter"] = emitter_from_wires({
        "global_time": ["global_time"],
        "face_df": ["Datasets", "Face"],
        "edge_df": ["Datasets", "Edge"],
        "vert_df": ["Datasets", "Vert"],
        "behaviors": ["Behaviors"],
    })
    sim = Composite(
        {
            "state": spec,
        },
        core=core,
    )
    sim.run(20)
    results = gather_emitter_results(sim)[("emitter",)]
    return results, sim

if __name__ == "__main__":
    from vivarium_tyssue import register_types
    import pandas as pd

    profiler = cProfile.Profile()
    # create the core object
    core = ProcessTypes()
    # register data types
    core = register_types(core)
    core.register_process("EulerSolver", EulerSolver)
    core.register_process("TestRegulations", TestRegulations)
    core.register_process("StochasticLineTension", StochasticLineTension)

    # results, sim = run_test_regulation(core, double=False)
    # history = sim.state["Tyssue"]["instance"].history
    # history.update_datasets()
    # draw_specs = config.draw.sheet_spec()
    # draw_specs["face"]["visible"] = True
    # draw_specs["face"]["visible"] = True
    # draw_specs["face"]["alpha"] = 1
    # draw_specs["face"]["color"] = "blue"
    # draw_specs["edge"]["color"] = "black"
    # create_gif(history, "test.gif", coords = ["x", "z"], **draw_specs)
    # df = pd.DataFrame.from_records(results[10]["face_df"], index="face")
    start = time.time()
    profiler.enable()
    results1, sim1 = run_test_stochastic(core)
    profiler.disable()
    print(f"{time.time() - start} seconds")
    history = sim1.state = sim1.state["Tyssue"]["instance"].history
    history.update_datasets()
    draw_specs = config.draw.sheet_spec()
    draw_specs["face"]["visible"] = True
    draw_specs["face"]["visible"] = True
    draw_specs["face"]["alpha"] = 1
    draw_specs["face"]["color"] = "blue"
    draw_specs["edge"]["color"] = "black"
    create_gif(history, "test_stochastic.gif", coords = ["x", "z"], **draw_specs)
    df = pd.DataFrame.from_records(results1[10]["face_df"], index="face")

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(40)
    print(stream.getvalue())