import numpy as np
import time
from pprint import pprint
import cProfile
import pstats
import io

from bigraph_schema import allocate_core
from process_bigraph import Step, Process, Composite
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

def get_test_config():
    return {
        "name": "Test Cylinder",
        "eptm": "test_cylinder.hf5",
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {
                "area_elasticity": 1,
                "prefered_area": 1,
                "perimeter_elasticity": 1,
                "prefered_perimeter": 3.5,
            },
            "edge_df": {
                "line_tension": 0,
                "is_active": 1,
            },
            "vert_df": {
                "viscosity": 1,
                "vessel_elasticity": 1,
                "prefered_radius": 2.5,
                "is_alive": 1,
            }
        },
        "geom": "VesselGeometry",
        "effectors": ["LineTension", "FaceAreaElasticity", "PerimeterElasticity", "VesselSurfaceElasticity"],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory",
        "settings": {
            "threshold_length": 0.03
        },
        "auto_reconnect": True, # if True, will automatically perform reconnections
        "bounds": None, # bounds the displacement of the vertices at each time step
        "output_columns": {}, # dict containing lists of column names to emit for each dataframe
        "maps": {},
    }

def get_test_config_flat():
    return {
        "name": "Test Square",
        "eptm": "test_square.hf5",
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {
                "area_elasticity": 1,
                "prefered_area": 1,
                "perimeter_elasticity": 0.1,
                "prefered_perimeter": 3.6,
                "migration_strength": [0.1 if i == 33 else 0.0 for i in range(206)],
                "is_alive": 1,
                "mx": 1,
                "mz": 0,
                "my": 1,
            },
            "edge_df": {
                "line_tension": 0,
                "is_active": 1,
            },
            "vert_df": {
                "viscosity": 1,
                "is_alive": 1,
            }
        },
        "geom": "SheetGeometry",
        "effectors": [
            "LineTension",
            "FaceAreaElasticity",
            "PerimeterElasticity",
            # "ActiveMigration"
        ],
        "ref_effector": "FaceAreaElasticity",
        "factory": "model_factory_bound",
        "settings": {
            "threshold_length": 0.03
        },
        "auto_reconnect": True, # if True, will automatically perform reconnections
        "bounds": None, # bounds the displacement of the vertices at each time step
        "output_columns": {} # dict containing lists of column names to emit for each dataframe
    }

def get_test_spec(interval=0.1, config=None):
    return {
        "Tyssue": {
            "_type": "process",
            "address": "local:EulerSolver",
            "config": config,
            "inputs": {
                "behaviors": ["Behaviors"],
                "global_time": ["global_time"],
            },
            "outputs": {
                "datasets": ["Datasets"],
                "network_changed": ["Network Changed"],
                "behaviors_update": ["Behaviors"],
            },
            "interval": interval,
        },
        "Network Changed": False,
        "Behaviors": {}
    }

def run_test_solver(core, config=None, tf=20, df = 0.1):
    spec = get_test_spec(interval=df, config=config)
    spec["emitter"] = emitter_from_wires({
        "global_time": ["global_time"],
        "face_df": ["Datasets", "face_df"],
        "edge_df": ["Datasets", "edge_df"],
        "vert_df": ["Datasets", "vert_df"],
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

def run_test_regulation(core, double = False, tf=20, dt=0.1):
    if double:
        spec = get_test_regulation_spec(interval=dt, double=True)
    else:
        spec = get_test_regulation_spec(interval=dt)
    spec["emitter"] = test_emitter

    sim = Composite(
        {
            "state": spec,
        },
        core=core,
    )
    sim.run(tf)
    results = gather_emitter_results(sim)[("emitter",)]
    return results, sim

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

def run_test_stochastic(core, config = None, tf=20, dt=0.1, jamming=False):

    if jamming:
        spec = get_test_jamming_spec(interval=dt, config=config, tau=0.2, sigma=0.1)
    else:
        spec = get_test_stochastic_spec(interval=dt, config=config, tau=0.2, sigma=0.1)

    spec["emitter"] = test_emitter
    sim = Composite(
        {
            "state": spec,
        },
        core=core,
    )
    sim.run(tf)
    results = gather_emitter_results(sim)[("emitter",)]
    return results, sim

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

def get_test_gradient_spec(interval=0.1, config=None, tau=1.0, sigma=1.0):
    if callable(config):
        spec = get_test_stochastic_spec(interval=interval, config=config(), tau=tau, sigma=sigma)
    else:
        spec = get_test_stochastic_spec(interval=interval, config=config, tau=tau, sigma=sigma)
    spec["Gradient"] = {
        "_type": "step",
        "address": "local:ParameterGradient",
        "config": {
            "gradient_type": "linear",
            "axis": "x",
            "args": {
                "m": -0.1,
                "c": 4.6,
            },
            "model_parameters": {
                "prefered_perimeter": "face"
            }
        },
        "inputs": {
            "datasets": ["Datasets"],
        },
        "outputs": {
            "behaviors": ["Behaviors"],
        },
    }
    return spec

test_emitter = emitter_from_wires({
        "global_time": ["global_time"],
        "face_df": ["Datasets", "face_df"],
        "edge_df": ["Datasets", "edge_df"],
        "vert_df": ["Datasets", "vert_df"],
        "behaviors": ["Behaviors"],
})

def run_test_gradient(core, config = None, tf=20, dt=0.1):
    spec = get_test_gradient_spec(interval=dt, config=config, tau=0.2, sigma=0.1)
    spec["Tyssue"]["config"]["parameters"]["face_df"]["migration_strength"] = [0.1 if i == 96 else 0.0 for i in range(206)]
    spec["Tyssue"]["config"]["parameters"]["face_df"]["my"] = 0
    spec["emitter"] = test_emitter
    sim = Composite(
        {
            "state": spec,
        },
        core=core,
    )
    sim.run(tf)
    results = gather_emitter_results(sim)[("emitter",)]
    return results, sim

def get_test_anisotropic_spec(interval=0.1, config=None):
    if callable(config):
        spec = get_test_spec(interval=interval, config=config())
    else:
        spec = get_test_spec(interval=interval, config=config)

    spec["Anisotropic"] = {
        "_type": "step",
        "address": "local:AnisotropicTension",
        "config": {
            "axes": ["x", "y"],
            "tension_values": [0, 0.2]
        },
        "inputs": {
            "datasets": ["Datasets"],
        },
        "outputs": {
            "behaviors": ["Behaviors"],
        },
    }
    return spec

def run_test_anisotropic(core, config = None, tf=20, dt=0.1):
    spec = get_test_anisotropic_spec(interval=dt, config=config)
    spec["Tyssue"]["config"]["factory"] = "model_factory"
    spec["Tyssue"]["config"]["parameters"]["face_df"]["prefered_perimeter"] = 3.4
    spec["emitter"] = test_emitter
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

    import pandas as pd
    from matplotlib import pyplot as plt

    from bigraph_viz import plot_bigraph
    from vivarium_tyssue.data_types import register_types
    from vivarium_tyssue.processes import register_processes, EulerSolver

    from tyssue import config
    from tyssue.draw import create_gif

    # create the core object
    core = allocate_core()
    # register processes
    core = register_types(core)
    core.register_link("EulerSolver", EulerSolver)
    core.register_link("EulerSolver", EulerSolver)
    core = register_processes(core)

    # results, sim = run_test_solver(core, config=get_test_config())
    # pprint(results[10]["face_df"])
    # pprint(results[8]["face_df"])
    # history = sim.state["Tyssue"]["instance"].history
    # history.update_datasets()
    draw_specs = config.draw.sheet_spec()
    draw_specs["face"]["visible"] = True
    draw_specs["face"]["visible"] = True
    draw_specs["face"]["alpha"] = 1
    draw_specs["face"]["color"] = "blue"
    draw_specs["edge"]["color"] = "black"
    # create_gif(history, "test.gif", coords = ["x", "z"], **draw_specs)

    # start = time.time()
    # results, sim = run_test_solver(core, config=get_test_config_flat(), tf = 40, df = 0.1)
    # end = time.time()
    # print(f"{time.time() - start} seconds")
    # pprint(results[10]["face_df"])
    # pprint(results[8]["face_df"])
    # history = sim.state["Tyssue"]["instance"].history
    # history.update_datasets()
    # draw_specs = config.draw.sheet_spec()
    # cmap = plt.get_cmap("autumn")
    # color_map = cmap([0.2 if i == 33 else 0.0 for i in range(206)])
    # draw_specs["face"]["visible"] = True
    # draw_specs["face"]["alpha"] = 0.5
    # draw_specs["face"]["color"] = color_map
    # draw_specs["edge"]["color"] = "black"
    # create_gif(history, "test_flat.gif", coords=["x", "y"], num_frames = 200, **draw_specs)

    # results, sim = run_test_regulation(core, double=False)
    # history = sim.state["Tyssue"]["instance"].history
    # history.update_datasets()
    # draw_specs = config.draw.sheet_spec()
    # draw_specs["face"]["visible"] = True
    # draw_specs["face"]["alpha"] = 0.5
    # draw_specs["face"]["color"] = "blue"
    # draw_specs["edge"]["color"] = "black"
    # create_gif(history, "test_division.gif", coords = ["x", "z"], **draw_specs)
    # df = pd.DataFrame(results[10]["face_df"]).set_index("face")

    # for jamming in [True, False]:
    #     if jamming:
    #         file_name = "test_jamming_flat.gif"
    #     else:
    #         file_name = "test_stochastic_migration.gif"
    #     start = time.time()
    #     results1, sim1 = run_test_stochastic(
    #         core,
    #         config=get_test_config_flat(),
    #         tf=20,
    #         dt=0.1,
    #         jamming=jamming)
    #     history = sim1.state["Tyssue"]["instance"].history
    #     history.update_datasets()

    # draw_specs = config.draw.sheet_spec()
    # cmap = plt.get_cmap("autumn")
    # color_map = cmap([1 if i == 96 else 0.0 for i in range(206)])
    # draw_specs["face"]["visible"] = True
    # draw_specs["face"]["alpha"] = 0.5
    # draw_specs["face"]["color"] = color_map
    # draw_specs["edge"]["color"] = "black"
    # draw_specs["edge"]["color"] = "black"
    #     # create_gif(history, output=file_name, coords = ["x", "y"], **draw_specs, num_frames=200)
    #
    # results2, sim2 = run_test_gradient(core, config=get_test_config_flat(), tf=400, dt=0.01)
    # history = sim2.state["Tyssue"]["instance"].history
    # history.update_datasets()
    # create_gif(history, output="test_gradient.gif", coords = ["x", "y"], **draw_specs, num_frames=200)

    results3, sim3 = run_test_anisotropic(core, config=get_test_config_flat(), tf=200, dt=0.1)
    history = sim3.state["Tyssue"]["instance"].history
    history.update_datasets()
    create_gif(history, output="test_anisotropic.gif", coords = ["x", "y"], **draw_specs, num_frames=200)