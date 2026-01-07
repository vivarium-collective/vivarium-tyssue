import logging
import warnings
import inspect
import time

from pprint import pprint

from bigraph_schema import allocate_core
from process_bigraph import Process, Composite
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from vivarium_tyssue.maps import *

from tyssue.behaviors.event_manager import EventManager
from tyssue.behaviors.sheet.basic_events import reconnect
from tyssue.core.history import History
from tyssue.io.hdf5 import load_datasets

log = logging.getLogger(__name__)

def set_pos(eptm, geom, pos):
    """Updates the vertex position of the :class:`Epithelium` object.

    Assumes that pos is passed as a 1D array to be reshaped as (eptm.Nv, eptm.dim)

    """
    log.debug("set pos")
    eptm.vert_df.loc[eptm.active_verts, eptm.coords] = pos.reshape((-1, eptm.dim))
    geom.update_all(eptm)

class EulerSolver(Process):
    """Generalized Euler solver for Tyssue-based epithelial simulations
    """
    config_schema = {
        "name": "string", #name for epithelium object
        "eptm": "string", #saved tyssue epithelium file
        "tissue_type": "string", #key indicating the desired tissue type from TISSUE_MAP
        "parameters": "map[map[float]]",
        "geom": "string", #key indicating the desired geometry class in GEOMETRY_MAP
        "effectors": "list[string]", #list of strings representing effectors from the EFFECTORS_MAP
        "ref_effector": "string", #string, representing the effector from the EFFECTORS_MAP
        "factory": "string", #key indicating the factory class to generate from the FACTORY_MAP
        "auto_reconnect": "boolean", # if True, will automatically perform reconnections
        "bounds": "map[float]", # bounds the displacement of the vertices at each time step
        "output_columns": "map[list[string]]", # dict containing lists of column names to emit for each dataframe
        "settings": "map",
    }

    def initialize(self, config):
        self._set_pos = set_pos
        self.geom = GEOMETRY_MAP[config["geom"]]
        datasets = load_datasets(config["eptm"])
        self.tyssue_type = TISSUE_MAP[config["tissue_type"]]
        self.eptm = self.tyssue_type("epithelium", datasets)
        self.eptm.network_changed = False
        self.eptm.settings.update(config["settings"])
        self.geom.update_all(self.eptm)
        effectors = [EFFECTORS_MAP[effector] for effector in config["effectors"]]
        self.model = FACTORY_MAP[config["factory"]](effectors, EFFECTORS_MAP[config["ref_effector"]])
        self.history = History(self.eptm)
        if len(config["parameters"]) > 0:
            for dataframe, parameters in config["parameters"].items():
                df = getattr(self.eptm, dataframe)
                for parameter, value in parameters.items():
                    df[parameter] = value

        manager = EventManager()
        if self.config["auto_reconnect"]:
            if "reconnect" not in [n[0].__name__ for n in manager.next]:
                manager.append(reconnect)

        self.manager = manager
        if len(self.config["bounds"]) > 0:
            self.bounds = self.config["bounds"]
        else:
            self.bounds = None

    def initial_state(self):
        dicts = {}
        output_columns = self.config.get("output_columns", {})
        # if output_columns is empty, just dump all dataframes as dicts
        if not output_columns:
            for df_name in ["vert_df", "edge_df", "face_df", "cell_df"]:
                if getattr(self.eptm, df_name) is not None:
                    dicts[df_name] = getattr(self.eptm, df_name).reset_index().to_dict(orient="records")
                else:
                    dicts[df_name] = {}
        else:
            for df_name in ["vert_df", "edge_df", "face_df", "cell_df"]:
                if hasattr(self.eptm, df_name):
                    if df_name in output_columns.keys():
                        cols = output_columns.get(df_name)
                        if not "unique_id" in cols:
                            cols.append("unique_id")
                        df = getattr(self.eptm, df_name)
                        if cols:
                            df = df[cols]
                        dicts[df_name] = df.reset_index().to_dict(orient="records")
                    else:
                        dicts[df_name] = getattr(self.eptm, df_name).reset_index().to_dict(orient="records")
                else:
                    dicts[df_name] = {}
        vert_df, edge_df, face_df, cell_df = (
            dicts.get("vert_df"),
            dicts.get("edge_df"),
            dicts.get("face_df"),
            dicts.get("cell_df"),
        )
        return {
            "datasets": {
                "Vert": vert_df,
                "Edge": edge_df,
                "Face": face_df,
                "Cell": cell_df,
            }
        }

    @property
    def current_pos(self):
        return self.eptm.vert_df.loc[
            self.eptm.active_verts, self.eptm.coords
        ].values.ravel()

    def set_pos(self, pos):
        """Updates the eptm vertices position"""
        return self._set_pos(self.eptm, self.geom, pos)

    def record(self, t):
        self.history.record(time_stamp=t)

    def ode_func(self):
        """Computes the models' gradient.
        Returns
        -------
        dot_r : 1D np.ndarray of shape (self.eptm.Nv * self.eptm.dim, )
        """

        grad_U = self.model.compute_gradient(self.eptm).loc[self.eptm.active_verts]
        return (
                -grad_U.values
                / self.eptm.vert_df.loc[self.eptm.active_verts, "viscosity"].values[:, None]
        ).ravel()

    def inputs(self):
        return {
            "behaviors": "map[behaviors]",
            "global_time": "float",
        }

    def outputs(self):
        return {
            "datasets": "map[tyssue_dset]",
            "network_changed": "boolean",
            "behaviors_update": "map",
        }

    def update(self, inputs, interval):
        print(inputs["global_time"])
        if len(inputs["behaviors"]) > 0:
            for behavior, kwargs in inputs["behaviors"].items():
                func = BEHAVIOR_MAP[kwargs["func"]]
                del kwargs["func"]
                arg_names = [name for name, param in inspect.signature(func).parameters.items()]
                if "manager" in arg_names:
                    # kwargs["manager"] = self.manager
                    self.manager.append(func, **kwargs)
                else:
                    self.manager.append(func, **kwargs)

        pos = self.current_pos
        dot_r = self.ode_func()
        if self.bounds is not None:
            dot_r = np.clip(dot_r, *self.bounds)
        pos = pos + dot_r * interval
        self.set_pos(pos)

        if self.manager is not None:
            self.manager.execute(self.eptm)
            self.geom.update_all(self.eptm)
            self.manager.update()

        if self.eptm.network_changed:
            network_changed = True
        else:
            network_changed = False

        self.eptm.network_changed = False

        self.record(inputs["global_time"])

        dicts = {}

        output_columns = self.config.get("output_columns", {})
        # if output_columns is empty, just dump all dataframes as dicts
        if not output_columns:
            for df_name in ["vert_df", "edge_df", "face_df", "cell_df"]:
                if getattr(self.eptm, df_name) is not None:
                    dicts[df_name] = getattr(self.eptm, df_name).reset_index().to_dict(orient="records")
                else:
                    dicts[df_name] = {}
        else:
            for df_name in ["vert_df", "edge_df", "face_df", "cell_df"]:
                if hasattr(self.eptm, df_name):
                    if df_name in output_columns.keys():
                        cols = output_columns.get(df_name)
                        if not "unique_id" in cols:
                            cols.append("unique_id")
                        df = getattr(self.eptm, df_name)
                        if cols:
                            df = df[cols]
                        dicts[df_name] = df.reset_index().to_dict(orient="records")
                    else:
                        dicts[df_name] = getattr(self.eptm, df_name).reset_index().to_dict(orient="records")
                else:
                    dicts[df_name] = {}

        vert_df, edge_df, face_df, cell_df = (
            dicts.get("vert_df"),
            dicts.get("edge_df"),
            dicts.get("face_df"),
            dicts.get("cell_df"),
        )

        to_remove = [key for key in inputs["behaviors"]]

        return {
            "datasets": {
                "Vert": vert_df,
                "Edge": edge_df,
                "Face": face_df,
                "Cell": cell_df,
            },
            "network_changed": network_changed,
            "behaviors_update": {
                "_remove": to_remove,
            },
        }

def get_test_config():
    return {
        "name": "Test Cylinder",
        "eptm": "test_cylinder.hf5",
        "tissue_type": "Sheet",
        "parameters": {
            "face_df": {
                "area_elasticity": 1,
                "prefered_area": 1,
                "perimeter_elasticity": 0.1,
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
        "auto_reconnect": True, # if True, will automatically perform reconnections
        "bounds": None, # bounds the displacement of the vertices at each time step
        "output_columns": {} # dict containing lists of column names to emit for each dataframe
    }

def get_test_spec(interval=0.1):
    return {
        "Tyssue": {
            "_type": "process",
            "address": "local:EulerSolver",
            "config": get_test_config(),
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

def run_test_solver(core):
    spec = get_test_spec()
    spec["emitter"] = emitter_from_wires({
        "global_time": ["global_time"],
        "face_df": ["Datasets", "Face"],
        "edge_df": ["Datasets", "Edge"],
        "vert_df": ["Datasets", "Vert"],
    })
    sim = Composite(
        {
            "state": spec,
        },
        core=core,
    )
    sim.run(5)
    results = gather_emitter_results(sim)[("emitter",)]
    return results, sim

if __name__ == "__main__":
    from vivarium_tyssue.data_types import register_types
    import pandas as pd
    # create the core object
    core = allocate_core()
    core.register_link("EulerSolver", EulerSolver)
    core = register_types(core)
    # register data types

    start= time.time()
    results, sim = run_test_solver(core)
    print(f"{time.time() - start} seconds")
    df = pd.DataFrame.from_records(results[10]["face_df"], index="face")
    pprint(df)
    pprint(results[10]["face_df"])

    # results1, sim1 = run_test_regulation(core, double=False)
    # history = sim1.state["Tyssue"]["instance"].history
    # history.update_datasets()
    # draw_specs = config.draw.sheet_spec()
    # draw_specs["face"]["visible"] = True
    # draw_specs["face"]["visible"] = True
    # draw_specs["face"]["alpha"] = 1
    # draw_specs["face"]["color"] = "blue"
    # draw_specs["edge"]["color"] = "black"
    # create_gif(history, "test.gif", coords = ["x", "z"], **draw_specs)
    # df = pd.DataFrame.from_records(results1[10]["face_df"], index="face")

