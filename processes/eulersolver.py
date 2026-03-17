import logging
import warnings
import inspect
import time

from pprint import pprint

from bigraph_schema import allocate_core
from bigraph_schema.schema import get_frame_schema
from process_bigraph import Process, Composite
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from vivarium_tyssue.maps import *
from vivarium_tyssue.core_maps import GEOMETRY_MAP

from tyssue.behaviors.event_manager import EventManager
from tyssue.behaviors.sheet.basic_events import reconnect
from tyssue.core.history import History
from tyssue.io.hdf5 import load_datasets
from tyssue.draw import create_gif
from tyssue import config

log = logging.getLogger(__name__)

maps = {
    "GEOMETRY_MAP": GEOMETRY_MAP,
    "FACTORY_MAP": FACTORY_MAP,
    "EFFECTORS_MAP": EFFECTORS_MAP,
    "TISSUE_MAP": TISSUE_MAP,
    "BEHAVIOR_MAP": BEHAVIOR_MAP,
}

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
        "parameters": "map[map]",
        "geom": "string", #key indicating the desired geometry class in GEOMETRY_MAP
        "effectors": "list[string]", #list of strings representing effectors from the EFFECTORS_MAP
        "ref_effector": "string", #string, representing the effector from the EFFECTORS_MAP
        "factory": "string", #key indicating the factory class to generate from the FACTORY_MAP
        "auto_reconnect": "boolean", # if True, will automatically perform reconnections
        "bounds": "map[float]", # bounds the displacement of the vertices at each time step
        "output_columns": "map[list[string]]", # dict containing lists of column names to emit for each dataframe
        "settings": "map",
        "maps": "map", #map of maps, if using default
    }

    def initialize(self, config):
        self.maps = maps
        if self.config["maps"]:
            self.maps.update(self.config["maps"])
        self._set_pos = set_pos
        self.geom = self.maps["GEOMETRY_MAP"][config["geom"]]
        datasets = load_datasets(config["eptm"])
        self.tyssue_type = self.maps["TISSUE_MAP"][config["tissue_type"]]
        self.eptm = self.tyssue_type("epithelium", datasets)
        self.eptm.network_changed = False
        self.eptm.settings.update(config["settings"])
        self.geom.update_all(self.eptm)
        effectors = [self.maps["EFFECTORS_MAP"][effector] for effector in config["effectors"]]
        self.model = self.maps["FACTORY_MAP"][config["factory"]](effectors, self.maps["EFFECTORS_MAP"][config["ref_effector"]])
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

    def output_dfs(self):
        output_columns = self.config.get("output_columns", {})
        # if output_columns is empty, just dump all dataframes as dicts
        output_dfs = {}
        if not output_columns:
            for df_name in ["vert_df", "edge_df", "face_df", "cell_df"]:
                if getattr(self.eptm, df_name) is not None:
                    output_dfs[df_name] = getattr(self.eptm, df_name)
                else:
                    output_dfs[df_name] = {}
        else:
            for df_name in ["vert_df", "edge_df", "face_df", "cell_df"]:
                if getattr(self.eptm, df_name):
                    print(df_name)
                    if df_name in output_columns.keys():
                        cols = output_columns.get(df_name)
                        if not "unique_id" in cols:
                            cols.append("unique_id")
                        df = getattr(self.eptm, df_name)
                        if cols:
                            df = df[cols]
                        output_dfs[df_name] = df
                    else:
                        output_dfs[df_name] = getattr(self.eptm, df_name)
                else:
                    output_dfs[df_name] = {}
        return output_dfs

    def initial_state(self):
        outputs = self.output_dfs()
        for df_name, df in outputs.items():
            if not isinstance(df, dict):
                outputs[df_name] = df.to_dict(orient="list")
        return {
            "datasets": outputs,
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
            "behaviors": "behaviors",
            "global_time": "float",
        }

    def outputs(self):
        datasets = {
            "_type": "tyssue_data",
            "vert_df": {
                "_columns": get_frame_schema(self.eptm.vert_df)
            },
            "edge_df": {
                "_columns": get_frame_schema(self.eptm.edge_df)
            },
            "face_df": {
                "_columns": get_frame_schema(self.eptm.face_df)
            },
        }
        if self.eptm.cell_df:
            datasets["cell_df"] = {"_columns": get_frame_schema(self.eptm.cell_df)}
        else:
            datasets["cell_df"] = {}
        return {
            "datasets": datasets,
            "network_changed": "boolean",
            "behaviors_update": "map",
        }

    def update(self, inputs, interval):
        print(inputs["global_time"])
        if len(inputs["behaviors"]) > 0:
            for behavior, kwargs in inputs["behaviors"].items():
                func = self.maps["BEHAVIOR_MAP"][kwargs["func"]]
                del kwargs["func"]
                arg_names = [name for name, param in inspect.signature(func).parameters.items()]
                if "geom" in arg_names:
                    kwargs["geom"] = self.maps["BEHAVIOR_MAP"][kwargs["geom"]]
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

        dfs = self.output_dfs()

        to_remove = [key for key in inputs["behaviors"]]

        return {
            "datasets": dfs,
            "network_changed": network_changed,
            "behaviors_update": {
                "_remove": to_remove,
            },
        }

if __name__ == "__main__":
    from vivarium_tyssue.data_types import register_types
    from vivarium_tyssue.processes import register_processes
    import pandas as pd
    from matplotlib import pyplot as plt
    # create the core object
    core = allocate_core()
    core.register_link("EulerSolver", EulerSolver)
    core = register_types(core)
    core = register_processes(core)
    # register data types



