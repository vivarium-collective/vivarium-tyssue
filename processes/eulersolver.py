import logging
import warnings

import numpy as np
from pprint import pprint

from process_bigraph import Process, Composite, ProcessTypes
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from vivarium_tyssue.maps import *

from tyssue import Sheet
from tyssue.behaviors.event_manager import EventManager
from tyssue.behaviors.sheet.basic_events import reconnect
from tyssue.core.history import History
from tyssue.io.hdf5 import load_datasets
from tyssue import config

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
        "geom": "string", #key indicating the desired geometry class in GEOMETRY_MAP
        "effectors": "list[string]", #list of strings representing effectors from the EFFECTORS_MAP
        "ref_effector": "string", #string, representing the effector from the EFFECTORS_MAP
        "factory": "string", #key indicating the factory class to generate from the FACTORY_MAP
        "auto_reconnect": "bool", # if True, will automatically perform reconnections
        "bounds": "tuple", # bounds the displacement of the vertices at each time step
        "emit_columns": "map[list[string]]" # dict containing lists of column names to emit for each dataframe
    }

    def initialize(self, config):
        self._set_pos = set_pos
        self.geom = GEOMETRY_MAP[config["geom"]]
        datasets = load_datasets(config["eptm"])
        self.eptm = Sheet("epithelium", datasets)
        self.eptm.network_changed = False
        effectors = [EFFECTORS_MAP[effector] for effector in config["effectors"]]
        self.model = FACTORY_MAP[config["factory"]](effectors, EFFECTORS_MAP[config["ref_effector"]])
        self.history = History(self.eptm)

        manager = EventManager()
        if self.config["auto_reconnect"]:
            if "reconnect" not in [n[0].__name__ for n in manager.next]:
                manager.append(reconnect)

        self.manager = manager
        self.bounds = self.config["bounds"]

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
            "behaviors": "maybe[map]"
        }

    def outputs(self):
        return {
            "vert_df": "tyssue_dset",
            "edge_df": "tyssue_dset",
            "face_df": "tyssue_dset",
            "cell_df": "tyssue_dset",
            "network_changed": "bool",
        }

    def update(self, inputs, interval):

        if len(inputs["behaviors"]) > 0:
            for behavior, args in inputs["behaviors"].items():
                func = BEHAVIOR_MAP[behavior]
                self.manager.append(func, args)
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

        emit_columns = self.config.get("emit_columns", {})
        # if emit_columns is empty, just dump all dataframes as dicts
        if not emit_columns:
            for df_name in ["vert_df", "edge_df", "face_df", "cell_df"]:
                if hasattr(self.eptm, df_name):
                    dicts[df_name] = getattr(self.eptm, df_name).reset_index().to_dict(orient="records")
                else:
                    dicts[df_name] = {}
        else:
            for df_name in ["vert_df", "edge_df", "face_df", "cell_df"]:
                if hasattr(self.eptm, df_name):
                    cols = emit_columns.get(df_name)
                    df = getattr(self.eptm, df_name)
                    if cols:
                        df = df[cols]
                    dicts[df_name] = df.reset_index().to_dict(orient="records")
                else:
                    dicts[df_name] = {}

        vert_df, edge_df, face_df, cell_df = (
            dicts.get("vert_df"),
            dicts.get("edge_df"),
            dicts.get("face_df"),
            dicts.get("cell_df"),
        )

        return {
            "vert_df": vert_df,
            "edge_df": edge_df,
            "face_df": face_df,
            "cell_df": cell_df,
            "network_changed": network_changed,
        }

def get_test_config():
    return {
        "name": "Test Cylinder",
        "eptm": "test_cylinder.hf5",
        "geom": "VesselGeometry",
        "effectors": ["LineTension", "FaceAreaElasticity", "PerimeterElasticity"],
        "ref_effector": "PerimeterElasticity",
        "factory": "model_factory_vessel",
        "auto_reconnect": True, # if True, will automatically perform reconnections
        "bounds": (-1000, 1000), # bounds the displacement of the vertices at each time step
        "emit_columns": {} # dict containing lists of column names to emit for each dataframe
    }

def get_test_spec(interval=0.1):
    return {
        "Tyssue": {
            "_type": "process",
            "address": "local:EulerSolver",
            "config": get_test_config(),
            "inputs": {
                "behaviors": "Behaviors"
            },
            "outputs": {
                "vert_df": "Vertex",
                "edge_df": "Edge",
                "face_df": "Face",
                "cell_df": "Cell",
                "network_changed": "Network Changed",
            },
        },
        "Vertex": [],
        "Edge": [],
        "Face": [],
        "Cell": [],
        "Network Changed": False,
        "Behaviors": {}
    }

def run_test_solver(core):
    spec = get_test_spec()
    spec["emitter"] = emitter_from_wires({
        "global_time": ["global_time"],
        "face_df": ["Face"],
        "edge_df": ["Edge"],
        "vert_df": ["Vert"],
    })
    sim = Composite(
        {
            "state": spec,
        },
        core=core,
    )
    results=sim.run(10)
    pprint(results[20])

if __name__ == "__main__":
    from vivarium_tyssue import register_types
    # create the core object
    core = ProcessTypes()
    # register data types
    core = register_types(core)
    core.register_process("EulerSolver", EulerSolver)

    run_test_solver(core)