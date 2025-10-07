import logging
import warnings

import numpy as np

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

class EulerCylinder(Process):
    """Generalized Euler solver for Tyssue-based epithelial simulations
    """
    config_schema = {
        "eptm": "string", #saved tyssue epithelium file
        "geom": "string", #key indicating the desired geometry class in GEOMETRY_MAP
        "effectors": "list[string]", #list of strings representing effectors from the EFFECTORS_MAP
        "ref_effector": "string", #string, representing the effector from the EFFECTORS_MAP
        "factory": "string", #key indicating the factory class to generate from the FACTORY_MAP
        "auto_reconnect": "bool", # if True, will automatically perform reconnections
        "bounds": "tuple", # bounds the displacement of the vertices at each time step
    }

    def initialize(self, config):
        self._set_pos = set_pos
        self.geom = GEOMETRY_MAP[config["geom"]]
        datasets = load_datasets(config["eptm"])
        self.eptm = Sheet("epithelium", datasets)
        self.eptm.network_changed = False
        effectors = [effector for effector in config["effectors"].values()]
        self.model = FACTORY_MAP[config["factory"]](effectors, config["ref_effector"])
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
        .. math::
        \frac{dr_i}{dt} = -\frac{\nabla U_i}{\eta_i}
        """

        grad_U = self.model.compute_gradient(self.eptm).loc[self.eptm.active_verts]
        return (
                -grad_U.values
                / self.eptm.vert_df.loc[self.eptm.active_verts, "viscosity"].values[:, None]
        ).ravel()

    def update(self, inputs, interval):
        pos = self.current_pos
        dot_r = self.ode_func()
        if self.bounds is not None:
            dot_r = np.clip(dot_r, *self.bounds)
        pos = pos + dot_r * interval
        self.eptm.set_pos(pos)

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