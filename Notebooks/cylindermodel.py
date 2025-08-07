import random
from pprint import pprint
import io
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import logging

import tyssue
from tyssue import History
from tyssue.dynamics import effectors, model_factory, model_factory_vessel, model_factory_cylinder
from tyssue.generation.hexagonal_grids import hexa_cylinder
from tyssue.generation.shapes import sheet_from_cell_centers
from tyssue.geometry.sheet_geometry import CylinderGeometry as geom, CylinderGeometryInit
from tyssue import config
from tyssue.draw import sheet_view, create_gif
from tyssue.geometry.vessel_geometry import VesselGeometry
from tyssue.solvers.viscous import EulerSolver
from tyssue.draw.ipv_draw import browse_history
from tyssue.topology.sheet_topology import cell_division, drop_face, split_vert
from tyssue.topology.base_topology import remove_face
from tyssue.behaviors import EventManager
import ipyvolume as ipv

from process_bigraph import Process, Composite, ProcessTypes
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

def divide(sheet, manager, geom, division_rate, dt):
    update_stem_cells(sheet, height=None)
    stem_cells = sheet.face_df.loc[sheet.face_df["stem_cell"] == 1]
    n_stem = len(stem_cells)
    cell_ids = list(stem_cells["unique_id"])
    n_divisions = np.random.binomial(n=n_stem, p=division_rate * dt)
    dividing_cells = np.random.choice(cell_ids, size=n_divisions, replace=False)

    for cell_id in dividing_cells:
        cell_idx = int(sheet.face_df[sheet.face_df["unique_id"] == cell_id].index[0])
        daughter = cell_division(sheet, cell_idx, geom)
    manager.append(divide, geom=geom, division_rate=division_rate, dt=dt)

def apoptosis(sheet, manager, death_rate, dt):
    update_stem_cells(sheet, height=None)
    dying_cells = sheet.face_df.loc[sheet.face_df["dying_cell"] == 1]
    n_dying = len(dying_cells)
    cell_ids = list(dying_cells["unique_id"])
    n_deaths = np.random.binomial(n=n_dying, p=death_rate * dt)
    to_kill = np.random.choice(cell_ids, size=n_deaths, replace=False)

    for cell_id in to_kill:
        cell_idx = int(sheet.face_df[sheet.face_df["unique_id"] == cell_id].index[0])
        vertex = remove_face(sheet, cell_idx)
        # split_vert(sheet, vertex)
    manager.append(apoptosis, death_rate=death_rate, dt=dt)

def update_stem_cells(eptm, height=None):
    if height is None:
        height = (eptm.face_df["z"].max() - eptm.face_df["z"].min())/3
    bot_threshold = eptm.face_df["z"].min() + height
    top_threshold = eptm.face_df["z"].max() - height
    eptm.face_df['stem_cell'] = (eptm.face_df['z'] <= bot_threshold).astype(int)
    eptm.face_df['dying_cell'] = (eptm.face_df['z'] >= top_threshold).astype(int)

#generate cylindrical tissue
points_xyz = hexa_cylinder(16,30, radius = 2.576, noise = 0.0, capped = True)
sheet = sheet_from_cell_centers(points_xyz)
sheet = sheet.extract_bounding_box(z_boundary = (-10.1, 10.1), coords=['x', 'y', 'z'])
sheet.sanitize(trim_borders=False)
CylinderGeometryInit.update_all(sheet)
sheet.network_changed = False

#generate model
model = model_factory_cylinder([
    effectors.LineTension,
    effectors.FaceAreaElasticity,
    # effectors.PerimeterElasticity,
    effectors.LumenVolumeElasticity,
    effectors.SurfaceElasticity
    # effectors.ChiralTorque
], effectors.FaceAreaElasticity)

#set model parameters
sheet.face_df["area_elasticity"] = 1
sheet.face_df["prefered_area"] = 1
sheet.face_df["perimeter_elasticity"] = 0.1
sheet.face_df["prefered_perimeter"] = 3.5
sheet.edge_df["line_tension"] = np.random.uniform(low=-0.1, high=0.1, size=sheet.edge_df.shape[0])
sheet.vert_df["viscosity"] = 1
sheet.vert_df["prefered_deviation"] = 0
sheet.vert_df["surface_elasticity"] = 5
sheet.settings["lumen_vol_elasticity"] = 0.1
sheet.settings["lumen_prefered_vol"] = sheet.settings["lumen_vol"]
sheet.settings["vol_cell"] = sheet.settings["lumen_vol"]/len(sheet.face_df)
sheet.settings["threshold_length"] = 0.1
geom.update_all(sheet)

#initialize history object
history = History(sheet)

#initialize manager and add division and apoptosis
manager = EventManager()
manager.append(divide, geom=geom, division_rate=0.01, dt=0.05)
manager.append(apoptosis, death_rate=0.01, dt=0.05)

#initialize solver
solver = EulerSolver(
    sheet,
    geom,
    model,
    history=history,
    auto_reconnect=True,
    manager=manager,
)

#set time variables
dt = 0.05
tf = 5
sheet.settings["dt"] = dt
sheet.settings["p_4"] = 1/dt
sheet.settings["p_5p"] = 1/dt
#run simulation
res = solver.solve(tf=tf, dt = dt)

#create visualization specs
draw_specs = tyssue.config.draw.sheet_spec()
draw_specs["face"]["visible"] = True
draw_specs["face"]["alpha"] = 0.2

fig, ax = sheet_view(sheet, coords=["x", "z"], **draw_specs)
plt.show()

