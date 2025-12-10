import numpy as np

from tyssue.topology.sheet_topology import cell_division, remove_face
from vivarium_tyssue.maps import GEOMETRY_MAP

def update_stem_cells(eptm):
    """updates which cells in a cylinder model are classified as stem cells"""
    eptm.face_df['stem_cell'] = 0
    eptm.face_df['dying_cell'] = 0
    eptm.face_df.loc[(eptm.face_df["boundary"] != 1) & (eptm.face_df["z"] < 0), "stem_cell"] = 1
    eptm.face_df.loc[(eptm.face_df["z"] > 0), "dying_cell"] = 1

def fix_points_cylinder(sheet, radius):
    """fixes vertices on a cylinder surface"""
    xy = sheet.vert_df[['x', 'y']].to_numpy()
    r = np.linalg.norm(xy, axis=1)
    r_safe = np.where(r == 0, 1e-12, r)
    xy_on_cylinder = (radius / r_safe)[:, None] * xy
    sheet.vert_df['x'] = xy_on_cylinder[:, 0]
    sheet.vert_df['y'] = xy_on_cylinder[:, 1]

def divide_cylinder(sheet, manager, geom, division_rate, dt, radius):
    """basic function to apply division in a cylinder crypt_gillespie model"""
    update_stem_cells(sheet)
    stem_cells = sheet.face_df.loc[(sheet.face_df["stem_cell"] == 1) & (sheet.face_df["area"]>=0.7*sheet.face_df["area"].mean())].copy()
    n_stem = len(stem_cells)
    cell_ids = list(stem_cells["unique_id"])
    n_divisions = np.random.binomial(n=n_stem, p=division_rate * dt)
    dividing_cells = np.random.choice(cell_ids, size=n_divisions, replace=False)
    for cell_id in dividing_cells:
        cell_idx = int(sheet.face_df[sheet.face_df["unique_id"] == cell_id].index[0])
        daughter = cell_division(sheet, cell_idx, geom)
    manager.append(divide_cylinder, geom=geom, division_rate=division_rate, dt=dt, radius=radius)
    fix_points_cylinder(sheet, radius=radius)

def apoptosis_cylinder(sheet, manager, death_rate, dt, radius, geom):
    """basic function to apply cell death in a cylinder model"""
    update_stem_cells(sheet)
    dying_cells = sheet.face_df.loc[sheet.face_df["dying_cell"] == 1]
    n_dying = len(dying_cells)
    cell_ids = list(dying_cells["unique_id"])
    n_deaths = np.random.binomial(n=n_dying, p=death_rate * dt)
    to_kill = np.random.choice(cell_ids, size=n_deaths, replace=False)

    for cell_id in to_kill:
        cell_idx = int(sheet.face_df[sheet.face_df["unique_id"] == cell_id].index[0])
        # if sheet.face_df.loc[cell_idx, "boundary"] == 0:
        vertex = remove_face(sheet, cell_idx)
        # else:
        #     vertex = drop_face(sheet, cell_idx, geom)
        # split_vert(sheet, vertex)
    manager.append(apoptosis_cylinder, death_rate=death_rate, dt=dt, radius=radius, geom=geom)
    fix_points_cylinder(sheet, radius=radius)

def divide_cell(sheet, geom, radius=None, cell_uid=None, cell_idx=None):
    """divides a cell within a tyssue sheet indexed by the cell idx or its unique id
    Parameters:
        sheet: tyssue sheet object, the cylindrical sheet object to perform division on
        geom: a tyssue geometry class, the geometry being used
        radius: float, radius of the cylinder
        cell_uid: integer, unique cell id of the cell (must be provided if cell_idx is None)
        cell_idx: integer, cell index of the cell in sheet.cell_df (must be provided if cell_uid is None)
    """
    if cell_uid is None:
        if cell_idx is None:
            raise ValueError("cell_uid or cell_idx must be specified")
    if cell_uid is not None:
        cell_idx = int(sheet.face_df[sheet.face_df["unique_id"] == cell_uid].index[0])
    if radius is None:
        radius = (sheet.vert_df["x"].max() - sheet.vert_df["x"].min())/2
    daughter = cell_division(sheet, cell_idx, geom)
    fix_points_cylinder(sheet, radius=radius)
    return daughter

def apoptosis_cell(sheet, geom, radius=None, cell_uid=None, cell_idx=None):
    """removes a cell from a cylindrical tyssue sheet"""
    if cell_uid is None:
        if cell_idx is None:
            raise ValueError("cell_uid or cell_idx must be specified")
    if cell_uid is not None:
        cell_idx = int(sheet.face_df[sheet.face_df["unique_id"] == cell_uid].index[0])
    if radius is None:
        radius = (sheet.vert_df["x"].max() - sheet.vert_df["x"].min())/2
    vertex = remove_face(sheet, cell_idx)
    fix_points_cylinder(sheet, radius=radius)
    geom.update_all(sheet)

def division(sheet, manager, geom="SheetGeometry", cell_id=0, crit_area=2.0, growth_rate=0.1, dt=1.):
    """Defines a division behavior.

    Parameters
    ----------

    sheet: a :class:`Sheet` object
    cell_id: int
        the index of the dividing cell
    crit_area: float
        the area at which
    growth_rate: float
        increase in the prefered are per unit time
        A_0(t + dt) = A0(t) * (1 + growth_rate * dt)
    """
    geom = GEOMETRY_MAP[geom]
    if sheet.face_df.loc[cell_id, "area"] > crit_area:
        # restore prefered_area
        sheet.face_df.loc[cell_id, "prefered_area"] = 1.0
        # Do division
        daughter = cell_division(sheet, cell_id, geom)
        # Update the topology
        sheet.reset_index(order=True)
        # update geometry
        geom.update_all(sheet)
        print(f"cell n°{daughter} is born")
    else:
        #
        sheet.face_df.loc[cell_id, "prefered_area"] *= (1 + dt * growth_rate)
        manager.append(division, cell_id=cell_id, crit_area=crit_area, growth_rate=growth_rate, dt=dt)