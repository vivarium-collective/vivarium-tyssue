import numpy as np

from tyssue.topology.sheet_topology import cell_division, remove_face
from pbg_tyssue.core_maps import GEOMETRY_MAP

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

#Cell Divisions

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

def division(
        sheet, manager, geom= "SheetGeometry", cell_uid=0, crit_area=2.0, growth_rate=0.1, dt=1.
):
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
    if type(geom) == str:
        geometry = GEOMETRY_MAP[geom]
    else:
        geometry = geom
    cell_id = int(sheet.face_df[sheet.face_df["unique_id"] == cell_uid].index[0])
    if sheet.face_df.loc[cell_id, "area"] > crit_area:
        # restore prefered_area
        sheet.face_df.loc[cell_id, "prefered_area"] = 1.0
        # Do division
        daughter = cell_division(sheet, cell_id, geometry)
        # Update the topology
        sheet.reset_index(order=True)
        # update geometry
        geometry.update_all(sheet)
        sheet.network_changed = True
        print(f"cell n°{daughter} is born")
    else:
        #
        sheet.face_df.loc[cell_id, "prefered_area"] *= (1 + dt * growth_rate)
        manager.append(
            division,
            geom=geom,
            cell_uid=cell_uid,
            crit_area=crit_area,
            growth_rate=growth_rate,
            dt=dt
        )

def divide_crypt(
        sheet, manager, geom= "SheetGeometry", cell_uid=0, cell_type="None", crit_area=2.0, growth_rate=0.1, dt=1.
    ):
    if type(geom) == str:
        geometry = GEOMETRY_MAP[geom]
    else:
        geometry = geom
    cell_id = int(sheet.face_df[sheet.face_df["unique_id"] == cell_uid].index[0])
    sheet.face_df.loc[cell_id, "cell_type"] = "dividing"
    if sheet.face_df.loc[cell_id, "area"] > crit_area:
        # restore prefered_area
        sheet.face_df.loc[cell_id, "prefered_area"] = 1.0
        # Do division
        daughter = cell_division(sheet, cell_id, geometry)
        # Update the topology
        sheet.reset_index(order=True)
        # update geometry
        geometry.update_all(sheet)
        sheet.network_changed = True
        sheet.face_df.loc[cell_id, "cell_type"] = cell_type
        sheet.face_df.loc[daughter, "cell_type"] = cell_type
        print(f"cell n°{daughter} is born")
    else:
        #
        sheet.face_df.loc[cell_id, "prefered_area"] *= (1 + dt * growth_rate)
        manager.append(
            divide_crypt,
            geom=geom,
            cell_uid=cell_uid,
            cell_type=cell_type,
            crit_area=crit_area,
            growth_rate=growth_rate,
            dt=dt
        )

#Apoptosis behaviors
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

def apoptosis_extrusion(
        sheet, manager, geom= "SheetGeometry", cell_uid=0, crit_area=0.5, shrink_rate=0.1, dt=1.
):
    if type(geom) == str:
        geometry = GEOMETRY_MAP[geom]
    else:
        geometry = geom
    try:
        cell_id = int(sheet.face_df[sheet.face_df["unique_id"] == cell_uid].index[0])
    except:
        print("Cell not found, skipping event")
        return
    sheet.face_df.loc[cell_id, "cell_type"] = "extruding"
    if sheet.face_df.loc[cell_id, "area"] < crit_area:
        # Restore prefered_area
        sheet.face_df.loc[cell_id, "prefered_area"] = 1.0
        # Remove the cell division
        vertex = remove_face(sheet, cell_id)
        # Update the topology
        sheet.reset_index(order=True)
        # update geometry
        geometry.update_all(sheet)
        sheet.network_changed = True
    else:
        #
        sheet.face_df.loc[cell_id, "prefered_area"] *= (1 - dt * shrink_rate)
        manager.append(
            apoptosis_extrusion,
            geom=geom,
            cell_uid=cell_uid,
            crit_area=crit_area,
            shrink_rate=shrink_rate,
            dt=dt
        )

def update_tension(sheet, manager, tension_update=None):
    if sheet.edge_df["line_tension"].dtype == "int64":
        sheet.edge_df["line_tension"] = sheet.edge_df["line_tension"].astype(float)
    if tension_update:
        sheet.edge_df.loc[
            sheet.edge_df["unique_id"].isin(tension_update),
            "line_tension"
        ] = sheet.edge_df["unique_id"].map(tension_update)

def cell_jamming(sheet, manager, rate, limits, dt):
    if (sheet.face_df["prefered_perimeter"].mean()) > limits[0] or (sheet.face_df["prefered_perimeter"].mean() < limits[1]):
        sheet.face_df["prefered_perimeter"] *= (1 + rate * dt)
        manager.append(cell_jamming, rate=rate, limits=limits, dt=dt)
    else:
        print("Jamming Complete")

def apply_gradient(sheet, manager, parameter_updates=None):
    """
    Parameters:
    sheet: a :class:`Sheet` object
    manager: a :class:`Manager` object
    parameter_updates: a dictionary of parameters (keys) and dataframe name & updates (values)
    """
    if parameter_updates:
        for parameter, updates in parameter_updates.items():
            sheet.datasets[updates["dataframe"]].loc[
                sheet.datasets[updates["dataframe"]]["unique_id"].isin(updates["update"]),
                parameter
            ] = sheet.datasets[updates["dataframe"]]["unique_id"].map(
                updates["update"]
            )

def differentiation(sheet, manager, cell_uid, new_type):
    sheet.face_df.loc[sheet.face_df["unique_id"] == cell_uid, "cell_type"] = new_type
