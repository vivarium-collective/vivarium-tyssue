import numpy as np

from tyssue.topology.sheet_topology import cell_division, remove_face

def update_stem_cells(eptm):
    eptm.face_df['stem_cell'] = 0
    eptm.face_df['dying_cell'] = 0
    eptm.face_df.loc[(eptm.face_df["boundary"] != 1) & (eptm.face_df["z"] < 0), "stem_cell"] = 1
    eptm.face_df.loc[(eptm.face_df["z"] > 0), "dying_cell"] = 1

def fix_points_cylinder(sheet, radius):
    xy = sheet.vert_df[['x', 'y']].to_numpy()
    r = np.linalg.norm(xy, axis=1)
    r_safe = np.where(r == 0, 1e-12, r)
    xy_on_cylinder = (radius / r_safe)[:, None] * xy
    sheet.vert_df['x'] = xy_on_cylinder[:, 0]
    sheet.vert_df['y'] = xy_on_cylinder[:, 1]

def divide_cylinder(sheet, manager, geom, division_rate, dt, radius):
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
