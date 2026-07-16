import numpy as np

from tyssue.topology.sheet_topology import cell_division, remove_face
from vivarium_tyssue.core_maps import GEOMETRY_MAP

# Install the tyssue topology robustness shims (drop_two_sided_faces /
# split_vert guards) as soon as any behavior is imported — cell_division and
# remove_face below reach those buggy helpers, so they must be patched before
# the first division / extrusion. Idempotent; build_core() also calls this.
from vivarium_tyssue.behaviors.tyssue_patches import apply_tyssue_topology_patches
apply_tyssue_topology_patches()


def _uid_to_face_index(sheet):
    """``unique_id -> face_df index label`` map, cached on the sheet.

    The behaviors (and the Gillespie process) look a single cell up by its
    ``unique_id`` many times per step; done as ``face_df[face_df["unique_id"] ==
    uid]`` that is a full-frame boolean scan each call and dominated the crypt
    profile. Here we build the map once and reuse it until ``face_df`` is
    replaced. ``reset_index`` (fired by every division / extrusion / reconnect)
    reassigns ``sheet.face_df`` to a fresh object, so an identity check
    (``cache is sheet.face_df``) invalidates the cache exactly when the
    row<->uid mapping can change; in-place writes (flagging ``cell_type``,
    scaling ``prefered_area``) keep the same object and stay valid.
    """
    fd = sheet.face_df
    cache = getattr(sheet, "_uid_face_cache", None)
    if cache is not None and cache[0] is fd:
        return cache[1]
    mapping = {int(u): idx for u, idx in zip(fd["unique_id"].to_numpy(), fd.index.to_numpy())}
    sheet._uid_face_cache = (fd, mapping)
    return mapping


def face_index_for(sheet, uid):
    """Return the ``face_df`` index label for a cell's ``unique_id``, or ``None``
    if no live face carries it (extruded / not yet born). O(1) via the cached map
    in :func:`_uid_to_face_index` instead of an O(Nfaces) boolean scan."""
    return _uid_to_face_index(sheet).get(int(uid))


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
        cell_idx = int(face_index_for(sheet, cell_uid))
    if radius is None:
        radius = (sheet.vert_df["x"].max() - sheet.vert_df["x"].min())/2
    daughter = cell_division(sheet, cell_idx, geom)
    fix_points_cylinder(sheet, radius=radius)
    return daughter

# ---------------------------------------------------------------------------
# Batched grow/shrink of committed cells
#
# division() / apoptosis_extrusion() used to re-run one manager callback per
# committed cell every step just to scale a single ``prefered_area`` scalar —
# tens of thousands of pandas .loc calls dominating the crypt profile. Instead a
# commit records the cell's *own* growth parameters (which still arrive per-cell
# from whatever external process emitted the behavior) into transient per-face
# ``commit_*`` columns, and a single batched grower (queued once) does the whole
# grow/shrink in one vectorized pass, firing the real topology op only for the
# cells that crossed their own threshold. The rates stay a property of the
# emitted behavior — nothing moves onto the sheet settings or the solver config.
# ---------------------------------------------------------------------------

# commit_state: 0 = not committed, 1 = dividing, 2 = extruding.
_COMMIT_DEFAULTS = {"commit_state": 0.0, "commit_rate": 0.0, "commit_crit": 0.0, "commit_dt": 0.0}
_GROWER_NAME = "_grow_committed_cells"


def _ensure_commit_cols(sheet):
    fd = sheet.face_df
    for col, default in _COMMIT_DEFAULTS.items():
        if col not in fd.columns:
            fd[col] = default
    if "commit_type" not in fd.columns:
        # target cell_type to restore on a division's mother/daughter ("" = none)
        fd["commit_type"] = ""


def _ensure_grower(sheet, manager, geom):
    """Queue the single batched grower on the manager's ``next`` deque, unless one
    is already queued (the manager doesn't de-dup a behavior that carries no
    face_id, so we do it here)."""
    if any(tup[0].__name__ == _GROWER_NAME for tup in manager.next):
        return
    manager.append(_grow_committed_cells, geom=geom)


def _clear_commit(sheet, idx, restore_type=None):
    fd = sheet.face_df
    if restore_type:
        fd.loc[idx, "cell_type"] = restore_type
    fd.loc[idx, "commit_state"] = 0.0
    fd.loc[idx, "commit_rate"] = 0.0
    fd.loc[idx, "commit_crit"] = 0.0
    fd.loc[idx, "commit_dt"] = 0.0
    fd.loc[idx, "commit_type"] = ""


def _do_division(sheet, geometry, cell_uid):
    """Actual split of a division-ready cell (identical topology handling to the
    former in-callback path), then restore its target cell_type and clear the
    commit flags on both mother and daughter."""
    cell_id = face_index_for(sheet, cell_uid)
    if cell_id is None:
        return
    cell_id = int(cell_id)
    target_type = sheet.face_df.loc[cell_id, "commit_type"]
    sheet.face_df.loc[cell_id, "prefered_area"] = 1.0
    daughter = cell_division(sheet, cell_id, geometry)
    sheet.reset_index(order=True)
    geometry.update_all(sheet)
    sheet.network_changed = True
    restore = target_type if target_type else None
    _clear_commit(sheet, cell_id, restore_type=restore)
    _clear_commit(sheet, daughter, restore_type=restore)
    print(f"cell n°{daughter} is born")


def _do_extrusion(sheet, geometry, cell_uid):
    """Actual removal of an extrusion-ready cell (identical topology handling to
    the former in-callback path)."""
    cell_id = face_index_for(sheet, cell_uid)
    if cell_id is None:
        return
    cell_id = int(cell_id)
    sheet.face_df.loc[cell_id, "prefered_area"] = 1.0
    try:
        remove_face(sheet, cell_id)
    except Exception as exc:  # noqa: BLE001
        print(f"remove_face failed for cell {cell_uid} ({type(exc).__name__}); skipping")
        # Drop the commit so the grower doesn't retry this cell forever.
        cid = face_index_for(sheet, cell_uid)
        if cid is not None:
            _clear_commit(sheet, int(cid))
        return
    sheet.reset_index(order=True)
    _drop_isolated_verts(sheet)
    # Project vertices back onto the cylinder surface BEFORE updating geometry
    # (a removed face's centroid vertex can land off the r=prefered_radius
    # surface and make the next Euler step blow up to NaN).
    radius = sheet.settings.get("radius")
    if radius is None and "prefered_radius" in sheet.vert_df.columns:
        radius = float(sheet.vert_df["prefered_radius"].mean())
    if radius:
        fix_points_cylinder(sheet, radius=radius)
    geometry.update_all(sheet)
    sheet.network_changed = True


def _grow_committed_cells(sheet, manager, geom="SheetGeometry"):
    """One vectorized grow/shrink pass over every committed cell, then fire the
    real division / extrusion for any that crossed their own threshold. Re-queues
    itself while commitments remain."""
    geometry = GEOMETRY_MAP[geom] if isinstance(geom, str) else geom
    fd = sheet.face_df
    if "commit_state" not in fd.columns:
        return
    state = fd["commit_state"].to_numpy()
    div_mask = state == 1.0
    ext_mask = state == 2.0
    if not div_mask.any() and not ext_mask.any():
        return  # nothing pending -> let the grower drop out of the queue

    # Vectorized growth / shrink of every committed cell's prefered_area at its
    # own recorded rate. A_0(t+dt) = A_0(t) * (1 +/- dt*rate).
    prefered = fd["prefered_area"].to_numpy(dtype=float).copy()
    rate = fd["commit_rate"].to_numpy(dtype=float)
    dt = fd["commit_dt"].to_numpy(dtype=float)
    prefered[div_mask] *= (1.0 + dt[div_mask] * rate[div_mask])
    prefered[ext_mask] *= (1.0 - dt[ext_mask] * rate[ext_mask])
    fd["prefered_area"] = prefered

    # Threshold crossers -> real topology ops (resolved by unique_id, since each
    # op reindexes face_df). Division on area>crit; extrusion once the actual
    # area falls below crit OR the death target (prefered_area) has collapsed.
    area = fd["area"].to_numpy(dtype=float)
    crit = fd["commit_crit"].to_numpy(dtype=float)
    DEATH_FLOOR = 0.5
    uid = fd["unique_id"].to_numpy()
    div_ready = uid[div_mask & (area > crit)]
    ext_ready = uid[ext_mask & ((area < crit) | (prefered < DEATH_FLOOR))]

    for cell_uid in div_ready:
        _do_division(sheet, geometry, int(cell_uid))
    for cell_uid in ext_ready:
        _do_extrusion(sheet, geometry, int(cell_uid))

    # Re-queue while any commitment is still pending (recompute: the topology ops
    # above cleared / removed some rows).
    fd = sheet.face_df
    if "commit_state" in fd.columns and (fd["commit_state"].to_numpy() != 0.0).any():
        _ensure_grower(sheet, manager, geom)


def division(
        sheet, manager, geom= "SheetGeometry", cell_uid=0, cell_type=None, crit_area=2.0, growth_rate=0.1, dt=1.
):
    """Commit a cell to division.

    Records this cell's own growth parameters (from the emitting process) into
    its ``commit_*`` face columns, flags it ``"dividing"`` for colour-coding, and
    hands off to the batched grower (:func:`_grow_committed_cells`), which grows
    every committed cell's ``prefered_area`` and splits it once ``area`` exceeds
    ``crit_area``.

    Parameters
    ----------
    sheet: a :class:`Sheet` object
    cell_uid: int
        the unique_id of the dividing cell
    cell_type: str, optional
        If provided, the cell is flagged "dividing" while it grows and both the
        mother and daughter faces are stamped with this cell_type once division
        occurs. If None (default), no cell_type bookkeeping is done.
    crit_area: float
        the area at which the cell divides
    growth_rate: float
        increase in the prefered area per unit time
        A_0(t + dt) = A0(t) * (1 + growth_rate * dt)
    """
    # The cell may have been extruded between when the coupling queued this
    # division and when the manager runs it — its unique_id is then gone.
    cell_id = face_index_for(sheet, cell_uid)
    if cell_id is None:
        print("Cell not found, skipping division")
        return
    cell_id = int(cell_id)
    _ensure_commit_cols(sheet)
    fd = sheet.face_df
    if cell_type is not None:
        fd.loc[cell_id, "cell_type"] = "dividing"
        fd.loc[cell_id, "commit_type"] = cell_type
    fd.loc[cell_id, "commit_state"] = 1.0
    fd.loc[cell_id, "commit_rate"] = growth_rate
    fd.loc[cell_id, "commit_crit"] = crit_area
    fd.loc[cell_id, "commit_dt"] = dt
    _ensure_grower(sheet, manager, geom)

def _drop_isolated_verts(sheet):
    """Drop vertices that no edge references (isolated after a face removal).

    ``remove_face`` collapses a face's vertices into a single new centroid vert
    and drops the originals, but a boundary/degenerate removal can leave a vert
    with no incident edge. Such a vert keeps whatever stale (or NaN) position it
    had and pollutes ``np.isfinite`` checks / geometry, so prune them, then
    reindex so the mesh is contiguous again. No-op when nothing is isolated."""
    used = set(sheet.edge_df["srce"].to_numpy()) | set(sheet.edge_df["trgt"].to_numpy())
    isolated = sheet.vert_df.index.difference(list(used))
    if len(isolated):
        sheet.vert_df.drop(isolated, axis=0, inplace=True)
        sheet.reset_index(order=True)


#Apoptosis behaviors
def apoptosis_cell(sheet, geom, radius=None, cell_uid=None, cell_idx=None):
    """removes a cell from a cylindrical tyssue sheet"""
    if cell_uid is None:
        if cell_idx is None:
            raise ValueError("cell_uid or cell_idx must be specified")
    if cell_uid is not None:
        cell_idx = int(face_index_for(sheet, cell_uid))
    if radius is None:
        radius = (sheet.vert_df["x"].max() - sheet.vert_df["x"].min())/2
    vertex = remove_face(sheet, cell_idx)
    fix_points_cylinder(sheet, radius=radius)
    geom.update_all(sheet)

def apoptosis_extrusion(
        sheet, manager, geom= "SheetGeometry", cell_uid=0, crit_area=0.5, shrink_rate=0.1, dt=1.
):
    """Commit a cell to apoptotic extrusion.

    Records this cell's own shrink parameters into its ``commit_*`` columns, flags
    it ``"extruding"``, and hands off to the batched grower, which shrinks every
    committed cell's ``prefered_area`` and removes it once its actual ``area``
    falls below ``crit_area`` OR its death target ``prefered_area`` collapses
    below ``DEATH_FLOOR`` (0.5). The DEATH_FLOOR criterion matters in the crypt:
    extrusion is Wnt/z-biased to the free top rim, where cells stay mechanically
    stretched (large actual area), so without it a committed cell's actual area
    never reaches crit_area and it would never die.
    """
    cell_id = face_index_for(sheet, cell_uid)
    if cell_id is None:
        print("Cell not found, skipping event")
        return
    cell_id = int(cell_id)
    _ensure_commit_cols(sheet)
    fd = sheet.face_df
    fd.loc[cell_id, "cell_type"] = "extruding"
    fd.loc[cell_id, "commit_state"] = 2.0
    fd.loc[cell_id, "commit_rate"] = shrink_rate
    fd.loc[cell_id, "commit_crit"] = crit_area
    fd.loc[cell_id, "commit_dt"] = dt
    _ensure_grower(sheet, manager, geom)

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
    cell_id = face_index_for(sheet, cell_uid)
    if cell_id is not None:
        sheet.face_df.loc[cell_id, "cell_type"] = new_type
