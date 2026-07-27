import numpy as np

from tyssue.topology.sheet_topology import cell_division, remove_face
from tyssue.topology.monolayer_topology import cell_division as monolayer_cell_division
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
    """Relabel a cell's ``cell_type``. Dimension-aware: a 3D monolayer / bulk
    epithelium carries its cells in ``cell_df``, a 2D sheet in ``face_df``."""
    if _is_3d(sheet):
        cell_id = cell_index_for(sheet, cell_uid)
        if cell_id is not None:
            sheet.cell_df.loc[cell_id, "cell_type"] = new_type
        return
    cell_id = face_index_for(sheet, cell_uid)
    if cell_id is not None:
        sheet.face_df.loc[cell_id, "cell_type"] = new_type


# ---------------------------------------------------------------------------
# 3D (monolayer / bulk) cell division & necrosis
#
# The 2D behaviors above track cells as faces (face_df) and grow prefered_area
# toward a critical AREA. A 3D monolayer / bulk epithelium tracks cells as
# volumetric cells (cell_df): the analogous behavior grows prefered_vol (the
# reference volume of the CellVolumeElasticity effector) toward a critical
# VOLUME and, when reached, splits the cell with tyssue's monolayer
# cell_division — with a per-division orientation so growth is genuinely 3D.
#
# These mirror the 2D commit-column / batched-grower design (one vectorized
# grow pass over all committed cells per step, then fire the topology op only
# for threshold crossers), but on cell_df / vol, and are kept separate so the
# battle-tested 2D crypt/tumor path is untouched.
# ---------------------------------------------------------------------------

def _is_3d(eptm):
    """True for a 3D monolayer / bulk epithelium (cells live in a non-empty
    ``cell_df``); False for a flat 2D Sheet (cells are faces)."""
    cell_df = getattr(eptm, "cell_df", None)
    return cell_df is not None and len(cell_df) > 0


def _uid_to_cell_index(eptm):
    """``unique_id -> cell_df index`` map, cached on the epithelium and
    invalidated whenever ``cell_df`` is replaced (every division reassigns it via
    ``pd.concat``). Mirrors :func:`_uid_to_face_index` for the 3D path."""
    cd = eptm.cell_df
    cache = getattr(eptm, "_uid_cell_cache", None)
    if cache is not None and cache[0] is cd:
        return cache[1]
    mapping = {int(u): idx for u, idx in zip(cd["unique_id"].to_numpy(), cd.index.to_numpy())}
    eptm._uid_cell_cache = (cd, mapping)
    return mapping


def cell_index_for(eptm, uid):
    """``cell_df`` index label for a cell's ``unique_id``, or ``None`` if no live
    cell carries it (divided away / not yet born)."""
    return _uid_to_cell_index(eptm).get(int(uid))


_CELL_COMMIT_DEFAULTS = {"commit_state": 0.0, "commit_rate": 0.0, "commit_crit": 0.0, "commit_dt": 0.0}
_GROWER_3D_NAME = "_grow_committed_cells_3d"
# A dividing cell abandons its division once its prefered_vol exceeds this multiple
# of its critical volume without the actual volume crossing (contact inhibition).
_COMMIT_VOL_CAP = 3.0


def _ensure_cell_commit_cols(eptm):
    cd = eptm.cell_df
    for col, default in _CELL_COMMIT_DEFAULTS.items():
        if col not in cd.columns:
            cd[col] = default
    if "commit_type" not in cd.columns:
        cd["commit_type"] = ""
    if "commit_orientation" not in cd.columns:
        cd["commit_orientation"] = "vertical"


def _ensure_grower_3d(eptm, manager, geom):
    if any(tup[0].__name__ == _GROWER_3D_NAME for tup in manager.next):
        return
    manager.append(_grow_committed_cells_3d, geom=geom)


def _clear_cell_commit(eptm, idx, restore_type=None):
    cd = eptm.cell_df
    if restore_type:
        cd.loc[idx, "cell_type"] = restore_type
    for col, default in _CELL_COMMIT_DEFAULTS.items():
        cd.loc[idx, col] = default
    if "commit_type" in cd.columns:
        cd.loc[idx, "commit_type"] = ""
    if "commit_orientation" in cd.columns:
        cd.loc[idx, "commit_orientation"] = "vertical"


def division_3d(
        eptm, manager, geom="MonolayerGeometry", cell_uid=0, cell_type=None,
        crit_vol=2.0, growth_rate=0.1, orientation="vertical", dt=1.0,
):
    """Commit a 3D cell to division (monolayer / bulk analogue of
    :func:`division`).

    Records the cell's own growth parameters into its ``commit_*`` columns on
    ``cell_df``, flags it ``"dividing"``, and hands off to the batched grower
    (:func:`_grow_committed_cells_3d`), which grows every committed cell's
    ``prefered_vol`` (the CellVolumeElasticity reference) and splits it once its
    measured ``vol`` exceeds ``crit_vol``.

    Parameters
    ----------
    cell_uid : int
        unique_id of the dividing cell
    cell_type : str, optional
        target type stamped on mother and daughter once division occurs; the
        cell is flagged ``"dividing"`` while it grows
    crit_vol : float
        volume at which the cell divides (analogue of ``crit_area``)
    growth_rate : float
        per-unit-time increase of the prefered volume:
        V_0(t + dt) = V_0(t) * (1 + growth_rate * dt)
    orientation : {"vertical", "horizontal", "apical"}
        division-plane orientation passed to tyssue's monolayer cell_division;
        the emitting process randomizes it so growth is not all in one plane
    """
    cell_id = cell_index_for(eptm, cell_uid)
    if cell_id is None:
        print("Cell not found, skipping division")
        return
    cell_id = int(cell_id)
    _ensure_cell_commit_cols(eptm)
    cd = eptm.cell_df
    # Ignore a duplicate emission for a cell already mid-event (the coupling may
    # re-select a cell before its "dividing" relabel propagates back).
    if cd.loc[cell_id, "commit_state"] != 0.0:
        return
    if cell_type is not None and "cell_type" in cd.columns:
        cd.loc[cell_id, "cell_type"] = "dividing"
        cd.loc[cell_id, "commit_type"] = cell_type
    cd.loc[cell_id, "commit_state"] = 1.0
    cd.loc[cell_id, "commit_rate"] = growth_rate
    cd.loc[cell_id, "commit_crit"] = crit_vol
    cd.loc[cell_id, "commit_dt"] = dt
    cd.loc[cell_id, "commit_orientation"] = orientation
    _ensure_grower_3d(eptm, manager, geom)


def apoptosis_3d(
        eptm, manager, geom="MonolayerGeometry", cell_uid=0, crit_vol=0.5, shrink_rate=0.1, dt=1.0,
):
    """Commit a 3D cell to apoptotic death.

    Shrinks the cell's ``prefered_vol`` and, once its measured ``vol`` falls below
    ``crit_vol`` (or the death target collapses), marks it ``"dead"`` / not alive.
    The cell is left in the mesh (no volumetric topology surgery, which is
    numerically fragile in 3D) — a stable "necrotic" state that still lets the
    coupled death fluxes act on the tissue and shows up in the visualization."""
    cell_id = cell_index_for(eptm, cell_uid)
    if cell_id is None:
        print("Cell not found, skipping event")
        return
    cell_id = int(cell_id)
    _ensure_cell_commit_cols(eptm)
    cd = eptm.cell_df
    if cd.loc[cell_id, "commit_state"] != 0.0:
        return
    if "cell_type" in cd.columns:
        cd.loc[cell_id, "cell_type"] = "extruding"
    cd.loc[cell_id, "commit_state"] = 2.0
    cd.loc[cell_id, "commit_rate"] = shrink_rate
    cd.loc[cell_id, "commit_crit"] = crit_vol
    cd.loc[cell_id, "commit_dt"] = dt
    _ensure_grower_3d(eptm, manager, geom)


def _do_division_3d(eptm, geometry, cell_uid):
    """Split a division-ready 3D cell along its committed orientation, give the new
    cell a fresh unique_id, restore the target cell_type on both, and clear their
    commit flags.

    tyssue's ``cell_division`` copies the mother's cell_df row for the daughter
    (so both momentarily share the mother's unique_id) and calls ``reset_index``,
    which reindexes cell_df into an unordered-``set`` order — so the integer label
    it *returns* for the daughter can point at a different row afterwards. We
    therefore ignore that label and locate the daughter as the row that now
    duplicates the mother's unique_id, which is reset-order independent."""
    cell_id = cell_index_for(eptm, cell_uid)
    if cell_id is None:
        return
    cell_id = int(cell_id)
    cd = eptm.cell_df
    target_type = cd.loc[cell_id, "commit_type"] if "commit_type" in cd.columns else ""
    orientation = cd.loc[cell_id, "commit_orientation"] if "commit_orientation" in cd.columns else "vertical"
    # Reset the prefered size so both daughters start near the baseline instead of
    # inheriting the grown (~crit) target and immediately re-dividing. (The daughter
    # copies the mother's row, so it inherits this reset value.)
    cd.loc[cell_id, "prefered_vol"] = 1.0
    if "prefered_area" in cd.columns:
        cd.loc[cell_id, "prefered_area"] = 1.0
    # A monolayer cell_division can raise "invalid topology" for some cells PART
    # WAY through, after already mutating the mesh (leaving an orphan face that the
    # next geometry update chokes on). Snapshot first and roll back on failure so a
    # skipped division never corrupts the tissue.
    eptm.backup()
    try:
        monolayer_cell_division(eptm, cell_id, orientation=orientation)
    except Exception as exc:  # noqa: BLE001
        print(f"cell_division failed for cell {cell_uid} ({type(exc).__name__}: {exc}); rolling back")
        eptm.restore()
        # restore() can shrink the vertex set back below its grown size; refresh
        # active_verts (reset_index/restore don't) so the solver's next gradient
        # doesn't index vertices that no longer exist.
        eptm.reset_topo()
        cid = cell_index_for(eptm, cell_uid)
        if cid is not None:
            cid = int(cid)
            tt = eptm.cell_df.loc[cid, "commit_type"] if "commit_type" in eptm.cell_df.columns else ""
            eptm.cell_df.loc[cid, "prefered_vol"] = 1.0
            if "prefered_area" in eptm.cell_df.columns:
                eptm.cell_df.loc[cid, "prefered_area"] = 1.0
            _clear_cell_commit(eptm, cid, restore_type=(str(tt) if tt else None))
        eptm.network_changed = True
        return
    # Mother and daughter now both carry cell_uid; give the daughter a fresh id.
    cd = eptm.cell_df
    twins = list(cd.index[cd["unique_id"] == cell_uid])
    restore = target_type if target_type else None
    if len(twins) < 2:
        # Division didn't duplicate as expected — just tidy the mother's commit.
        for idx in twins:
            _clear_cell_commit(eptm, idx, restore_type=restore)
        eptm.network_changed = True
        return
    daughter = twins[-1]
    new_uid = int(cd["unique_id"].max()) + 1
    cd.loc[daughter, "unique_id"] = new_uid
    if "id" in cd.columns:
        cd.loc[daughter, "id"] = int(cd["id"].max()) + 1
    for idx in twins:
        _clear_cell_commit(eptm, idx, restore_type=restore)
    # Reconcile indices and refresh geometry BEFORE any further division in the
    # same grower burst (like the 2D _do_division): a subsequent split must see a
    # consistent, up-to-date mesh, otherwise it builds a degenerate face and the
    # next MonolayerGeometry.update_all mismatches face vs edge-group counts. Index
    # relabelling here is uid-safe — reset_index keeps column values (unique_id).
    eptm.reset_index()
    # reset_topo refreshes active_verts to the relabelled vertex set (reset_index
    # doesn't) so the solver's next gradient indexes live vertices only.
    eptm.reset_topo()
    geometry.update_all(eptm)
    eptm.network_changed = True
    print(f"cell n°{daughter} is born")


def _do_necrosis_3d(eptm, cell_uid):
    """Mark a death-ready 3D cell dead (cell_type='dead', is_alive=0) and clear its
    commit flags. Kept in the mesh — see :func:`apoptosis_3d`."""
    cell_id = cell_index_for(eptm, cell_uid)
    if cell_id is None:
        return
    cell_id = int(cell_id)
    if "cell_type" in eptm.cell_df.columns:
        eptm.cell_df.loc[cell_id, "cell_type"] = "dead"
    if "is_alive" in eptm.cell_df.columns:
        eptm.cell_df.loc[cell_id, "is_alive"] = 0
    _clear_cell_commit(eptm, cell_id)


def _grow_committed_cells_3d(eptm, manager, geom="MonolayerGeometry"):
    """One vectorized grow/shrink pass over every committed 3D cell, then fire the
    real division / necrosis for any that crossed their own threshold. Re-queues
    itself while commitments remain. Mirrors :func:`_grow_committed_cells`."""
    geometry = GEOMETRY_MAP[geom] if isinstance(geom, str) else geom
    cd = eptm.cell_df
    if "commit_state" not in cd.columns:
        return
    state = cd["commit_state"].to_numpy()
    div_mask = state == 1.0
    ext_mask = state == 2.0
    if not div_mask.any() and not ext_mask.any():
        return

    # V_0(t+dt) = V_0(t) * (1 +/- dt*rate); prefered_area tracks V_0^(2/3) so the
    # face-area term doesn't fight the volume growth (same coupling tyssue's
    # monolayer grow() action uses).
    rate = cd["commit_rate"].to_numpy(dtype=float)
    dt = cd["commit_dt"].to_numpy(dtype=float)
    grow = 1.0 + dt * rate
    shrink = 1.0 - dt * rate
    prefered_v = cd["prefered_vol"].to_numpy(dtype=float).copy()
    prefered_v[div_mask] *= grow[div_mask]
    prefered_v[ext_mask] *= shrink[ext_mask]
    cd["prefered_vol"] = prefered_v
    if "prefered_area" in cd.columns:
        pa = cd["prefered_area"].to_numpy(dtype=float).copy()
        pa[div_mask] *= grow[div_mask] ** (2.0 / 3.0)
        pa[ext_mask] *= np.clip(shrink[ext_mask], 0.0, None) ** (2.0 / 3.0)
        cd["prefered_area"] = pa

    vol = cd["vol"].to_numpy(dtype=float)
    crit = cd["commit_crit"].to_numpy(dtype=float)
    uid = cd["unique_id"].to_numpy()
    DEATH_FLOOR = 0.1
    div_ready = uid[div_mask & (vol > crit)]
    ext_ready = uid[ext_mask & ((vol < crit) | (prefered_v < DEATH_FLOOR))]

    # Contact inhibition: a dividing cell wedged in the crowded tumor core can
    # never reach crit_vol however much its prefered_vol is inflated. Left
    # unchecked such cells accumulate as permanently-committed "dividing" cells
    # that grow every step forever — a runaway that also balloons runtime. Once a
    # cell's prefered_vol has been driven past _COMMIT_VOL_CAP x crit without its
    # actual vol crossing, give up on that division (the coupling may retry later
    # when the tissue has room).
    jammed = uid[div_mask & (vol <= crit) & (prefered_v > _COMMIT_VOL_CAP * crit)]

    for cell_uid in div_ready:
        _do_division_3d(eptm, geometry, int(cell_uid))
    for cell_uid in ext_ready:
        _do_necrosis_3d(eptm, int(cell_uid))
    for cell_uid in jammed:
        idx = cell_index_for(eptm, int(cell_uid))
        if idx is not None:
            idx = int(idx)
            tt = eptm.cell_df.loc[idx, "commit_type"] if "commit_type" in eptm.cell_df.columns else ""
            eptm.cell_df.loc[idx, "prefered_vol"] = 1.0
            if "prefered_area" in eptm.cell_df.columns:
                eptm.cell_df.loc[idx, "prefered_area"] = 1.0
            _clear_cell_commit(eptm, idx, restore_type=(str(tt) if tt else None))

    cd = eptm.cell_df
    if "commit_state" in cd.columns and (cd["commit_state"].to_numpy() != 0.0).any():
        _ensure_grower_3d(eptm, manager, geom)
