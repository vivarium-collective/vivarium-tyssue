"""``unique_id`` bookkeeping across tyssue topology changes.

Every tyssue low-level topology helper builds a new element by *cloning* an
existing row: ``face_division`` concatenates the mother's face row (and one of its
edge rows, twice), ``add_vert``, ``split_vert`` and ``remove_face`` concatenate a
source vertex's row, ``close_face`` an edge's. The clone therefore carries the
original's ``unique_id`` and nothing downstream mints a new one —
``reset_index(order=True)`` renumbers the *index* but preserves column values.

tyssue repairs this inside its own division behavior
(``tyssue.behaviors.sheet.basic_events.division``), which is why the low-level
helpers leave it alone; a caller driving ``cell_division`` / ``remove_face`` /
``split_vert`` directly, as this package does, has to do the same.

Left unrepaired the duplicates are not cosmetic: ``face_index_for`` resolves a uid
to the *first* matching row, so after a division one twin becomes permanently
unaddressable and can never be selected for division, differentiation or extrusion.

Convention (tyssue's): the original keeps its id and only the clone is renumbered,
so one daughter continues the mother's lineage. tyssue's helpers always *append*
the clone, so "first occurrence wins" identifies the original — which is why
:func:`refresh_unique_ids` must run before any ``reset_index`` reorders the rows.
"""
import numpy as np

UID_ELEMENTS = ("vert", "edge", "face", "cell")


def reserve_uids(eptm, element, count):
    """Reserve ``count`` fresh unique_ids for ``element``, advancing the per-element
    counter tyssue seeds in ``Epithelium.__init__``
    (``specs[element]['unique_id_max']``).

    Falls back to ``max(unique_id) + 1`` so a mesh whose specs lack the counter — or
    whose ids have outrun it — still gets non-colliding ids. Because the counter only
    ever moves forward, an id retired by an extrusion is never handed out again.
    """
    specs = getattr(eptm, "specs", None)
    elem_specs = specs.setdefault(element, {}) if isinstance(specs, dict) else {}
    df = eptm.datasets.get(element)
    start = int(elem_specs.get("unique_id_max", 0) or 0)
    if df is not None and len(df) and "unique_id" in df.columns:
        start = max(start, int(df["unique_id"].max()) + 1)
    elem_specs["unique_id_max"] = start + count
    return np.arange(start, start + count)


def refresh_unique_ids(eptm, elements=UID_ELEMENTS):
    """Renumber every element row that duplicates an earlier row's ``unique_id``.

    Call immediately after a topology change and *before* ``reset_index``. Idempotent
    and a no-op on a clean mesh, so it is safe to call defensively. Returns
    ``{element: n_renumbered}`` for whatever needed repair.
    """
    repaired = {}
    datasets = getattr(eptm, "datasets", None)
    if not datasets:
        return repaired
    for element in elements:
        df = datasets.get(element)
        if df is None or not len(df) or "unique_id" not in df.columns:
            continue
        dup = df["unique_id"].duplicated(keep="first").to_numpy()
        n = int(dup.sum())
        if not n:
            continue
        df.loc[df.index[dup], "unique_id"] = reserve_uids(eptm, element, n)
        repaired[element] = n
    return repaired
