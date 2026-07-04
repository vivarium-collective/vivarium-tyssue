"""tyssue topology robustness shims for long crypt / division / extrusion runs.

The Gillespie crypt composite drives many ``cell_division`` and ``remove_face``
(apoptosis / extrusion) operations plus the default ``reconnect`` (T1) event.
Two tyssue 1.x topology helpers crash on the transient, slightly-malformed mesh
states these produce. Both are library bugs we cannot fix in site-packages, so
we monkey-patch behavior-identical, defensively-guarded replacements over the
tyssue module namespaces. Applied idempotently from
:func:`vivarium_tyssue.core.build_core` (via ``behaviors``) before any composite
runs.

1. ``base_topology.drop_two_sided_faces`` — builds
   ``num_sides = edge_df.groupby("face").size()`` (indexed only by the faces that
   still have edges) and then indexes ``face_df[num_sides < 3]``. After a
   ``remove_face`` / ``collapse_edge`` leaves an *orphan* face (present in
   ``face_df`` but with no rows in ``edge_df``), the boolean Series is not aligned
   to ``face_df.index`` and pandas raises
   ``IndexingError: Unalignable boolean Series``. We reindex ``num_sides`` onto
   ``face_df.index`` (orphans -> 0 sides) so the mask always aligns; orphan faces
   are then correctly treated as <3-sided and dropped. It is referenced as a
   module global by ``remove_face``, ``collapse_edge`` and (via ``collapse_edge``)
   ``type1_transition``, so patching the ``base_topology`` name covers every path.

2. ``sheet_topology.split_vert`` — ``detach_vertices`` (fired by the ``reconnect``
   event on rank>=4 rosette vertices) calls ``split_vert(sheet, vert)`` which does
   ``(prev_v,) = face_edges[face_edges["trgt"] == vert]["srce"]`` (and the same for
   ``next_v``). This assumes the chosen face borders ``vert`` with exactly one
   incoming and one outgoing edge. On a transiently degenerate face (a vertex that
   appears twice around a face, or a 2-sided remnant) the match has 0 or >1 rows and
   the unpack raises ``ValueError: too many values to unpack``. Rather than abort the
   whole step, we skip the pathological vertex (return ``[]``, i.e. "no split
   performed") — exactly how ``detach_vertices`` already tolerates a failed detach.
   Patched on ``sheet_topology`` *and* on ``behaviors.sheet.actions`` (which binds
   ``sheet_split = sheet_topology.split_vert`` at import time and is the name
   ``detach_vertices`` actually calls).
"""


def apply_tyssue_topology_patches() -> None:
    """Idempotently install the topology robustness shims. No-op if the tyssue
    topology modules can't be imported."""
    _patch_drop_two_sided_faces()
    _patch_split_vert()


def _patch_drop_two_sided_faces() -> None:
    try:
        from tyssue.topology import base_topology as _bt
    except Exception:
        return
    if getattr(_bt.drop_two_sided_faces, "_vivarium_tyssue_patched", False):
        return

    def drop_two_sided_faces(eptm):
        """Reindex-safe drop of 1/2-sided faces (see module docstring, bug 1)."""
        # size() is indexed by the faces that still own edges; reindex onto the
        # full face_df.index so orphan faces (no edges) count as 0 sides and the
        # boolean mask always aligns instead of raising Unalignable boolean Series.
        num_sides = (
            eptm.edge_df.groupby("face").size().reindex(eptm.face_df.index, fill_value=0)
        )
        if num_sides.min() > 2:
            return
        two_sided = eptm.face_df.index[num_sides.values < 3]
        edges = eptm.edge_df[eptm.edge_df["face"].isin(two_sided)].index
        eptm.edge_df.drop(edges, axis=0, inplace=True)
        eptm.face_df.drop(two_sided, axis=0, inplace=True)

    drop_two_sided_faces._vivarium_tyssue_patched = True
    _bt.drop_two_sided_faces = drop_two_sided_faces


def _patch_split_vert() -> None:
    try:
        from tyssue.topology import sheet_topology as _st
    except Exception:
        return
    if getattr(_st.split_vert, "_vivarium_tyssue_patched", False):
        return

    _orig_split_vert = _st.split_vert

    def split_vert(sheet, vert, face=None, *args, **kwargs):
        """Guarded split_vert: skip (return []) a vertex whose chosen face does not
        border it with exactly one in/out edge, instead of raising ValueError on the
        ``(prev_v,) = ...`` unpack (see module docstring, bug 2)."""
        try:
            return _orig_split_vert(sheet, vert, face, *args, **kwargs)
        except ValueError:
            # too many / too few values to unpack -> degenerate local topology;
            # leave the vertex intact (reconnect tolerates a skipped detach).
            return []

    split_vert._vivarium_tyssue_patched = True
    _st.split_vert = split_vert

    # detach_vertices binds `sheet_split = sheet_topology.split_vert` at import
    # time, so also rebind the already-imported name it actually calls.
    try:
        from tyssue.behaviors.sheet import actions as _actions
        _actions.sheet_split = split_vert
    except Exception:
        pass
