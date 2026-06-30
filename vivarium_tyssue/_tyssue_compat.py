"""tyssue / numpy 2.x compatibility shims.

tyssue 1.1.0's ``merge_vertices`` (tyssue/behaviors/sheet/actions.py) does
``np.random.shuffle(edge_df.index.to_numpy())``. Under numpy 2.x,
``Index.to_numpy()`` returns a **read-only** array, so ``shuffle`` raises
``ValueError: array is read-only``. This fires whenever the EventManager's
default ``reconnect`` event (EulerSolver ``auto_reconnect: true``) finds a short
edge — i.e. on essentially any non-trivial run with divisions/extrusions.

We patch ``merge_vertices`` with a behavior-identical version that shuffles a
writeable copy. ``reconnect`` (in basic_events) binds ``merge_vertices`` from
``actions`` at import time, so we replace it in both module namespaces. Applied
from :func:`vivarium_tyssue.core.build_core`, which runs before any composite.
"""
import numpy as np


def apply_tyssue_compat() -> None:
    """Idempotently patch tyssue's merge_vertices for numpy 2.x. No-op if the
    tyssue behavior modules can't be imported."""
    try:
        from tyssue.behaviors.sheet import actions, basic_events
    except Exception:
        return

    collapse_edge = actions.collapse_edge

    def merge_vertices(sheet):
        d_min = sheet.settings.get("threshold_length", 1e-3)
        short = np.array(
            sheet.edge_df[sheet.edge_df["length"] < d_min].index.to_numpy(), copy=True
        )
        np.random.shuffle(short)
        if not short.shape[0]:
            return -1
        while short.shape[0]:
            collapse_edge(sheet, short[0], allow_two_sided=False)
            short = np.array(
                sheet.edge_df[sheet.edge_df["length"] < d_min].index.to_numpy(), copy=True
            )
            np.random.shuffle(short)
        return 0

    merge_vertices._vivarium_tyssue_patched = True  # marker for tests
    actions.merge_vertices = merge_vertices
    basic_events.merge_vertices = merge_vertices
