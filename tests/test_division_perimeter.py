"""Growth and division behaviors keep a cell's target shape.

``grow_divide`` inflates a committed cell's ``prefered_area``; with
``scale_perimeter`` (the default) its ``prefered_perimeter`` follows as
P_0 ∝ sqrt(A_0) -- doubling P_0 quadruples A_0 -- and both halves get the
pre-growth perimeter target back on division. ``grow_cells`` and ``split_cells``
do the same in two separate steps, for processes that decide when each happens.
"""
from collections import deque

import numpy as np
import pytest

from tyssue import Sheet, SheetGeometry
from tyssue.generation import three_faces_sheet

from vivarium_tyssue.behaviors.behaviors import (
    _do_division,
    _grow_committed_cells,
    grow_cells,
    grow_divide,
    split_cells,
)

P0 = 3.5


class _Manager:
    """The two members of tyssue's EventManager the commit/grower path touches."""

    def __init__(self):
        self.next = deque()

    def append(self, func, **kwargs):
        self.next.append((func, kwargs))


def _sheet():
    sheet = Sheet("test", *three_faces_sheet())
    SheetGeometry.update_all(sheet)
    sheet.face_df["prefered_area"] = 1.0
    sheet.face_df["prefered_perimeter"] = P0
    sheet.face_df["cell_type"] = "normal"
    return sheet


def _commit(sheet, manager, face=0, **kwargs):
    uid = int(sheet.face_df.loc[face, "unique_id"])
    # crit_area out of reach, so the grower only grows.
    grow_divide(sheet, manager, cell_uid=uid, cell_type="normal", crit_area=1e9,
                growth_rate=0.1, dt=1.0, **kwargs)
    return uid


def test_perimeter_target_follows_sqrt_of_area_target():
    sheet, manager = _sheet(), _Manager()
    _commit(sheet, manager)
    for _ in range(30):
        _grow_committed_cells(sheet, manager)
    fd = sheet.face_df
    a_ratio = fd.loc[0, "prefered_area"] / 1.0
    assert a_ratio > 4.0
    assert fd.loc[0, "prefered_perimeter"] == pytest.approx(P0 * np.sqrt(a_ratio), rel=1e-12)
    # Uncommitted neighbours are untouched.
    assert (fd.loc[1:, "prefered_perimeter"] == P0).all()
    assert (fd.loc[1:, "prefered_area"] == 1.0).all()


def test_quadrupled_area_target_doubles_perimeter_target():
    sheet, manager = _sheet(), _Manager()
    _commit(sheet, manager)
    fd = sheet.face_df
    while fd.loc[0, "prefered_area"] < 4.0:
        _grow_committed_cells(sheet, manager)
        fd = sheet.face_df
    a, p = fd.loc[0, "prefered_area"], fd.loc[0, "prefered_perimeter"]
    assert (p / P0) ** 2 == pytest.approx(a, rel=1e-12)


def test_division_restores_pre_growth_perimeter_on_both_halves():
    sheet, manager = _sheet(), _Manager()
    uid = _commit(sheet, manager)
    for _ in range(10):
        _grow_committed_cells(sheet, manager)
    assert sheet.face_df.loc[0, "prefered_perimeter"] > P0

    _do_division(sheet, SheetGeometry, uid)
    fd = sheet.face_df
    assert len(fd) == 4
    assert (fd["prefered_perimeter"] == P0).all()
    assert (fd["commit_perim0"] == 0.0).all()
    assert (fd["prefered_area"] == 1.0).all()


def test_scale_perimeter_false_keeps_perimeter_target():
    sheet, manager = _sheet(), _Manager()
    _commit(sheet, manager, scale_perimeter=False)
    for _ in range(10):
        _grow_committed_cells(sheet, manager)
    fd = sheet.face_df
    assert fd.loc[0, "prefered_area"] > 2.0
    assert fd.loc[0, "prefered_perimeter"] == P0


def test_recommit_while_growing_keeps_original_base():
    sheet, manager = _sheet(), _Manager()
    uid = _commit(sheet, manager)
    for _ in range(10):
        _grow_committed_cells(sheet, manager)
    _commit(sheet, manager)          # committed again mid-growth
    assert sheet.face_df.loc[0, "commit_perim0"] == P0
    _do_division(sheet, SheetGeometry, uid)
    assert (sheet.face_df["prefered_perimeter"] == P0).all()


def test_sheet_without_perimeter_target_still_grows():
    sheet, manager = _sheet(), _Manager()
    sheet.face_df.drop(columns="prefered_perimeter", inplace=True)
    _commit(sheet, manager)
    for _ in range(5):
        _grow_committed_cells(sheet, manager)
    fd = sheet.face_df
    assert "prefered_perimeter" not in fd.columns
    assert fd.loc[0, "prefered_area"] == pytest.approx(1.1 ** 5)


# ----------------------------------------------------- grow_cells / split_cells
def test_grow_cells_grows_only_listed_cells_at_constant_shape():
    sheet = _sheet()
    uid = int(sheet.face_df.loc[1, "unique_id"])
    for _ in range(20):
        grow_cells(sheet, None, cell_uids=[uid], growth_rate=0.1, dt=0.5)
    fd = sheet.face_df
    a = 1.05 ** 20
    assert fd.loc[1, "prefered_area"] == pytest.approx(a)
    assert fd.loc[1, "prefered_perimeter"] == pytest.approx(P0 * np.sqrt(a), rel=1e-12)
    others = fd.index != 1
    assert (fd.loc[others, "prefered_area"] == 1.0).all()
    assert (fd.loc[others, "prefered_perimeter"] == P0).all()


def test_grow_cells_without_perimeter_scaling():
    sheet = _sheet()
    grow_cells(sheet, None, cell_uids=sheet.face_df["unique_id"].tolist(),
               growth_rate=1.0, dt=1.0, scale_perimeter=False)
    assert (sheet.face_df["prefered_area"] == 2.0).all()
    assert (sheet.face_df["prefered_perimeter"] == P0).all()


def test_split_cells_divides_at_once_and_restores_targets():
    sheet = _sheet()
    uid = int(sheet.face_df.loc[0, "unique_id"])
    for _ in range(5):
        grow_cells(sheet, None, cell_uids=[uid], growth_rate=0.1, dt=1.0)
    before = sheet.face_df["unique_id"].tolist()
    split_cells(sheet, None, geom=SheetGeometry, cell_uids=[uid], prefered_area=1.0)

    fd = sheet.face_df
    assert len(fd) == 4
    assert (fd["prefered_area"] == 1.0).all()
    np.testing.assert_allclose(fd["prefered_perimeter"], P0, rtol=1e-12)
    assert (fd["unique_id"] == uid).sum() == 1      # the mother keeps her id
    assert not fd["unique_id"].duplicated().any()
    assert sheet.network_changed
    # The daughter (the one new uid) records her mother; every other cell has none.
    daughter = ~fd["unique_id"].isin(before)
    assert fd.loc[daughter, "mother_uid"].tolist() == [uid]
    assert (fd.loc[~daughter, "mother_uid"] == -1).all()


def test_split_cells_resolves_every_uid_after_reindexing():
    sheet = _sheet()
    uids = sheet.face_df["unique_id"].tolist()
    split_cells(sheet, None, geom="SheetGeometry", cell_uids=uids + [10**6])
    assert len(sheet.face_df) == 6                   # every cell split; unknown uid ignored
    assert not sheet.face_df["unique_id"].duplicated().any()
