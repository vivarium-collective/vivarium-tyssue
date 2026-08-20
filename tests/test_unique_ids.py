"""unique_id bookkeeping across topology changes.

tyssue's low-level helpers build every new element by cloning an existing row, so
the clone inherits the original's ``unique_id`` (see ``refresh_unique_ids``). These
tests pin the repair: after a division, no element may carry a duplicate id, and
exactly one cell keeps the mother's.
"""
import numpy as np
import pytest

from tyssue import Sheet, SheetGeometry
from tyssue.generation import three_faces_sheet

from vivarium_tyssue.behaviors.behaviors import (
    _do_division,
    divide_cell,
    face_index_for,
    refresh_unique_ids,
)

ELEMENTS = ("vert", "edge", "face")


def _sheet():
    sheet = Sheet("test", *three_faces_sheet())
    SheetGeometry.update_all(sheet)
    sheet.face_df["prefered_area"] = 1.0
    sheet.face_df["commit_type"] = ""
    sheet.face_df["commit_state"] = 0.0
    sheet.face_df["commit_rate"] = 0.0
    sheet.face_df["commit_crit"] = 0.0
    sheet.face_df["commit_dt"] = 0.0
    return sheet


def _duplicates(eptm):
    return {
        el: int(eptm.datasets[el]["unique_id"].duplicated().sum())
        for el in ELEMENTS
        if "unique_id" in eptm.datasets[el].columns
    }


def test_fixture_starts_clean():
    sheet = _sheet()
    assert _duplicates(sheet) == {el: 0 for el in ELEMENTS}


def test_division_leaves_no_duplicate_ids():
    sheet = _sheet()
    mother_uid = int(sheet.face_df.loc[0, "unique_id"])
    _do_division(sheet, SheetGeometry, mother_uid)

    assert len(sheet.face_df) == 4, "the division should have added a face"
    dups = _duplicates(sheet)
    assert dups == {el: 0 for el in ELEMENTS}, f"duplicate unique_ids remain: {dups}"


def test_one_cell_keeps_the_mother_id():
    sheet = _sheet()
    mother_uid = int(sheet.face_df.loc[0, "unique_id"])
    _do_division(sheet, SheetGeometry, mother_uid)

    holders = (sheet.face_df["unique_id"] == mother_uid).sum()
    assert holders == 1, "exactly one daughter should continue the mother's lineage"
    assert face_index_for(sheet, mother_uid) is not None


def test_daughter_ids_do_not_collide_with_retired_ids():
    """A fresh id must never reuse one already handed out, even after removals."""
    sheet = _sheet()
    seen = set(sheet.face_df["unique_id"].astype(int))
    for _ in range(3):
        uid = int(sheet.face_df["unique_id"].iloc[0])
        _do_division(sheet, SheetGeometry, uid)
        new = set(sheet.face_df["unique_id"].astype(int)) - seen
        assert not (new & seen), "a reissued id collided with an existing one"
        seen |= new
    assert _duplicates(sheet) == {el: 0 for el in ELEMENTS}


def test_divide_cell_helper_also_repairs():
    sheet = _sheet()
    uid = int(sheet.face_df.loc[0, "unique_id"])
    divide_cell(sheet, SheetGeometry, radius=1.0, cell_uid=uid)
    assert _duplicates(sheet) == {el: 0 for el in ELEMENTS}


def test_refresh_is_idempotent_and_noop_when_clean():
    sheet = _sheet()
    before = {el: sheet.datasets[el]["unique_id"].tolist() for el in ELEMENTS}
    assert refresh_unique_ids(sheet) == {}
    for el in ELEMENTS:
        assert sheet.datasets[el]["unique_id"].tolist() == before[el]


def test_refresh_repairs_an_injected_duplicate_keeping_the_first():
    sheet = _sheet()
    sheet.face_df.loc[2, "unique_id"] = sheet.face_df.loc[0, "unique_id"]
    kept = int(sheet.face_df.loc[0, "unique_id"])
    repaired = refresh_unique_ids(sheet)
    assert repaired == {"face": 1}
    assert int(sheet.face_df.loc[0, "unique_id"]) == kept, "first occurrence keeps the id"
    assert int(sheet.face_df.loc[2, "unique_id"]) != kept
    assert _duplicates(sheet) == {el: 0 for el in ELEMENTS}


# --- density regulator: degenerate faces must not abort a run -----------------

def test_cell_to_density_survives_a_zero_area_face():
    """A momentarily degenerate face (area 0) used to raise ZeroDivisionError from
    the Gillespie's density regulator and kill the whole run."""
    import pandas as pd
    from vivarium_tyssue.models.crypt_gillespie.jump_rates import cell_to_density

    face_df = pd.DataFrame({"unique_id": [0, 1, 2], "area": [1.0, 0.0, 0.5]})
    value = cell_to_density(face_df, 1)
    assert np.isfinite(value)
    assert value > 0


def test_cell_to_density_clamp_is_on_the_saturated_plateau():
    """The clamp must be behaviour-preserving: reg_pol saturates at 1 for any area
    at or below ~0.77 (K=1, k=0.3), so a degenerate face and a merely small one
    give the same regulation value."""
    import pandas as pd
    from vivarium_tyssue.processes.gillespie import reg_pol
    from vivarium_tyssue.models.crypt_gillespie.jump_rates import cell_to_density

    face_df = pd.DataFrame({"unique_id": [0, 1], "area": [0.0, 0.5]})
    K, k = 1.0, 0.3
    assert reg_pol(cell_to_density(face_df, 0), K, k) == 1
    assert reg_pol(cell_to_density(face_df, 1), K, k) == 1


def test_cell_to_density_unchanged_for_normal_areas():
    import pandas as pd
    from vivarium_tyssue.models.crypt_gillespie.jump_rates import cell_to_density

    face_df = pd.DataFrame({"unique_id": [0, 1], "area": [1.0, 2.0]})
    assert cell_to_density(face_df, 0) == pytest.approx(1.0)
    assert cell_to_density(face_df, 1) == pytest.approx(0.5)
