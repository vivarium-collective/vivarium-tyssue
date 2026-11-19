"""The bundled meshes must load without unpickling anything.

compose-api refuses an HDF5 upload that pandas/PyTables would unpickle code from
(biosimulations/compose-api#538, ``compose_api/registry/uploads.py``): an object
column (a VLArray with ``PSEUDOATOM=object``, which pandas writes for any
``object`` column in fixed format), or an attribute whose pickle imports or calls
something. Plain-data pickles (pandas' table-format ``info`` dicts) are allowed.
This mirrors that check with PyTables, reading attributes raw so nothing is
unpickled here either.

tyssue's ``save_datasets`` writes fixed format, which pickles string and
mixed-type columns: store integers as int64 and write frames that hold text with
``format="table"``.
"""
import pickletools
from pathlib import Path

import numpy as np
import pytest

tables = pytest.importorskip("tables")

DATASETS = Path(__file__).resolve().parents[1] / "workspace" / "datasets"

# Opcodes through which unpickling imports or calls something.
_CALLS = {"GLOBAL", "STACK_GLOBAL", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "REDUCE",
          "BUILD", "EXT1", "EXT2", "EXT4"}


def _runs_code(raw):
    try:
        return any(op.name in _CALLS for op, _, _ in pickletools.genops(raw))
    except ValueError:
        return True


def _hazards(path):
    found = []
    with tables.open_file(str(path), "r") as h5:
        version = h5.format_version or "0"
        if int(version.partition(".")[0]) < 2:
            found.append(f"PyTables format {version}")
        for node in h5.walk_nodes("/"):
            attrs = node._v_attrs
            for name in attrs._v_attrnames:
                raw = attrs._g_getattr(node, name)         # raw: never unpickles
                if isinstance(raw, np.ndarray) and raw.ndim == 0:
                    raw = raw.item()
                if isinstance(raw, str):
                    raw = raw.encode("utf-8", "surrogateescape")
                if not isinstance(raw, bytes):
                    continue
                if name == "PSEUDOATOM" and raw == b"object":
                    found.append(f"{node._v_pathname}: object column")
                elif raw.endswith(b".") and raw not in (b"0", b"0.") and _runs_code(raw):
                    found.append(f"{node._v_pathname}: attribute {name!r} pickle runs code")
    return found


@pytest.mark.parametrize("path", sorted(DATASETS.glob("*.hf5")), ids=lambda p: p.name)
def test_dataset_has_no_pickled_code(path):
    assert _hazards(path) == []
