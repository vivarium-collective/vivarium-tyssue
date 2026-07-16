"""Regression test for DataFrameParquetEmitter string-type normalization.

pandas 3 infers an object-string column as pyarrow ``large_string`` in some
frames and ``string`` in others (e.g. the transient ``commit_type`` column the
division/extrusion behaviors add mid-run), which made ``concat_tables`` at flush
raise ``ArrowTypeError: incompatible types: large_string vs string``. The emitter
now downcasts every frame's large-string fields to the small variant before
buffering. Here we force the mismatch explicitly (via ArrowDtype, so it
reproduces even on pandas 2) and assert the flush succeeds and round-trips.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from process_bigraph import allocate_core

from vivarium_tyssue.emitters import DataFrameParquetEmitter


def _frame(commit_type_values, dtype):
    df = pd.DataFrame({
        "unique_id": [1, 2, 3],
        "area": [1.0, 1.1, 1.2],
        "commit_type": pd.Series(commit_type_values, dtype=dtype),
    })
    return df


def test_flush_handles_mixed_string_and_large_string(tmp_path):
    emitter = DataFrameParquetEmitter(
        {"out_dir": str(tmp_path), "batch_size": 100}, core=allocate_core()
    )

    # Frame 1: commit_type as pyarrow large_string; Frame 2: as string. Without
    # normalization these two buffered tables can't be concatenated.
    f1 = _frame(["", "", ""], pd.ArrowDtype(pa.large_string()))
    f2 = _frame(["sc", "", "pc"], pd.ArrowDtype(pa.string()))

    emitter.update({"global_time": 0.0, "Datasets_face_df": f1})
    emitter.update({"global_time": 0.005, "Datasets_face_df": f2})
    emitter.close()  # flushes the trailing buffer — must not raise ArrowTypeError

    files = list(tmp_path.rglob("*.pq"))
    assert files, "expected a parquet file to be written"
    table = pq.read_table(files[0])
    assert table.num_rows == 6
    assert pa.types.is_string(table.schema.field("commit_type").type)
