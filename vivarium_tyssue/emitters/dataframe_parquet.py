"""DataFrame-native Parquet emitter for tyssue (and any process emitting pandas DataFrames).

The generic ``pbg_emitters.ParquetEmitter`` is *row-oriented*: every emit is
``flatten_dict``-ed into a single history row, with per-field numpy/Polars dtype
reconciliation. That model fights tyssue, whose per-tick observables are whole
pandas DataFrames (``vert_df`` 410x11, ``edge_df`` ~1200xN, ``face_df``,
``cell_df``) — they either blow up to a nested ``Object`` column polars can't
write, or have to be melted into ragged list-columns.

``DataFrameParquetEmitter`` is *table-oriented* instead. tyssue dataframes are
already long tables (one row per mesh element), so each DataFrame-valued input
port is streamed to its OWN hive-partitioned Parquet dataset via Arrow's
columnar path (``pyarrow.Table.from_pandas`` -> ``ParquetWriter``), with a
prepended ``time`` (+ ``experiment_id``, ``element_id``) column. This is both:

* **faithful** — full per-element trajectories in long format
  ``(experiment_id, time, element_id, <df columns...>)``; and
* **fast / lean** — columnar pandas->Arrow->Parquet with no per-cell Python
  flattening, no dtype bouncing, streaming batched append (bounded memory),
  native Parquet compression.

Output layout (one dataset per element type)::

    <out_uri>/<experiment_id>/frames/<port>/[<key>=<val>/...]/<n>.pq

Read back is trivial long-format::

    duckdb.sql("SELECT * FROM read_parquet('<out>/<exp>/frames/vert_df/**/*.pq')")

Lifecycle mirrors ``ParquetEmitter``: ``update(state)`` per tick, ``close()`` to
flush the trailing batch, and the ``flush_all_in_composite(composite)`` helper
the dashboard runner calls after the run loop.
"""
from __future__ import annotations

import os
from typing import Any

from process_bigraph.emitter import Emitter


class DataFrameParquetEmitter(Emitter):
    """Table-oriented Parquet emitter: one hive-partitioned dataset per DataFrame port."""

    config_schema = {
        **Emitter.config_schema,
        # Local dir or full fsspec URI for the output root (one required).
        "out_dir":           {"_type": "string",       "_default": ""},
        "out_uri":           {"_type": "string",       "_default": ""},
        # Ticks buffered per frame before a Parquet file is flushed.
        "batch_size":        {"_type": "integer",      "_default": 100},
        # Hive partition keys, read from `metadata` (e.g. ["experiment_id"]).
        "partitioning_keys": {"_type": "list[string]", "_default": []},
        "metadata":          {"_type": "map",          "_default": {}},
        # Which emitted scalar carries the simulation time stamped onto each row.
        "time_field":        {"_type": "string",       "_default": "global_time"},
    }

    def __init__(self, config: dict[str, Any], core: Any) -> None:
        super().__init__(config, core)

        if config.get("out_uri"):
            self.out_uri: str = config["out_uri"]
        elif config.get("out_dir"):
            self.out_uri = os.path.abspath(config["out_dir"])
        else:
            raise ValueError(
                "DataFrameParquetEmitter requires config['out_dir'] or config['out_uri']"
            )

        self.batch_size: int = max(1, int(config.get("batch_size", 100)))
        self.partitioning_keys: list[str] = list(config.get("partitioning_keys") or [])
        self.metadata: dict[str, Any] = dict(config.get("metadata") or {})
        self.experiment_id: str = str(self.metadata.get("experiment_id", "default"))
        self.time_field: str = str(config.get("time_field", "global_time"))

        # Per-port (frame) state: buffered Arrow tables, tick count, file counter.
        self._buffers: dict[str, list] = {}
        self._tick_counts: dict[str, int] = {}
        self._file_idx: dict[str, int] = {}
        self._closed: bool = False

    # -- partition path ----------------------------------------------------
    def _partition_subpath(self) -> str:
        """Hive partition dirs (``key=value``) from ``metadata``, '' if none."""
        parts = [f"{k}={self.metadata.get(k, '')}" for k in self.partitioning_keys]
        return os.path.join(*parts) if parts else ""

    @staticmethod
    def _frame_name(port: str) -> str:
        """Clean dataset name from the wired port key (strips a `Datasets_` prefix)."""
        return port[len("Datasets_"):] if port.startswith("Datasets_") else port

    # -- emit --------------------------------------------------------------
    def update(self, state: dict[str, Any]) -> dict:
        """Append each DataFrame-valued input to its per-frame buffer; flush per batch."""
        import pandas as pd
        import pyarrow as pa

        t = state.get(self.time_field, state.get("time"))
        for port, val in state.items():
            if not isinstance(val, pd.DataFrame) or val.empty:
                continue
            df = val.copy()
            # Long-format identity columns first: experiment_id, time, element_id,
            # then the dataframe's own columns. element_id is the mesh index.
            df.insert(0, "element_id", df.index.to_numpy())
            df.insert(0, "time", t)
            df.insert(0, "experiment_id", self.experiment_id)
            table = pa.Table.from_pandas(df, preserve_index=False)

            name = self._frame_name(port)
            self._buffers.setdefault(name, []).append(table)
            self._tick_counts[name] = self._tick_counts.get(name, 0) + 1
            if self._tick_counts[name] >= self.batch_size:
                self._flush_frame(name)
        return {}

    # -- flush -------------------------------------------------------------
    def _flush_frame(self, name: str) -> None:
        """Concat the buffered Arrow tables for one frame and write a Parquet file."""
        tables = self._buffers.get(name)
        if not tables:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        combined = pa.concat_tables(tables, promote_options="default")
        out_dir = os.path.join(self.out_uri, self.experiment_id, "frames",
                               name, self._partition_subpath())
        os.makedirs(out_dir, exist_ok=True)
        idx = self._file_idx.get(name, 0)
        pq.write_table(combined, os.path.join(out_dir, f"{idx:06d}.pq"),
                       compression="zstd")
        self._file_idx[name] = idx + 1
        self._buffers[name] = []
        self._tick_counts[name] = 0

    def close(self, success: bool = True) -> None:
        """Flush every frame's trailing (< batch_size) buffer. Idempotent."""
        if self._closed:
            return
        for name in list(self._buffers):
            self._flush_frame(name)
        self._closed = True

    def __del__(self) -> None:  # defensive durability; interpreter-shutdown order is undefined
        try:
            self.close(success=False)
        except Exception:
            pass

    @staticmethod
    def flush_all_in_composite(composite: Any, success: bool = True) -> int:
        """Close every DataFrameParquetEmitter instance in a composite. Returns count.

        Mirrors ``ParquetEmitter.flush_all_in_composite``: the composite builds the
        emitter inside its step factory, so the run driver never holds the instance
        and can't call ``close()`` directly. Call after the run loop to make each
        frame's trailing batch durable.
        """
        closed = 0

        def _walk(node: Any) -> None:
            nonlocal closed
            if isinstance(node, dict):
                inst = node.get("instance")
                if isinstance(inst, DataFrameParquetEmitter) and not inst._closed:
                    inst.close(success=success)
                    closed += 1
                for v in node.values():
                    _walk(v)

        _walk(getattr(composite, "state", None) or {})
        return closed
