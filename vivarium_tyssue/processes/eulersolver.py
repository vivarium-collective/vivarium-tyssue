import logging
import os
import warnings
import inspect
import time

from pprint import pprint

from bigraph_schema import allocate_core
from bigraph_schema.schema import get_frame_schema
from process_bigraph import Process, Composite
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from vivarium_tyssue.maps import *
from vivarium_tyssue.core_maps import GEOMETRY_MAP
from vivarium_tyssue.processes.utils import (
    compute_geometry,
    geometry_supported,
    gradient_supported,
    has_vessel_effector,
    is_bulk_geometry,
    materialize_geometry,
    rust_bulk_geometry_update,
    rust_geometry_update,
    rust_kernels_available,
    rust_sheet_gradient,
)
from vivarium_tyssue.processes.kernels import (
    effector_covered,
    rust_model_gradient,
)
import numpy as np

from tyssue.behaviors.event_manager import EventManager
from tyssue.behaviors.sheet.basic_events import reconnect
from tyssue.core.history import History, HistoryHdf5
from tyssue.io.hdf5 import load_datasets
from tyssue import config

log = logging.getLogger(__name__)

maps = {
    "GEOMETRY_MAP": GEOMETRY_MAP,
    "FACTORY_MAP": FACTORY_MAP,
    "EFFECTORS_MAP": EFFECTORS_MAP,
    "TISSUE_MAP": TISSUE_MAP,
    "BEHAVIOR_MAP": BEHAVIOR_MAP,
}

def set_pos(eptm, geom, pos):
    """Updates the vertex position of the :class:`Epithelium` object.

    Assumes that pos is passed as a 1D array to be reshaped as (eptm.Nv, eptm.dim)
    """
    log.debug("set pos")
    eptm.vert_df.loc[eptm.active_verts, eptm.coords] = pos.reshape((-1, eptm.dim))
    geom.update_all(eptm)


class EulerSolver(Process):
    """Generalized Euler solver for Tyssue-based epithelial simulations
    """
    config_schema = {
        "name": "string", #name for epithelium object
        "eptm": "string", #saved tyssue epithelium file
        "tissue_type": "string", #key indicating the desired tissue type from TISSUE_MAP
        "parameters": "map[map]",
        "geom": "string", #key indicating the desired geometry class in GEOMETRY_MAP
        "effectors": "list[string]", #list of strings representing effectors from the EFFECTORS_MAP
        "ref_effector": "string", #string, representing the effector from the EFFECTORS_MAP
        "factory": "string", #key indicating the factory class to generate from the FACTORY_MAP
        "auto_reconnect": "boolean", # if True, will automatically perform reconnections
        "bounds": "map[float]", # bounds the displacement of the vertices at each time step
        "output_columns": "map[list[string]]", # dict containing lists of column names to emit for each dataframe
        "history_columns": "map[list[string]]", # per-df extra columns to RECORD in
                             # the History (keys vert_df/edge_df/face_df/cell_df).
                             # Absent/empty for a df -> record all columns (default).
                             # When a df is listed, record only the coords+topology
                             # minimum needed to rebuild the eptm PLUS the listed cols.
        "settings": "map",
        "maps": "map", #map of maps, leave empty if using default
        "backend": "string", # "python" (default) or "rust": route compute_gradient
                             # through the tyssue_kernels sheet_gradient kernel when
                             # the model is supported; falls back to python otherwise.
        "substeps": "integer", # >1 integrates this many native Euler steps of
                               # dt=interval/substeps per update(), materializing
                               # the DataFrames only once at the end (rust sheet
                               # models). Default 1 = one step, bit-identical.
        "max_displacement": "float", # >0 clamps the per-step |Δposition| of every
                               # vertex to this value (explicit-Euler safety net).
                               # Default 0 = off (unchanged behavior). Used by the
                               # gillespie crypt model, where a division / extrusion
                               # (remove_face) transient can momentarily produce a
                               # huge local gradient that would otherwise explode to
                               # NaN in one step; clamping lets it relax over a few
                               # steps instead.
        "record_history": "boolean", # default True: build a tyssue History and
                               # record every step so `.history` drives create_gif
                               # / to_archive. Set False for large rust runs to
                               # avoid the per-step full-copy RAM growth.
        "history_file": "string", # stream history to this HDF5 file (HistoryHdf5)
                               # instead of RAM; flat memory, reusable archive.
        "history_save_every": "integer", # with history_file, record every N-th step.
    }

    def initialize(self, config):
        self.maps = maps
        if self.config["maps"]:
            self.maps.update(self.config["maps"])
        self._set_pos = set_pos
        self.geom = self.maps["GEOMETRY_MAP"][config["geom"]]
        datasets = load_datasets(config["eptm"])
        self.tyssue_type = self.maps["TISSUE_MAP"][config["tissue_type"]]
        self.eptm = self.tyssue_type("epithelium", datasets)
        self.eptm.network_changed = False
        self.eptm.settings.update(config["settings"])
        self.geom.update_all(self.eptm)
        effectors = [self.maps["EFFECTORS_MAP"][effector] for effector in config["effectors"]]
        self.model = self.maps["FACTORY_MAP"][config["factory"]](effectors, self.maps["EFFECTORS_MAP"][config["ref_effector"]])
        if len(config["parameters"]) > 0:
            for dataframe, parameters in config["parameters"].items():
                df = getattr(self.eptm, dataframe)
                for parameter, value in parameters.items():
                    # A dict value sets the parameter per mesh segment (apical /
                    # basal / lateral / sagittal), e.g. line_tension:
                    #   {apical: 1.0, default: 0.0}  ->  high tension on apical
                    # edges only (a 3D-monolayer "purse-string"). Falls back to
                    # the literal value when the df has no 'segment' column (a
                    # flat Sheet) so scalar configs are unchanged.
                    if isinstance(value, dict):
                        if "segment" in df.columns:
                            default = value.get("default", 0.0)
                            df[parameter] = df["segment"].map(value).fillna(default)
                        else:
                            df[parameter] = value.get("default", 0.0)
                    else:
                        df[parameter] = value

        manager = EventManager()
        if self.config["auto_reconnect"]:
            if "reconnect" not in [n[0].__name__ for n in manager.next]:
                manager.append(reconnect)

        self.manager = manager
        if len(self.config["bounds"]) > 0:
            self.bounds = self.config["bounds"]
        else:
            self.bounds = None

        # Backend selection: route compute_gradient through the Rust kernel when
        # requested AND the model is one the kernel reproduces exactly. Otherwise
        # fall back to Python transparently (with a warning if rust was asked for).
        self._backend = (config.get("backend") or "python").lower()
        self._is_bound = config["factory"] == "model_factory_bound"
        self._with_vessel = has_vessel_effector(config["effectors"])
        self._rust_gradient = False
        self._rust_geometry = False
        self._topo = None  # cached (signature, srce, trgt, face, active_pos) arrays
        self._active_full = False  # every vertex active & in order -> fast pos I/O
        self._geom_stash = None  # geometry arrays from the last rust set_pos, fed
        # straight to the next gradient (skips re-reading them from pandas)
        self._geom_lean = False  # True when the hot loop wrote only observable
        # geometry (edge length + face frame) and the intermediate edge coordinate
        # blocks in edge_df are stale — materialised full on demand (Phase B).
        self._substeps = max(1, int(config.get("substeps") or 1))
        # Explicit-Euler safety clamp on per-step vertex displacement (0 = off).
        self._max_disp = float(config.get("max_displacement") or 0.0)
        # Native multi-substep integration (DataFrames materialized once per
        # update instead of per step) needs both rust halves and a plain sheet
        # model — the vessel term reads position-derived vertex columns we don't
        # refresh mid-loop, so vessel/python fall back to per-step materializing.
        self._bulk_geometry = is_bulk_geometry(self.geom.__name__)
        # Effector classes + factory name, kept for the compositional rust path.
        self._effectors = effectors
        self._factory_name = config["factory"]
        # Compositional per-effector rust gradient: engages whenever the rust
        # kernels are available. Each effector runs through a rust primitive if
        # one is registered (kernels.EFFECTOR_KERNELS), else its tyssue
        # ``.gradient`` — so any model runs on the rust backend, covered terms
        # accelerated, nothing regressing. The fused ``_rust_gradient`` path below
        # is a faster special case for the standard 3-effector sheet model.
        self._rust_model = False
        if self._backend == "rust":
            self._rust_geometry = geometry_supported(self.geom.__name__, self.eptm.dim)
            if gradient_supported(config["effectors"], config["factory"], self.eptm.dim):
                self._rust_gradient = True
                log.info("EulerSolver: using Rust backend (fused gradient%s)",
                         " + geometry" if self._rust_geometry else "")
            elif rust_kernels_available():
                self._rust_model = True
                covered = [e for e in config["effectors"] if effector_covered(e)]
                fell_back = [e for e in config["effectors"] if not effector_covered(e)]
                log.info(
                    "EulerSolver: using Rust backend (compositional gradient%s); "
                    "rust effectors=%s, python-fallback effectors=%s",
                    " + geometry" if self._rust_geometry else "", covered, fell_back,
                )
            else:
                warnings.warn(
                    "backend='rust' requested but the compiled tyssue_kernels module "
                    "is not importable — falling back to Python. Build it with "
                    "`maturin develop --release` in rust-kernels/ (see its README)."
                )
        self._native_substeps = (
            self._rust_gradient and self._rust_geometry and not self._with_vessel
        )
        if self._substeps > 1 and not self._native_substeps:
            warnings.warn(
                "substeps>1 needs the rust sheet backend (geometry+gradient, "
                "non-vessel); integrating one step per update instead."
            )
        # Normalize any StringDtype columns from parameter assignment (pandas 3.0).
        self._coerce_string_columns()

        # Build after parameters so configured columns are recorded. With
        # history_file, stream each snapshot to HDF5 (HistoryHdf5) so memory stays
        # flat over long/repeated runs instead of accumulating a copy per step.
        self._record_history = bool(config.get("record_history", False))
        history_file = config.get("history_file")
        save_every = config.get("history_save_every") or None  # 0/"" -> every step
        if not self._record_history:
            self.history = None
        elif history_file:
            # HistoryHdf5 appends (record opens the store in "a" mode); delete a
            # stale file first so a rerun holds one simulation, not several.
            parent = os.path.dirname(history_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if os.path.exists(history_file):
                os.remove(history_file)
            self.history = HistoryHdf5(
                self.eptm, hf5file=history_file, overwrite=True,
                save_every=save_every, dt=1.0 if save_every else None,
            )
            # HDF5 tables can't serialize object-dtype columns (tyssue's topology
            # bookkeeping: cell, *_o, unique_id_max); drop them — gifs don't need them.
            ds = self.eptm.datasets
            for el in list(self.history.columns):
                keep = [c for c in self.history.columns[el] if ds[el][c].dtype != object]
                self.history.columns[el] = keep
                self.history.dtypes[el] = ds[el][keep].dtypes
        else:
            self.history = History(self.eptm)
        self._apply_history_columns()

    def _apply_history_columns(self):
        """Trim the History's recorded columns per the ``history_columns`` config.

        Only dataframes named in the config are touched, so the default (record
        everything) is unchanged. A listed dataframe records the coords/topology
        minimum needed to rebuild the epithelium plus the listed columns, and
        nothing else. Config keys are ``vert_df``/``edge_df``/``face_df``/``cell_df``
        (mirroring ``output_columns``); History elements drop the ``_df`` suffix.
        """
        requested = self.config.get("history_columns") or {}
        if self.history is None or not requested:
            return
        ds = self.eptm.datasets
        has_cell = getattr(self.eptm, "cell_df", None) is not None
        minima = {
            "vert": list(self.eptm.coords),
            "edge": ["srce", "trgt", "face"] + (["cell"] if has_cell else []),
            "face": [],
            "cell": [],
        }
        is_hdf5 = isinstance(self.history, HistoryHdf5)
        for el in list(self.history.columns):
            listed = requested.get(f"{el}_df")
            if listed is None:
                continue  # not listed -> keep default recording for this df
            available = ds[el].columns
            keep = [c for c in dict.fromkeys(minima.get(el, []) + list(listed))
                    if c in available]
            if is_hdf5:
                # HDF5 can't serialize object-dtype columns (topology bookkeeping).
                keep = [c for c in keep if ds[el][c].dtype != object]
                self.history.dtypes[el] = ds[el][keep].dtypes
            else:
                seed = ds[el][keep].reset_index(drop=False)
                if "time" not in keep:
                    seed["time"] = 0
                self.history.datasets[el] = seed
            self.history.columns[el] = keep

    def _coerce_string_columns(self):
        """pandas 3.0 gives scalar-string column assignments (e.g. ``cell_type``)
        a ``StringDtype``, which bigraph-schema's ``get_frame_schema`` can't
        introspect ('StringDtype' object has no attribute 'fields'). Coerce any
        such columns back to plain object dtype on the live epithelium dataframes
        so both ``outputs()`` (schema) and emission work."""
        import pandas as pd
        for df_name in ("vert_df", "edge_df", "face_df", "cell_df"):
            df = getattr(self.eptm, df_name, None)
            if df is None or isinstance(df, dict):
                continue
            for col in df.columns:
                if isinstance(df[col].dtype, pd.StringDtype):
                    df[col] = df[col].astype(object)

    def output_dfs(self):
        output_columns = self.config.get("output_columns", {})
        # if output_columns is empty, just dump all dataframes as dicts
        output_dfs = {}
        if not output_columns:
            for df_name in ["vert_df", "edge_df", "face_df", "cell_df"]:
                if getattr(self.eptm, df_name) is not None:
                    output_dfs[df_name] = getattr(self.eptm, df_name)
                else:
                    output_dfs[df_name] = {}
        else:
            for df_name in ["vert_df", "edge_df", "face_df", "cell_df"]:
                df_present = getattr(self.eptm, df_name, None)
                if df_present is not None and len(df_present) > 0:
                    if df_name in output_columns.keys():
                        cols = output_columns.get(df_name)
                        if not "unique_id" in cols:
                            cols.append("unique_id")
                        df = getattr(self.eptm, df_name)
                        if cols:
                            df = df[cols]
                        output_dfs[df_name] = df
                    else:
                        output_dfs[df_name] = getattr(self.eptm, df_name)
                else:
                    output_dfs[df_name] = {}
        return output_dfs

    def initial_state(self):
        outputs = self.output_dfs()
        for df_name, df in outputs.items():
            if not isinstance(df, dict):
                outputs[df_name] = df.to_dict(orient="list")
        return {
            "datasets": outputs,
        }

    @property
    def current_pos(self):
        vd, coords = self.eptm.vert_df, self.eptm.coords
        if self._active_full:  # all verts active & in index order
            return vd[coords].values.ravel()
        return vd.loc[self.eptm.active_verts, coords].values.ravel()

    def _topo_arrays(self):
        """Cached topology lookups for the current mesh, rebuilt only when the
        signature (Nv, Ne, Nf) changes. Returns ``(srce, trgt, face, active_pos)``:
        the positional uint32 srce/trgt/face index arrays the kernels consume, plus
        ``active_pos`` — the positions of ``active_verts`` in ``vert_df`` order, so
        the gradient can be sliced to the active DOFs without rebuilding a lookup
        dict every step. ``active_verts`` is a stable epithelium attribute reset
        only on an index reset, which is exactly when our signature changes."""
        e = self.eptm
        sig = (e.Nv, e.Ne, e.Nf)
        if self._topo is None or self._topo[0] != sig:
            vmap = {v: i for i, v in enumerate(e.vert_df.index)}
            fmap = {v: i for i, v in enumerate(e.face_df.index)}
            srce = np.ascontiguousarray(e.edge_df["srce"].map(vmap).values, dtype=np.uint32)
            trgt = np.ascontiguousarray(e.edge_df["trgt"].map(vmap).values, dtype=np.uint32)
            face = np.ascontiguousarray(e.edge_df["face"].map(fmap).values, dtype=np.uint32)
            active = e.active_verts
            active_pos = np.fromiter((vmap[v] for v in active), dtype=np.intp, count=len(active))
            # When every vertex is active and in index order, position read/write
            # can use whole-column pandas access (~20× faster than label-based
            # ``.loc[active_verts]``, which pays index-alignment every step).
            self._active_full = len(active) == e.Nv and active.equals(e.vert_df.index)
            # bulk (Monolayer/Bulk) geometry also needs the per-edge cell index
            cell = None
            if self._bulk_geometry and "cell" in e.edge_df.columns:
                cmap = {c: i for i, c in enumerate(e.cell_df.index)}
                cell = np.ascontiguousarray(e.edge_df["cell"].map(cmap).values, dtype=np.uint32)
            self._topo = (sig, srce, trgt, face, active_pos, cell)
        return self._topo[1], self._topo[2], self._topo[3], self._topo[4]

    def set_pos(self, pos):
        """Updates the eptm vertices position, then refreshes geometry."""
        if not self._rust_geometry:
            return self._set_pos(self.eptm, self.geom, pos)
        eptm = self.eptm
        srce, trgt, face, _ = self._topo_arrays()  # also refreshes _active_full
        p = pos.reshape((-1, eptm.dim))
        if self._active_full:  # whole-column write, ~20× cheaper than .loc[active]
            eptm.vert_df[eptm.coords] = p
            # p already IS the full vertex-position array — hand it to the kernel
            # so it needn't re-read what we just wrote.
            kernel_pos = p
        else:
            eptm.vert_df.loc[eptm.active_verts, eptm.coords] = p
            kernel_pos = None  # inactive verts also feed edges; let the kernel read all
        if self._bulk_geometry:  # 3D volumetric: bulk kernel, python gradient reads DFs
            cell = self._topo[5]
            self._geom_stash = rust_bulk_geometry_update(
                eptm, srce, trgt, face, cell, pos=kernel_pos
            )
            self._geom_lean = False  # bulk gradient reads DFs → must stay full
        else:
            # Sheet path: the Rust gradient reads geometry from the stash, so the
            # hot loop only needs the observable columns in pandas each step; the
            # intermediate coordinate blocks are materialised on demand (Phase B).
            self._geom_stash = rust_geometry_update(
                eptm, self.geom, srce, trgt, face, pos=kernel_pos, full=False
            )
            self._geom_lean = True

    def _integrate_native(self, interval, substeps):
        """Integrate ``substeps`` explicit-Euler steps of ``dt = interval/substeps``
        entirely in native arrays, touching the DataFrames only once at the end.

        Bit-identical to running ``substeps`` single-step updates at ``dt`` and
        sampling the last — the intermediate states simply aren't materialized.
        Sheet (non-vessel) rust models only (see ``_native_substeps``)."""
        eptm = self.eptm
        coords = eptm.coords
        srce, trgt, face, active_pos = self._topo_arrays()
        dt = interval / substeps
        vpos = np.ascontiguousarray(eptm.vert_df[coords].values, dtype=np.float64)
        visc = eptm.vert_df["viscosity"].values[active_pos][:, None]
        geom = self._geom_stash
        if geom is None:  # first update: seed geometry from the current frame
            old_len = eptm.edge_df["length"].values.copy()
            geom = compute_geometry(eptm, srce, trgt, face, vpos, old_len)
        topo = (srce, trgt, face)
        for _ in range(substeps):
            grad = rust_sheet_gradient(eptm, self._is_bound, False, topo=topo, geom=geom)
            dot_r = -grad[active_pos] / visc
            if self.bounds is not None:
                dot_r = np.clip(dot_r, *self.bounds)
            vpos[active_pos] += dot_r * dt
            geom = compute_geometry(eptm, srce, trgt, face, vpos, geom["length"])
        # Single materialization at the interface (positions + geometry frames).
        if self._active_full:
            eptm.vert_df[coords] = vpos
        else:
            eptm.vert_df.loc[eptm.active_verts, coords] = vpos[active_pos]
        materialize_geometry(eptm, geom, which=("edge", "face"), full=False)
        self._geom_stash = geom
        self._geom_lean = True

    def to_dataframes(self, which=("edge", "face"), full=True):
        """Materialize the current native geometry into the epithelium DataFrames
        and return the epithelium. Public "convert only where you need it" hook:
        after a native run the frames already hold positions + the observable
        geometry (length/area/perimeter); call this to also refresh the full
        intermediate edge coordinate blocks (``full=True``, the default) — e.g.
        before inspecting raw geometry. No-op on the python backend (frames are
        always current)."""
        if self._geom_stash is not None and (full or self._geom_lean):
            materialize_geometry(self.eptm, self._geom_stash, which=which, full=full)
            if full:
                self._geom_lean = False
        return self.eptm

    def record(self, t):
        """Snapshot the current epithelium into the tyssue History (no-op when
        history recording is disabled). On the rust/lean hot path the intermediate
        edge-coordinate blocks are left native, so materialize the full geometry
        first, otherwise History would archive stale sub-coordinates."""
        if self.history is None:
            return
        if self._geom_lean:
            self.to_dataframes(full=True)
        self.history.record(time_stamp=t)

    def ode_func(self):
        """Computes the models' gradient.
        Returns
        -------
        dot_r : 1D np.ndarray of shape (self.eptm.Nv * self.eptm.dim, )
        """
        active = self.eptm.active_verts
        if self._rust_gradient:
            srce, trgt, face, active_pos = self._topo_arrays()
            grad = rust_sheet_gradient(
                self.eptm, self._is_bound, self._with_vessel,
                topo=(srce, trgt, face), geom=self._geom_stash,
            )  # (Nv, dim)
            grad_U = grad[active_pos]
        elif self._rust_model:
            # Compositional rust path: assemble the gradient effector-by-effector
            # (covered terms via rust primitives, the rest via their tyssue
            # ``.gradient``). Pass the native stash only when geometry is lean
            # (its intermediate edge blocks aren't in pandas); otherwise the
            # freshly-materialized frames are current.
            srce, trgt, face, active_pos = self._topo_arrays()
            geom = self._geom_stash if self._geom_lean else None
            grad = rust_model_gradient(
                self.eptm, self._effectors, self._factory_name,
                topo=(srce, trgt, face), geom=geom,
            )  # (Nv, dim)
            grad_U = grad[active_pos]
        else:
            grad_U = self.model.compute_gradient(self.eptm).loc[active].values
        return (
                -grad_U
                / self.eptm.vert_df.loc[active, "viscosity"].values[:, None]
        ).ravel()

    def inputs(self):
        return {
            "behaviors": "list[node]",
            "global_time": "float",
        }

    def _frame_schema_cached(self, name, df):
        """``get_frame_schema`` re-introspects every column each call, and the
        engine invokes ``outputs()`` on every process update — but the schema
        only changes when a df's columns or dtypes change (division adds columns,
        StringDtype coercion changes dtypes). Cache on a cheap (name, dtype)
        signature and recompute only when it actually changes."""
        sig = tuple(zip(map(str, df.columns), map(str, df.dtypes)))
        cache = getattr(self, "_schema_cache", None)
        if cache is None:
            cache = self._schema_cache = {}
        hit = cache.get(name)
        if hit is None or hit[0] != sig:
            cache[name] = (sig, get_frame_schema(df))
        return cache[name][1]

    def outputs(self):
        datasets = {
            "_type": "tyssue_data",
            "vert_df": {"_columns": self._frame_schema_cached("vert_df", self.eptm.vert_df)},
            "edge_df": {"_columns": self._frame_schema_cached("edge_df", self.eptm.edge_df)},
            "face_df": {"_columns": self._frame_schema_cached("face_df", self.eptm.face_df)},
        }
        # cell_df is None for a 2D Sheet but a (non-empty) DataFrame for a 3D
        # Monolayer — test explicitly, since `if df:` is ambiguous on a frame.
        cell_df = getattr(self.eptm, "cell_df", None)
        if cell_df is not None and len(cell_df) > 0:
            datasets["cell_df"] = {"_columns": self._frame_schema_cached("cell_df", cell_df)}
        else:
            datasets["cell_df"] = {}
        return {
            "datasets": datasets,
            "network_changed": "boolean",
            "behaviors_update": "list",
        }

    def update(self, inputs, interval):
        log.debug("EulerSolver step t=%s", inputs["global_time"])
        behavior_update = []
        if len(inputs["behaviors"]) > 0:
            for kwargs in inputs["behaviors"]:
                func = self.maps["BEHAVIOR_MAP"][kwargs["func"]]
                del kwargs["func"]
                arg_names = [name for name, param in inspect.signature(func).parameters.items()]
                if "geom" in arg_names:
                    kwargs["geom"] = self.maps["GEOMETRY_MAP"][kwargs["geom"]]
                    self.manager.append(func, **kwargs)
                else:
                    self.manager.append(func, **kwargs)
            behavior_update = {"_remove": "all"}

        if self._native_substeps and self._substeps > 1:
            # Many native Euler steps per update; DataFrames materialized once.
            self._integrate_native(interval, self._substeps)
        else:
            pos = self.current_pos
            dot_r = self.ode_func()
            if self.bounds is not None:
                dot_r = np.clip(dot_r, *self.bounds)
            delta = dot_r * interval
            if self._max_disp:
                # Bound any single-step motion (division / extrusion transient)
                # so a momentary huge gradient relaxes over several steps instead
                # of exploding to NaN. Component-wise clamp keeps it cheap.
                np.clip(delta, -self._max_disp, self._max_disp, out=delta)
            pos = pos + delta
            self.set_pos(pos)

        if self.manager is not None:
            # Behaviours may read the full edge geometry (normals, ucoords, …) that
            # the lean hot-loop left native; refresh the frames only when behaviours
            # actually run this step (the manager is repopulated from inputs each
            # update, so no behaviours in → nothing to execute). Pure integration
            # steps skip it and keep the lean fast path.
            if inputs["behaviors"] and self._geom_lean:
                self.to_dataframes(full=True)
            self.manager.execute(self.eptm)
            # Only rebuild geometry if a behavior actually changed the mesh.
            # Every topology-changing path (division, T1/T2, reconnect) sets
            # network_changed (both tyssue's topology ops and our behaviors do);
            # parameter-only behaviors (e.g. line-tension setters) don't move
            # vertices, so the geometry from set_pos above is still current and
            # a second full update_all is pure waste (~half of update_all cost).
            if self.eptm.network_changed:
                # Topology changed: full python update_all rebuilds geometry AND
                # the boundary/opposite index for the new mesh; drop the cached
                # positional arrays so the next set_pos rebuilds them.
                self.geom.update_all(self.eptm)
                self._topo = None
                self._geom_stash = None  # stale after a topology change / update_all
                self._geom_lean = False  # update_all rewrote all geometry columns
            self.manager.update()

        if self.eptm.network_changed:
            network_changed = True
        else:
            network_changed = False

        self.eptm.network_changed = False

        # Behaviors (differentiation, division) may re-introduce StringDtype
        # cell_type columns under pandas 3.0; coerce before schema/emission.
        # Nothing else adds string columns mid-run (init already coerced), so
        # skip the all-column scan on plain integration steps — the common case.
        if inputs["behaviors"] or network_changed:
            self._coerce_string_columns()

        self.record(inputs["global_time"])

        dfs = self.output_dfs()

        return {
            "datasets": dfs,
            "network_changed": network_changed,
            "behaviors_update": behavior_update,
        }

if __name__ == "__main__":
    from vivarium_tyssue.data_types import register_types
    from vivarium_tyssue.processes import register_processes
    import pandas as pd
    from matplotlib import pyplot as plt
    # create the core object
    core = allocate_core()
    core.register_link("EulerSolver", EulerSolver)
    core = register_types(core)
    core = register_processes(core)
    # register data types



