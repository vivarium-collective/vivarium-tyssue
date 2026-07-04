import logging
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
    SUPPORTED_EFFECTORS,
    compute_geometry,
    geometry_supported,
    gradient_supported,
    has_vessel_effector,
    is_bulk_geometry,
    materialize_geometry,
    rust_bulk_geometry_update,
    rust_geometry_update,
    rust_sheet_gradient,
)
import numpy as np

from tyssue.behaviors.event_manager import EventManager
from tyssue.behaviors.sheet.basic_events import reconnect
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
        "settings": "map",
        "maps": "map", #map of maps, leave empty if using default
        "backend": "string", # "python" (default) or "rust": route compute_gradient
                             # through the tyssue_kernels sheet_gradient kernel when
                             # the model is supported; falls back to python otherwise.
        "substeps": "integer", # >1 integrates this many native Euler steps of
                               # dt=interval/substeps per update(), materializing
                               # the DataFrames only once at the end (rust sheet
                               # models). Default 1 = one step, bit-identical.
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
        self._substeps = max(1, int(config.get("substeps") or 1))
        # Native multi-substep integration (DataFrames materialized once per
        # update instead of per step) needs both rust halves and a plain sheet
        # model — the vessel term reads position-derived vertex columns we don't
        # refresh mid-loop, so vessel/python fall back to per-step materializing.
        self._bulk_geometry = is_bulk_geometry(self.geom.__name__)
        if self._backend == "rust":
            self._rust_geometry = geometry_supported(self.geom.__name__, self.eptm.dim)
            if gradient_supported(config["effectors"], config["factory"], self.eptm.dim):
                self._rust_gradient = True
                log.info("EulerSolver: using Rust backend (gradient%s)",
                         " + geometry" if self._rust_geometry else "")
            else:
                warnings.warn(
                    "backend='rust' requested but this model is unsupported by the "
                    "gradient kernel (needs the 3 standard sheet effectors "
                    f"{sorted(SUPPORTED_EFFECTORS)}, a supported factory, a 3D mesh, "
                    "and the compiled tyssue_kernels module) — falling back to Python."
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
        else:
            self._geom_stash = rust_geometry_update(
                eptm, self.geom, srce, trgt, face, pos=kernel_pos
            )

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
        materialize_geometry(eptm, geom, which=("edge", "face"))
        self._geom_stash = geom

    def to_dataframes(self, which=("edge", "face")):
        """Materialize the current native geometry into the epithelium DataFrames
        and return the epithelium. Public "convert only where you need it" hook:
        after a native run the frames hold positions + the last geometry already,
        but call this any time to force a fresh conversion (e.g. for inspection).
        No-op geometry write on the python backend (frames are always current)."""
        if self._geom_stash is not None:
            materialize_geometry(self.eptm, self._geom_stash, which=which)
        return self.eptm

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
            pos = pos + dot_r * interval
            self.set_pos(pos)

        if self.manager is not None:
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



