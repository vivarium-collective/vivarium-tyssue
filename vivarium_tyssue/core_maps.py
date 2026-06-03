"""Tyssue geometry lookup.

`vessel_geometry` (VesselGeometry, CylinderGeometryInit, RNRGeometry) ships only
in the custom tyssue fork this model was originally developed against. We import
each geometry submodule defensively and build the map from whatever classes the
installed tyssue actually provides, so the module imports cleanly on stock PyPI
tyssue too; composites naming a fork-only geometry fail only at run time (clear
KeyError) rather than breaking discovery for every composite.
"""
import importlib

_GEOM_SUBMODULES = [
    "base_geometry", "planar_geometry", "sheet_geometry", "vessel_geometry",
    "multisheetgeometry", "bulk_geometry", "cylinder_geometry",
]

_ns = {}
for _sub in _GEOM_SUBMODULES:
    try:
        _mod = importlib.import_module(f"tyssue.geometry.{_sub}")
    except Exception:  # noqa: BLE001 — fork-only submodule absent on stock tyssue
        continue
    for _k in dir(_mod):
        if not _k.startswith("_"):
            _ns[_k] = getattr(_mod, _k)

_WANTED = [
    "PlanarGeometry", "AnnularGeometry", "WeightedPerimeterPlanarGeometry", "SheetGeometry",
    "VesselGeometry", "CylinderGeometry", "ClosedSheetGeometry", "CylinderGeometryInit",
    "BulkGeometry", "RNRGeometry", "MonolayerGeometry", "ClosedMonolayerGeometry",
]
GEOMETRY_MAP = {name: _ns[name] for name in _WANTED if name in _ns}
