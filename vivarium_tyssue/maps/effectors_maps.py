"""Tyssue effector lookup.

`SurfaceElasticity`, `VesselSurfaceElasticity` and `ActiveMigration` ship only in
the custom tyssue fork this model was originally developed against. We build the
map from whatever the installed tyssue actually provides so the module imports
cleanly on stock PyPI tyssue too; composites listing a fork-only effector fail
only at run time (clear KeyError) rather than breaking discovery.
"""
from tyssue.dynamics import effectors as _eff

_WANTED = [
    "LengthElasticity", "PerimeterElasticity", "FaceAreaElasticity", "FaceVolumeElasticity",
    "CellAreaElasticity", "CellVolumeElasticity", "LumenVolumeElasticity", "LineTension",
    "FaceContractility", "SurfaceTension", "LineViscosity", "BorderElasticity",
    "LumenAreaElasticity", "RadialTension", "BarrierElasticity", "ChiralTorque",
    "SurfaceElasticity", "VesselSurfaceElasticity", "ActiveMigration",
]
EFFECTORS_MAP = {name: getattr(_eff, name) for name in _WANTED if hasattr(_eff, name)}
