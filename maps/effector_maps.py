from tyssue.dynamics.effectors import *

EFFECTORS_MAP = {
    "LengthElasticity": LengthElasticity,
    "PerimeterElasticity": PerimeterElasticity,
    "FaceAreaElasticity": FaceAreaElasticity,
    "FaceVolumeElasticity": FaceVolumeElasticity,
    "CellAreaElasticity": CellAreaElasticity,
    "CellVolumeElasticity": CellVolumeElasticity,
    "LumenVolumeElasticity": LumenVolumeElasticity,
    "LineTension": LineTension,
    "FaceContractility": FaceContractility,
    "SurfaceTension": SurfaceTension,
    "LineViscosity": LineViscosity,
    "BorderElasticity": BorderElasticity,
    "LumenAreaElasticity": LumenAreaElasticity,
    "RadialTension": RadialTension,
    "BarrierElasticity": BarrierElasticity,
    "ChiralTorque": ChiralTorque,
    "SurfaceElasticity": SurfaceElasticity,
    "BoundaryElasticity": BoundaryElasticity,
}