#!/usr/bin/env python
"""Generate the HRA-initialized scale demos: build the meshes from Human
Reference Atlas data, save them as tyssue ``.hf5`` datasets, and write matching
``.composite.yaml`` files that run them on the rust ``EulerSolver``.

    .venv/bin/python scripts/gen_hra_demos.py

Produces two demos:

- ``hra_crypt_field``  — a 2D field of intestinal crypts of Lieberkühn, each an
  exact copy of the HRA 2D-FTU cell layout (real Absorptive / Goblet / Stem /
  Neuroendocrine / Tuft cells), tiled up to ~8k cells to show 2D scale.
- ``hra_colon_surface`` — a 3D epithelial sheet draped over the real HRA large-
  intestine reference organ surface (GLB), decimated to ~3k cells, labelled by
  the crypt's ASCT+B cell-type proportions.

Cell types are preserved per-cell in the ``.hf5`` (``face_df['cell_type']``);
the composites deliberately do NOT set a ``cell_type`` parameter so those real
labels survive into the viewer.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML
from tyssue.io.hdf5 import save_datasets

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vivarium_tyssue.hra import (  # noqa: E402
    asctb_cell_types, sheet_from_ftu, sheet_from_organ_glb,
)

DATASETS = ROOT / "workspace" / "datasets"
COMPOSITES = ROOT / "vivarium_tyssue" / "composites"


def composite_doc(name, description, eptm_rel, tissue_type="Sheet", interval=0.02,
                  line_tension=0.02, frozen=False):
    """A relaxation composite on the rust EulerSolver (no cell_type param, so
    the per-cell HRA types in the .hf5 are preserved).

    ``frozen=True`` zeroes the elastic moduli and line tension so the mesh does
    not move — a structural display of a raw real-data mesh (whose irregular
    cells are too stiff for stable explicit-Euler relaxation). The tissue is
    shown exactly as HRA initialized it.
    """
    ka = 0.0 if frozen else 1.0
    kp = 0.0 if frozen else 0.1
    lt = 0.0 if frozen else line_tension
    return {
        "name": name,
        "description": description,
        "tags": ["tissue", "mechanics", "hra"],
        "requires": {"processes": ["EulerSolver"], "types": ["tyssue_data", "behaviors"]},
        "emitters": [],
        "parameters": {
            "interval": {"type": "float", "default": interval,
                         "description": "Emit / update interval. Solver dt = interval / substeps."},
            "substeps": {"type": "int", "default": 1,
                         "description": "Native Euler steps per update (rust sheet models)."},
        },
        "state": {
            "Tyssue": {
                "_type": "process",
                "address": "local:EulerSolver",
                "config": {
                    "name": name,
                    "eptm": eptm_rel,
                    "tissue_type": tissue_type,
                    # NB: no cell_type / prefered_area / prefered_perimeter here —
                    # those are baked PER-CELL in the .hf5 (real HRA types + rest
                    # targets) and must survive. A small line_tension drives gentle
                    # bounded motion off that rest state.
                    "parameters": {
                        "face_df": {"area_elasticity": ka, "perimeter_elasticity": kp,
                                    "is_alive": 1.0},
                        "edge_df": {"line_tension": lt, "is_active": 1.0},
                        "vert_df": {"viscosity": 1.0, "is_alive": 1.0},
                    },
                    "geom": "SheetGeometry",
                    "effectors": ["LineTension", "FaceAreaElasticity", "PerimeterElasticity"],
                    "ref_effector": "FaceAreaElasticity",
                    "factory": "model_factory",
                    "settings": {"threshold_length": 0.03},
                    "auto_reconnect": False,  # keep topology static: stable + small viewer files
                    "bounds": {}, "output_columns": {}, "maps": {},
                    "backend": "rust", "substeps": "${substeps}",
                },
                "inputs": {"behaviors": ["Behaviors"], "global_time": ["global_time"]},
                "outputs": {"datasets": ["Datasets"], "network_changed": ["Network Changed"],
                            "behaviors_update": ["Behaviors"]},
                "interval": "${interval}",
            },
            "Network Changed": False,
            "Behaviors": {},
        },
    }


def write_composite(name, doc):
    yaml = YAML()
    yaml.default_flow_style = False
    path = COMPOSITES / f"{name}.composite.yaml"
    with path.open("w") as fh:
        yaml.dump(doc, fh)
    print(f"  wrote {path.relative_to(ROOT)}")


def main():
    DATASETS.mkdir(parents=True, exist_ok=True)
    print("Building HRA-initialized demos...")

    # --- 2D: field of real intestinal crypts ------------------------------- #
    print("2D crypt field (HRA 2D-FTU crypt of Lieberkühn, tiled):")
    # ~190 cells / tile after sanitize; tile to ~8k cells.
    sheet, meta = sheet_from_ftu("crypt of Lieberkuhn", tile=(7, 6))
    save_datasets(str(DATASETS / "hra_crypt_field.hf5"), sheet)
    print(f"  {sheet.face_df.shape[0]} cells, types={meta['type_names']}")
    write_composite("hra_crypt_field", composite_doc(
        "HRA crypt field (2D)",
        "A 2D epithelium built as a 7×6 field of intestinal crypts of Lieberkühn, "
        "each an exact copy of the Human Reference Atlas 2D-FTU cell layout "
        "(real Absorptive / Goblet / Stem / Neuroendocrine / Tuft cells). "
        "Colour by cell type to see the HRA cell identities; scales to ~8k cells.",
        "workspace/datasets/hra_crypt_field.hf5", interval=0.02, frozen=True))

    # --- 3D: epithelium on the real colon surface -------------------------- #
    print("3D colon surface (HRA large-intestine reference organ GLB):")
    comp = asctb_cell_types("crypt of Lieberkuhn")
    sheet3, meta3 = sheet_from_organ_glb("large intestine", keep=0.15, cell_types=comp)
    save_datasets(str(DATASETS / "hra_colon_surface.hf5"), sheet3)
    print(f"  {sheet3.face_df.shape[0]} cells on real anatomy, {meta3['source']}")
    write_composite("hra_colon_surface", composite_doc(
        "HRA colon surface (3D)",
        "A 3D epithelial sheet draped over the real Human Reference Atlas large-"
        "intestine reference organ surface (decimated to ~3k cells), with cells "
        "labelled by the crypt's ASCT+B cell-type proportions. Orbit to inspect "
        "the anatomy; this is a real organ shape, not a generated mesh.",
        "workspace/datasets/hra_colon_surface.hf5", interval=0.02))

    print("done.")


if __name__ == "__main__":
    main()
