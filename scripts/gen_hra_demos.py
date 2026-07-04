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
                  line_tension=0.02, max_displacement=0.1, frozen=False):
    """A relaxation composite on the rust EulerSolver (no cell_type param, so
    the per-cell HRA types in the .hf5 are preserved).

    ``max_displacement`` clamps each vertex's per-step motion — a safety net that
    lets the irregular real-data mesh relax stably under explicit Euler (a stray
    stiff cell can't blow the whole mesh up in one step).

    ``frozen=True`` zeroes the moduli so the mesh doesn't move — a faithful
    structural display of the raw real-data cells (relaxing the irregular real
    geometry would distort the true layout).
    """
    ka, kp, lt = (0.0, 0.0, 0.0) if frozen else (1.0, 0.1, line_tension)
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
                    "max_displacement": max_displacement,
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

    # --- 2D: one faithful intestinal crypt --------------------------------- #
    # A single crypt of Lieberkühn straight from the HRA 2D-FTU illustration —
    # the real cell layout and its zonation (stem + neuroendocrine at the narrow
    # base, absorptive/goblet up the column, tuft near the flared villus top).
    # Shown frozen: this is a faithful structural display of the real HRA cells,
    # not a mechanics run (relaxing the raw real geometry would distort it).
    print("2D intestinal crypt (HRA 2D-FTU crypt of Lieberkühn, faithful):")
    sheet, meta = sheet_from_ftu("crypt of Lieberkuhn", tile=(1, 1))
    save_datasets(str(DATASETS / "hra_crypt_field.hf5"), sheet)
    print(f"  {sheet.face_df.shape[0]} cells, types={meta['type_names']}")
    write_composite("hra_crypt_field", composite_doc(
        "HRA intestinal crypt (2D)",
        "A single intestinal crypt of Lieberkühn, initialized straight from the "
        "Human Reference Atlas 2D-FTU illustration — the real cell layout coloured "
        "by real HRA cell type. Note the zonation: stem + neuroendocrine cells at "
        "the narrow base, absorptive and goblet cells up the column, tuft cells "
        "near the flared villus top.",
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
