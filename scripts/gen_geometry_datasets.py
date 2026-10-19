"""Generate the small tyssue meshes for the coupling-hierarchy geometry composites.

Writes two datasets into ``workspace/datasets/`` so the ``cross_section`` and
``lumenoid`` composites (which select the new ``CrossSectionGeometry`` /
``LumenoidGeometry`` by name) are reachable from the composite/dashboard layer:

  * ``cross_section.hf5`` — a 12-cell annular ring (tyssue's 2D cross-section),
    loaded by the composite as a plain ``Sheet``.
  * ``lumenoid.hf5``      — a small closed monolayer shell around a lumen,
    loaded as a ``Monolayer``.

Run:  python scripts/gen_geometry_datasets.py
"""
from pathlib import Path

from tyssue.generation import generate_ring, ellipsoid_sheet, extrude
from tyssue import Monolayer
from tyssue.io.hdf5 import save_datasets

OUT = Path(__file__).resolve().parent.parent / "workspace" / "datasets"


def gen_cross_section(path: Path) -> None:
    ring = generate_ring(Nf=12, R_in=4.0, R_out=6.0)
    save_datasets(str(path), ring)
    print(f"wrote {path} (annular ring, Nf={ring.Nf}, Nv={ring.Nv})")


def gen_lumenoid(path: Path) -> None:
    sheet = ellipsoid_sheet(a=5.0, b=5.0, c=5.0, n_zs=6)
    mono = Monolayer("lumenoid", extrude(sheet.datasets, method="normals", scale=0.75))
    save_datasets(str(path), mono)
    print(f"wrote {path} (closed monolayer, Nf={mono.Nf}, Nc={mono.Nc})")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gen_cross_section(OUT / "cross_section.hf5")
    gen_lumenoid(OUT / "lumenoid.hf5")
