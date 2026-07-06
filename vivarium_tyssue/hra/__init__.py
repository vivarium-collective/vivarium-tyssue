"""Initialize tyssue meshes from Human Reference Atlas (HRA) data.

Three entry points, one per HRA data modality:

- :func:`sheet_from_ftu`  — a 2D :class:`~tyssue.Sheet` from a cell-resolution
  2D Functional-Tissue-Unit illustration (e.g. the intestinal crypt of
  Lieberkühn), with each cell carrying its real HRA/CL cell type.
- :func:`sheet_from_organ_glb` — a :class:`~tyssue.Sheet` draped over a real
  HRA 3D reference-organ surface mesh (GLB), decimated to a target cell count.
- :func:`asctb_cell_types` — the ASCT+B cell-type roster (names + proportions)
  for an organ, used to label generated/organ tissues biologically.

All network access goes through the public HRA API/CDN (``humanatlas.io``) and
is cached on disk. See :mod:`vivarium_tyssue.hra.loaders`.
"""

from vivarium_tyssue.hra.loaders import (
    asctb_cell_types,
    ftu_catalog,
    reference_organs,
    sheet_from_ftu,
    sheet_from_organ_glb,
)

__all__ = [
    "sheet_from_ftu",
    "sheet_from_organ_glb",
    "asctb_cell_types",
    "ftu_catalog",
    "reference_organs",
]
