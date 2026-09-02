#!/usr/bin/env python
"""Generate ``crypt_cylinder_stable.hf5`` — the crypt cylinder labelled with the
*stationary* cell-type distribution instead of the naive initial one.

    conda run -n vivarium-tyssue python scripts/gen_crypt_stable.py

The geometry is copied verbatim from ``crypt_cylinder.hf5``; only
``face_df['cell_type']`` is reassigned, using
:func:`~vivarium_tyssue.models.crypt_gillespie.crypt_params.stable_cell_types` along
the crypt axis ``z``. That function (and its ``stable_spatial_prob``) carries
parameters fitted to the composition the Gillespie crypt settles on (sc 0.039 / pc 0.116 / ent 0.638 / gc 0.208, stem band confined to
the bottom ~4% and the ent/gc ramp starting at ~9% of the crypt height), so the mesh
this writes *starts* at the stationary state rather than taking ~26 time units to
relax into it.

This mesh is NOT what the ``discrete_events`` experiments use: ``gillespie`` starts
from the stock ``crypt_cylinder.hf5`` and ``gillespie_restart`` from the checkpoint
``gillespie`` ends on. It is kept as a ready-made stationary-composition start for
anyone who wants one — e.g. to skip the compositional transient without paying for a
settling run. Note that starting at the stationary *composition* does not remove the
dynamical transient; that came from the commitment queue and is addressed by
``GILL_GROWTH_RATE`` instead.
"""
import sys
from pathlib import Path

import numpy as np
from tyssue.io.hdf5 import load_datasets, save_datasets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tyssue import Sheet  # noqa: E402
from vivarium_tyssue.models.crypt_gillespie.crypt_params import (  # noqa: E402
    stable_cell_types,
)

DATASETS = ROOT / "workspace" / "datasets"
SRC = DATASETS / "crypt_cylinder.hf5"
DST = DATASETS / "crypt_cylinder_stable.hf5"
AXIS = "z"
SEED = 20260715          # matches Experiments/discrete_events SEED

# Composition the Gillespie crypt settles on, measured over the last 10% of frames
# of a tf=72 run (150 frames, ~715 cells each). Printed alongside the result so a
# regeneration that drifts is obvious.
TARGET = {"sc": 0.039, "pc": 0.116, "ent": 0.638, "gc": 0.208}


def main():
    if not SRC.exists():
        raise SystemExit(f"source mesh not found: {SRC}")
    src = load_datasets(str(SRC))
    # Sheet() mutates the very DataFrames load_datasets returned (it compacts
    # unique_id to 0..N-1 in place), so snapshot the source ids first.
    src_uids = {k: src[k]["unique_id"].to_numpy().copy() for k in ("vert", "edge", "face")}
    sheet = Sheet("crypt", src)
    before = sheet.face_df["cell_type"].value_counts(normalize=True)

    sheet.face_df["cell_type"] = stable_cell_types(sheet, AXIS, random_seed=SEED)
    after = sheet.face_df["cell_type"].value_counts(normalize=True)

    print(f"{SRC.name}: {len(sheet.face_df)} cells, relabelled along {AXIS!r} "
          f"(seed {SEED})")
    print(f"  {'type':>5}  {'was':>7} {'now':>7} {'target':>7}")
    for t in ("sc", "pc", "ent", "gc"):
        print(f"  {t:>5}  {before.get(t, 0.0):7.3f} {after.get(t, 0.0):7.3f} "
              f"{TARGET[t]:7.3f}")

    z = sheet.face_df[AXIS].to_numpy(float)
    zn = (z - z.min()) / (z.max() - z.min())
    ct = sheet.face_df["cell_type"].to_numpy()
    print(f"  normalised {AXIS} by type (mean, p10, p90):")
    for t in ("sc", "pc", "ent", "gc"):
        zz = zn[ct == t]
        if len(zz):
            print(f"    {t:>5} n={len(zz):3d}  {zz.mean():.3f}  "
                  f"{np.percentile(zz, 10):.3f}  {np.percentile(zz, 90):.3f}")

    # Building the Sheet compacts unique_id to 0..N-1; put the source's ids back so
    # the written mesh differs from crypt_cylinder.hf5 in cell_type and nothing else.
    for elem, df in (("vert", sheet.vert_df), ("edge", sheet.edge_df),
                     ("face", sheet.face_df)):
        df["unique_id"] = src_uids[elem]

    if DST.exists():
        DST.unlink()
    save_datasets(str(DST), sheet)
    print(f"wrote {DST}")


if __name__ == "__main__":
    main()
