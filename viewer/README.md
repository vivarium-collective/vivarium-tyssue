# vivarium-tyssue mesh viewer

An interactive 2D/3D viewer for tyssue vertex-model simulation runs — a zero-build
static three.js app in the same mould as the parsimony and pbg-cpm viewers.

Cells are drawn as polygons (fan-filled from each cell centroid) with an edge
wireframe; flat sheets render in an orthographic **2D** view, cylindrical/vessel
and other curved meshes in a **3D** orbit view (auto-detected per run). Colour cells
by area / perimeter / neighbour count / identity, scrub or play the time-series, and
hover any cell to read its measurements.

## Run it

```bash
# 1. export one or more composites to viewer/data/*.json (+ index.json)
.venv/bin/python scripts/export_viewer_data.py            # default showcase set
#    .venv/bin/python scripts/export_viewer_data.py --only anisotropic --frames 80

# 2. serve the static app (any static server works)
cd viewer && python3 -m http.server 8971
#    open http://localhost:8971
```

`viewer/data/` is gitignored — regenerate it with the export script.

## How it works

- **`index.html` + `viewer.js`** — the whole app. three.js is loaded from the
  `unpkg.com/three@0.160.0` importmap; no npm, no bundler, no server code.
- **`scripts/export_viewer_data.py`** — runs each composite, snapshots the
  epithelium mesh every emit frame, and writes a compact JSON time-series plus a
  `data/index.json` manifest that drives the run picker.

### Data contract

`data/index.json`:

```json
{ "models": [ { "file": "anisotropic.json", "slug": "anisotropic",
                "name": "Anisotropic elongation", "is3d": false,
                "n_cells": 206, "n_frames": 60 } ] }
```

Each `data/<slug>.json`:

```
{ name, blurb, is3d, n_cells, n_verts, bounds:[[x,y,z],[x,y,z]],
  face_fields:["area","perimeter","num_sides"],
  static_topology?: true,               // present when topology never changes
  tris?, edges?, face_of_tri?,          // hoisted here when static_topology
  frames: [ { t,
              verts:[x,y,z,...],         // Nv*3 vertex positions
              centroids:[x,y,z,...],     // Nf*3 cell centroids
              fields:{area:[...], perimeter:[...], num_sides:[...]},
              tris?, edges?, face_of_tri? // per-frame topology when it changes
            } ] }
```

`tris` index into the concatenated `[verts ; centroids]` position block — each
half-edge `(srce, trgt, face-centroid)` is one triangle, so a cell is a fan of
triangles around its centroid. `edges` are vertex-index pairs (the wireframe).
`face_of_tri[t]` is the cell id for triangle `t`, used for per-cell colouring and
hover picking. Topology is stored once at the model level when it never changes
(no T1 transition or division), else per frame.

## Adding a run

Add an entry to `SHOWCASE` in `scripts/export_viewer_data.py` (composite slug,
display name, blurb, emit interval, frame count) and re-run the exporter. Any
composite that runs on the `EulerSolver` works; runs that emit `edge_df`/`face_df`
geometry are all that's needed.
