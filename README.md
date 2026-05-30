# pbg-tyssue

A process-bigraph research workspace wrapping the [tyssue](https://github.com/DamCB/tyssue)
vertex-model simulator. Converted from
[`vivarium-tyssue`](https://github.com/vivarium-collective/vivarium-tyssue): the
scenarios that used to be assembled procedurally inside `tests/tests.py` are now
**declared composites**, and the `create_gif` / `create_gif_3d` rendering recipes
are now **Visualization Steps**. Launches in the vivarium-dashboard.

## Layout

```
pbg_tyssue/
├── core.py                 build_core() — registers processes, types, viz with the core
├── data_types.py           custom schema types (tyssue_data, behaviors, tyssue_dset, frame)
├── core_maps.py, maps/     geometry / effector / factory / behavior lookups (fork-resilient)
├── processes/              EulerSolver + behavior processes (Gillespie, regulations, …)
├── models/crypt_gillespie/ crypt cell-type model parameters
├── composites/             *.composite.yaml — the 7 declared scenarios   ← NEW
└── visualizations/         TissueSheetGif, TissueCryptGif3D (Visualization Steps)  ← NEW
datasets/                   test_cylinder.hf5, test_square.hf5, crypt_cylinder.hf5 mesh fixtures
scripts/gen_composites.py   regenerates composites/ from the canonical configs
```

## Composites

| Composite (`pbg_tyssue.composites.*`) | Processes added on the EulerSolver | Mesh / geometry | Runs on stock tyssue? |
|---|---|---|---|
| `base_solver`  | — (pure mechanical relaxation)        | cylinder · VesselGeometry | ✗ needs fork |
| `regulation`   | TestRegulations (periodic division)   | cylinder · VesselGeometry | ✗ needs fork |
| `stochastic`   | StochasticLineTension                 | square · SheetGeometry · `model_factory_bound` | ✗ needs fork |
| `jamming`      | StochasticLineTension + CellJamming   | square · SheetGeometry · `model_factory_bound` | ✗ needs fork |
| `gradient`     | StochasticLineTension + ParameterGradient (step) | square · SheetGeometry · `model_factory_bound` | ✗ needs fork |
| `anisotropic`  | AnisotropicTension (step)             | square · SheetGeometry · `model_factory` | ✅ **runs** |
| `gillespie`    | Gillespie (crypt cell-type dynamics)  | crypt cylinder · VesselGeometry | ✗ needs fork |

Each composite exposes an `interval` parameter (`${interval}`); the dashboard
injects an emitter at run time, so the specs carry no emitter (use `/pbg-emit`
to add one for a standalone run).

### The tyssue fork caveat

`vivarium-tyssue` was developed against a **custom tyssue fork** that adds
`VesselGeometry`, `model_factory_vessel` / `_cylinder` / `_bound`,
`VesselSurfaceElasticity`, `ActiveMigration`, etc. Stock PyPI `tyssue==1.1.0`
does not ship these, and the fork is not pinned anywhere in the original repo.
The `maps/` lookups are written to import cleanly on stock tyssue (omitting the
fork-only symbols), so **all 7 composites are declared, discoverable and
loadable**, but only `anisotropic` (which uses solely stock symbols) runs
end-to-end here. Point `tyssue` at the original fork to run the rest — no other
change needed.

## Visualizations

Two `pbg_superpowers.visualization.Visualization` Steps (auto-discovered into the
dashboard's Registry):

- **`TissueSheetGif`** — 2D animation of the sheet evolving. `coords: ["x","z"]`
  for the cylinder, `["x","y"]` for the flat sheet.
- **`TissueCryptGif3D`** — 3D crypt animation, cells colored by type.

Both follow render **Path C**: they read the study's `runs.db` directly (the run
must emit the `Datasets` store), reconstruct the tyssue `History`, and call
tyssue's `create_gif` / `create_gif_3d` (faithful), falling back to a
dependency-light matplotlib edge-mesh animation when the viewer stack is absent.

## Run it

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"        # add ".[dev,viz3d]" for the faithful 3D path
bash scripts/serve.sh             # launch the vivarium-dashboard
pytest tests/test_composites.py   # validate composites + the stock-tyssue run
```

In the dashboard: the **Composites** tab lists all 7; the **Registry** tab lists
the processes + the two Visualization Steps. Create a Study, add a baseline
composite, run it, and attach a visualization (`address: local:TissueSheetGif`).
