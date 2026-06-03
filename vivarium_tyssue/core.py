"""build_core() — the workspace's process-bigraph core.

The vivarium-dashboard builds a workspace's core by inserting the workspace root
on sys.path and calling ``<package>.core.build_core()`` (see
``_build_workspace_core`` in vivarium_dashboard/server.py). Workspace-local
packages are deliberately *not* pip-discovered via importlib.metadata, so this is
where the workspace registers its own processes, schema types and visualizations
into the core's link/registry — that's what makes ``local:EulerSolver``,
``local:Gillespie``, ``local:TissueSheetGif`` … resolve when the dashboard runs a
composite.

Process registration is wrapped so the core still builds (types + visualizations
+ composite discovery) even if the heavy tyssue simulator stack is unavailable.
"""
from process_bigraph import allocate_core


def build_core():
    core = allocate_core()

    # Custom schema types (tyssue_data, behaviors, tyssue_dset, frame).
    from vivarium_tyssue.data_types import register_types
    core = register_types(core)

    # Simulator processes (EulerSolver, TestRegulations, StochasticLineTension,
    # CellJamming, ParameterGradient, AnisotropicTension, Gillespie). These import
    # tyssue; degrade gracefully if that stack isn't installed.
    try:
        from vivarium_tyssue.processes import register_processes
        core = register_processes(core)
    except Exception as exc:  # noqa: BLE001 — best-effort; surface but don't crash
        print(f"vivarium_tyssue.core: tyssue processes not registered ({type(exc).__name__}: {exc})")

    # Visualization Steps (TissueSheetGif, TissueCryptGif3D).
    try:
        from vivarium_tyssue import visualizations as _viz
        for _name in getattr(_viz, "__all__", []):
            _cls = getattr(_viz, _name)
            if _name not in core.link_registry:
                core.register_link(_name, _cls)
    except Exception as exc:  # noqa: BLE001
        print(f"vivarium_tyssue.core: visualizations not registered ({type(exc).__name__}: {exc})")

    return core
