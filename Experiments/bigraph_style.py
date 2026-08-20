"""Shared bigraph-diagram styling for the experiment analysis notebooks.

Every experiment in ``Experiments/`` is a `process_bigraph` composite: a handful of
**processes** (the EulerSolver plus whatever drives it) wired to shared **stores**
(``Tissue State``, ``Behaviors``, ``global_time``, ...). ``bigraph_viz.plot_bigraph``
draws that composite; this module draws the *same* graph with a fixed palette so
every figure in the manuscript reads alike:

* **processes** (boxes) — shades of peach / pink,
* **stores** (circles) — shades of light blue.

Usage from an analysis notebook (the sim script next to it owns the spec)::

    import sys; sys.path.insert(0, str(sim.REPO / "Experiments"))
    from bigraph_style import plot_experiment_bigraph

    plot_experiment_bigraph(sim.build_divisions_spec(mesh), core=core,
                            filename="divisions_bigraph", out_dir=sim.OUT_DIR)

The returned object renders inline in Jupyter (it is `bigraph_viz`'s
``ResponsiveGraph``), and, when ``filename`` is given, is also written to disk.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from bigraph_viz.visualize_types import ResponsiveGraph, get_graphviz_fig

# --- palette ---------------------------------------------------------------
# (fill, border) pairs. Borders are the same hue a few shades deeper, so the
# nodes stay legible against a white page at figure resolution.

# Processes: peach and pink, alternating so neighbouring processes stay apart.
PROCESS_SHADES = [
    ("#FFD9BE", "#E0A277"),   # peach
    ("#F9A8C0", "#D4799A"),   # pink
    ("#FFC0A5", "#DD8E6E"),   # deep peach
    ("#F4B9D2", "#CE86A8"),   # light rose
    ("#FFB2A8", "#D97C74"),   # coral
    ("#E9A0BE", "#C4779A"),   # deep rose
]

# Stores: light blues.
STORE_SHADES = [
    ("#E3F1FB", "#93C1DE"),   # palest blue
    ("#CFE7F7", "#82B5D6"),
    ("#BBDCF2", "#71A9CD"),
    ("#A7D1EC", "#609DC4"),
]

FONT_COLOR = "#2B2B33"


def _unique_paths(nodes):
    """Node paths in first-seen order (graph_dict repeats wired-to nodes)."""
    seen, paths = set(), []
    for node in nodes:
        path = tuple(node["path"])
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _apply_palette(graph, graph_dict):
    """Re-declare every node with its palette fill/border.

    Graphviz merges repeated node statements, so restating a node here overrides
    only the colours ``get_graphviz_fig`` set, leaving shape / label / size alone.
    """
    for i, path in enumerate(_unique_paths(graph_dict.get("process_nodes", []))):
        fill, border = PROCESS_SHADES[i % len(PROCESS_SHADES)]
        graph.node(str(path), style="filled", fillcolor=fill, color=border,
                   penwidth="2", fontcolor=FONT_COLOR)

    # Stores are shaded by nesting depth (top-level stores palest, nested ones
    # deeper), cycling within a depth so sibling stores stay distinguishable.
    depth_counts: dict[int, int] = {}
    for path in _unique_paths(graph_dict.get("state_nodes", [])):
        depth = len(path) - 1
        index = depth_counts.get(depth, 0)
        depth_counts[depth] = index + 1
        fill, border = STORE_SHADES[(depth + index) % len(STORE_SHADES)]
        graph.node(str(path), style="filled", fillcolor=fill, color=border,
                   penwidth="2", fontcolor=FONT_COLOR)


def _stamp_png_dpi(path, dpi):
    """Record the render resolution in the PNG's ``pHYs`` chunk.

    Graphviz scales the drawing by its ``dpi`` attribute but writes no physical
    resolution, so the file reports "unknown dpi" next to the notebooks' 300-dpi
    matplotlib figures. Re-saving stamps it (pixels are untouched).
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.load()
            img.save(path, dpi=(float(dpi), float(dpi)))
    except Exception as exc:  # noqa: BLE001 — cosmetic metadata only
        print(f"note: could not stamp {dpi} dpi on {path} ({exc})")


def experiment_graph_dict(state, core, schema=None, **traversal_kwargs):
    """The ``graph_dict`` ``bigraph_viz`` builds for a composite spec."""
    schema = schema or {}
    try:
        compiled_schema, compiled_state, _ = core.realize(schema, state)
    except Exception:  # noqa: BLE001 — same fallback plot_bigraph uses
        compiled_schema, compiled_state = schema, state
    return core.call_method(
        "generate_graph_dict", compiled_schema, compiled_state, (),
        options=traversal_kwargs,
    )


def plot_experiment_bigraph(
        state,
        core,
        schema=None,
        filename=None,
        out_dir=None,
        file_format="png",
        dpi="300",          # publication resolution, like the notebooks' FIG_DPI
        show_unwired_ports=False,
        **kwargs,
):
    """Draw a composite spec as a peach/pink-process, light-blue-store bigraph.

    Same call surface as ``bigraph_viz.plot_bigraph`` (extra keywords go through to
    ``get_graphviz_fig`` / the graph traversal); ``core`` is the experiment's
    ``vivarium_tyssue.core.build_core()`` core, which knows the tyssue types.

    ``show_unwired_ports`` draws the ports a process *declares* but the spec never
    wires (e.g. ``TumorCoupling``'s per-species count outputs) as dangling arrows.
    Off by default: they carry no data in these runs and stretch the figure several
    times wider than the wiring it is meant to show.
    """
    graphviz_params = inspect.signature(get_graphviz_fig).parameters
    render_kwargs = {k: v for k, v in kwargs.items() if k in graphviz_params}
    traversal_kwargs = {k: v for k, v in kwargs.items() if k not in graphviz_params}
    render_kwargs.setdefault("dpi", dpi)

    graph_dict = experiment_graph_dict(state, core, schema=schema, **traversal_kwargs)
    if not show_unwired_ports:
        graph_dict["disconnected_input_edges"] = []
        graph_dict["disconnected_output_edges"] = []
    graph = get_graphviz_fig(graph_dict, **render_kwargs)
    _apply_palette(graph, graph_dict)

    if filename is not None:
        out_dir = Path(out_dir) if out_dir is not None else Path("bigraphs")
        out_dir.mkdir(parents=True, exist_ok=True)
        # cleanup=True: drop the intermediate DOT source next to the image.
        graph.render(filename=str(out_dir / filename), format=file_format, cleanup=True)
        written = out_dir / f"{filename}.{file_format}"
        if file_format == "png":
            _stamp_png_dpi(written, render_kwargs["dpi"])
        print(f"wrote {written}")

    return ResponsiveGraph(graph)
