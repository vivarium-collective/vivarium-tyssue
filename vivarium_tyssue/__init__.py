"""vivarium_tyssue — workspace Python package."""


def _ensure_top_level_metadata() -> None:
    """Self-heal the dist-info ``top_level.txt`` for composite discovery.

    pbg_superpowers.composite_discovery.discover_composites() (the dashboard's
    run-path composite resolver) finds a distribution's packages by reading
    ``top_level.txt`` from its dist-info. The hatchling/uv wheel build for this
    workspace does NOT emit ``top_level.txt``, so discovery skips our composites
    and "Run baseline" fails with "composite not found". setuptools writes it;
    hatchling doesn't — and it's regenerated (without the file) on every
    ``uv sync``. Write it on import (the dashboard always imports this package
    before running) so the workspace's composites stay discoverable. Best-effort:
    never raise.
    """
    try:
        import importlib.metadata as _m
        from pathlib import Path

        dist = _m.distribution("vivarium-tyssue")
        # dist._path points at the .dist-info dir for file-based installs.
        info_dir = getattr(dist, "_path", None)
        if info_dir is None:
            return
        top_level = Path(info_dir) / "top_level.txt"
        if not top_level.is_file():
            top_level.write_text("vivarium_tyssue\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 — discovery self-heal must never break import
        pass


_ensure_top_level_metadata()
