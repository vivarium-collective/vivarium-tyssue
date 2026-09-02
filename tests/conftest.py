"""Skip the tyssue-backend tests when the tyssue sim library is unavailable.

The tyssue vertex-model backend resolves to the vivarium-collective `vivatyssue`
fork, whose build clones a demo repo over SSH — impossible on a vanilla GitHub
runner. It lives in the optional `sim` extra (CI installs only .[dev]); the tests
that need it skip here, with a notice, and run in the repo's Docker image.
"""
from __future__ import annotations

import pathlib
import re
try:
    import tyssue  # noqa: F401
    _HAVE_BACKEND = True
except Exception:  # noqa: BLE001
    _HAVE_BACKEND = False
# Modules that reach the backend only indirectly (they import vivarium_tyssue, which
# imports tyssue) — a source scan cannot see those, so they stay listed by hand.
_BACKEND_TEST_MODULES = [
    "test_composites.py", "test_rust_kernels_equiv.py", "test_tumor_composites.py",
    "test_tumor_core.py", "test_tumor_coupling.py", "tests.py",
]


def _imports_tyssue(path: pathlib.Path) -> bool:
    """True if the module imports tyssue directly, at any indentation."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:  # noqa: BLE001 — unreadable file is not our problem here
        return False
    return re.search(r"^\s*(?:import\s+tyssue|from\s+tyssue[.\s])", text, re.M) is not None


# Union the hand-maintained list with a source scan. The list alone goes stale every
# time a test module is added — which is exactly how CI broke: five modules importing
# tyssue (test_unique_ids, test_eulersolver_history_columns, and the three added with
# the differential-adhesion / directional-tension experiments) were never added to it,
# so a `.[dev]` install failed at collection with ModuleNotFoundError instead of
# skipping.
_TESTS_DIR = pathlib.Path(__file__).parent
_SKIP = sorted(set(_BACKEND_TEST_MODULES) | {
    p.name for p in _TESTS_DIR.glob("test*.py") if _imports_tyssue(p)
})

collect_ignore = [] if _HAVE_BACKEND else list(_SKIP)


def pytest_configure(config):
    if not _HAVE_BACKEND:
        config.issue_config_time_warning(
            __import__("pytest").PytestConfigWarning(
                "tyssue backend unavailable — skipping "
                f"{len(_SKIP)} backend test module(s); run them "
                "in the repo's Docker image. Backend-free checks still run."),
            stacklevel=1)
