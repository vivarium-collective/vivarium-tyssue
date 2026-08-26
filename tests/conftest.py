"""Skip the tyssue-backend tests when the tyssue sim library is unavailable.

The tyssue vertex-model backend resolves to the vivarium-collective `vivatyssue`
fork, whose build clones a demo repo over SSH — impossible on a vanilla GitHub
runner. It lives in the optional `sim` extra (CI installs only .[dev]); the tests
that need it skip here, with a notice, and run in the repo's Docker image.
"""
from __future__ import annotations
try:
    import tyssue  # noqa: F401
    _HAVE_BACKEND = True
except Exception:  # noqa: BLE001
    _HAVE_BACKEND = False
_BACKEND_TEST_MODULES = [
    "test_composites.py", "test_rust_kernels_equiv.py", "test_tumor_composites.py",
    "test_tumor_core.py", "test_tumor_coupling.py", "tests.py",
]
collect_ignore = [] if _HAVE_BACKEND else list(_BACKEND_TEST_MODULES)
def pytest_configure(config):
    if not _HAVE_BACKEND:
        config.issue_config_time_warning(
            __import__("pytest").PytestConfigWarning(
                "tyssue backend unavailable — skipping "
                f"{len(_BACKEND_TEST_MODULES)} backend test module(s); run them "
                "in the repo's Docker image. Backend-free checks still run."),
            stacklevel=1)
