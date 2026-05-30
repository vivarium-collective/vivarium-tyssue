"""build_core() — wraps process_bigraph.allocate_core().

Imports declared in workspace.yaml are auto-discovered by
allocate_core() once they're pip-installed in the workspace venv.
"""
from process_bigraph import allocate_core


def build_core():
    return allocate_core()
