"""Tyssue model-factory lookup.

`model_factory_vessel`, `model_factory_cylinder` and `model_factory_bound` ship
only in the custom tyssue fork this model was originally developed against; stock
PyPI tyssue exposes just `model_factory`. We build the map from whatever the
installed tyssue actually provides so the module imports cleanly either way —
composites referencing a missing factory fail only at run time (clear KeyError),
rather than breaking process discovery for every composite.
"""
from tyssue.dynamics import factory as _factory

_WANTED = ["model_factory", "model_factory_vessel", "model_factory_cylinder", "model_factory_bound"]
FACTORY_MAP = {name: getattr(_factory, name) for name in _WANTED if hasattr(_factory, name)}
