from dataclasses import dataclass, field
from bigraph_schema.schema import Node, Map, List
from bigraph_schema.methods import apply
from vivarium_tyssue.processes import register_processes

@dataclass(kw_only=True)
class TyssueDset(List):
    _element: Map = field(default_factory=Map)

def tyssue_dset_update(schema, current, update, path):
    if not current:
        return [i for i in update]
    else:
        _current = {d["unique_id"]: d for d in current}
        _update = {d["unique_id"]: d for d in update}
        final = []
        for uid, upd in _update.items():
            if uid in _current:
                merged = {**_current[uid], **upd}
            else:
                merged = upd
            final.append(merged)
        return final

@apply.dispatch
def apply(schema: TyssueDset, current, update, path):
    return tyssue_dset_update(schema, current, update, path), []

@dataclass(kw_only=True)
class Behaviors(Map):
    _value: Map = field(default_factory=Map)

def behaviors_update(schema, current, update, path):
    _update = current.copy()
    if "_remove" in update.keys():
        if len(update["_remove"]) > 0:
            for i in update["_remove"]:
                if i in _update:
                    del _update[i]
        del update["_remove"]

    if len(update) > 0:
        # _update = _update.update(update)
        return {behavior:args for behavior, args in update.items()}
    else:
        return {}

@apply.dispatch
def apply(schema: Behaviors, current, update, path):
    return behaviors_update(schema, current, update, path), []

def register_types(core):
    core.register_type("tyssue_dset", TyssueDset)
    core.register_type("behaviors", Behaviors)
    return register_processes(core)