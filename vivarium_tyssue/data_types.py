from dataclasses import dataclass, field
import pandas as pd

from bigraph_schema import allocate_core
from bigraph_schema.schema import Node, Map, List, Frame
from bigraph_schema.methods import apply, reconcile
from bigraph_schema.methods.events import get_reconcile_sink

from vivarium_tyssue.behaviors.behaviors import _add_by_uid

#====================
#TyssueData Dataclass
#====================
@dataclass(kw_only=True)
class TyssueData(Node):
    vert_df: Frame = field(default_factory=Frame)
    edge_df: Frame = field(default_factory=Frame)
    face_df: Frame = field(default_factory=Frame)
    cell_df: Frame = field(default_factory=Frame)

# A frame in a TyssueData update is either a whole frame, which replaces the
# stored one (the EulerSolver's snapshot), or a delta:
#     {"_add_by_uid": [{"column": c, "component": comp, "delta": {uid: d}}, ...]}
# which adds each delta to the stored frame by unique_id, exactly as the
# update_parameter behavior does to the solver's mesh (see tissue_deltas). Within
# a tick the deltas land on top of that tick's replacement frame, whatever order
# the processes ran in, so a regulator's deltas are never wiped by the solver.
DELTAS = "_add_by_uid"
FRAME = "_frame"


def tissue_deltas(behaviors):
    """The TyssueData delta update mirroring the ``update_parameter`` nodes in
    ``behaviors``, so a regulator can write the parameter changes it sends the
    solver to the Tissue State store as well."""
    update = {}
    for node in behaviors:
        if node.get("func") != "update_parameter" or not node.get("parameter_update"):
            continue
        df = node["dataframe"].removesuffix("_df") + "_df"
        update.setdefault(df, {DELTAS: []})[DELTAS].append({
            "column": node["parameter"],
            "component": node.get("component"),
            "delta": node["parameter_update"],
        })
    return update


def _is_delta(value):
    return isinstance(value, dict) and DELTAS in value


@reconcile.dispatch
def reconcile(schema: TyssueData, updates: list):
    sink = get_reconcile_sink()
    parent = sink.path_stack if sink is not None else ()
    result = {}
    for update in updates:
        if not isinstance(update, dict):
            continue
        for key, value in update.items():
            if value is None:
                continue
            if key not in result and sink is not None:
                sink.paths.append(parent + (key,))
            current = result.get(key)
            if _is_delta(value):
                if not _is_delta(current):
                    current = {DELTAS: []} if current is None else {FRAME: current, DELTAS: []}
                    result[key] = current
                current[DELTAS].extend(value[DELTAS])
                if FRAME in value:
                    current[FRAME] = value[FRAME]
            elif _is_delta(current):
                current[FRAME] = value
            else:
                result[key] = value
    return result or None


def _apply_deltas(frame, entries, owned):
    if not isinstance(frame, pd.DataFrame):
        if not frame:  # no rows to add to (e.g. cell_df of a 2D sheet)
            return frame
        frame, owned = pd.DataFrame(frame), True
    if not owned:
        frame = frame.copy()  # readers may still hold the stored frame
    for entry in entries:
        _add_by_uid(frame, entry["column"], entry["delta"], entry.get("component"))
    return frame


@apply.dispatch
def apply(schema: TyssueData, current, update, path):
    if not isinstance(update, dict):
        return (current if update is None else update), []
    result = dict(current) if isinstance(current, dict) else {}
    for key, value in update.items():
        if _is_delta(value):
            owned = FRAME in value
            base = value[FRAME] if owned else result.get(key)
            result[key] = _apply_deltas(base, value[DELTAS], owned)
        else:
            result[key] = value
    return result, []

#====================
#TyssueDset Dataclass
#====================
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

#===================
#Behaviors Dataclass
#===================
@dataclass(kw_only=True)
class Behaviors(List):
    _value: List = field(default_factory=List)

def behaviors_update(schema, current, update, path):
    result = current.copy()

    # Handle removals
    for key in update.get("_remove", []):
        result.pop(key, None)

    # Apply updates (excluding _remove)
    for behavior, args in update.items():
        if behavior != "_remove":
            result[behavior] = args

    return result

@apply.dispatch
def apply(schema: Behaviors, current, update, path):
    return behaviors_update(schema, current, update, path), []

#=====================
#Registration of Types
#=====================
def register_types(core):
    core.register_type("tyssue_dset", TyssueDset)
    core.register_type("behaviors", Behaviors)
    core.register_type("tyssue_data", TyssueData)
    core.register_type("frame", Frame)
    return core

#=====
#Tests
#=====
def test_frame():
    core = allocate_core()
    _dict = {
        "a": [1.0, 5.0, 6.0, 6.0],
        "b": [1, 5, 6, 6],
        "c": [True, False, False, False],
    }
    df = pd.DataFrame(_dict)
    schema = core.infer(df)
    print("Done")

if __name__ == "__main__":
    test_frame()