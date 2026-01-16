from dataclasses import dataclass, field
import pandas as pd

from bigraph_schema import allocate_core
from bigraph_schema.schema import dtype_schema
from bigraph_schema.schema import Node, Map, List, Array
from bigraph_schema.methods import apply, infer, realize, serialize


#===============
#Frame Dataclass
#===============
@dataclass(kw_only=True)
class Frame(Node):
    _columns: dict = field(default_factory=dict)

def frame_update(scheme, current, update, path):
    pass

@apply.dispatch
def apply(schema: Frame, state, update, path):
    return update, []

@serialize.dispatch
def serialize(schema: Frame, state):
    if not state:
        return {}
    return state.to_dict(orient="list")

@realize.dispatch
def realize(core, schema: Frame, encode, path=()):
    if isinstance(encode, pd.DataFrame):
        return schema, encode, []
    elif not encode:
        return schema, {}, []
    else:
        return schema, pd.DataFrame(encode), []

@infer.dispatch
def infer(core, value: pd.DataFrame, path=()):
    columns = get_frame_schema(value)
    return Frame(_columns=columns), []

def get_frame_schema(df):
    schema = {}
    for column in df.columns:
        schema[column] = dtype_schema(df.loc[:, column].dtype)
    return schema

#====================
#TyssueData Dataclass
#====================
@dataclass(kw_only=True)
class TyssueData(Node):
    vert_df: Frame = field(default_factory=Frame)
    edge_df: Frame = field(default_factory=Frame)
    face_df: Frame = field(default_factory=Frame)
    cell_df: Frame = field(default_factory=Frame)

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