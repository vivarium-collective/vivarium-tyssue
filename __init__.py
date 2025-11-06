from vivarium_tyssue.processes import register_processes

def tyssue_dset_update(schema, current, update, top_schema, top_state, path, core):
    return [i for i in update]

def behaviors_update(schema, current, update, top_schema, top_state, path, core):
    return {behavior:args for behavior, args in update.items()}

tyssue_dset_type = {
    "_type": "tyssue_dset",
    "_inherit": "list[map[any]]",
    "_apply": tyssue_dset_update,
}

behaviors_type = {
    "_type": "behavior",
    "_inherit": "map[map[any]]",
    "_apply": behaviors_update,
}

def register_types(core):
    core.register("tyssue_dset", tyssue_dset_type)
    core.register("behaviors", behaviors_type)
    return register_processes(core)