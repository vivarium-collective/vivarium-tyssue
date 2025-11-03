from vivarium_tyssue.processes import register_processes

def tyssue_dset_update(schema, current, update, top_schema, top_state, path, core):
    return [i for i in update]

tyssue_dset_type = {
    "_type": "tyssue_dset",
    "inherit": "list[map]",
    "_apply": tyssue_dset_update,
}

def register_types(core):
    core.register("tyssue_dset", tyssue_dset_type)
    return register_processes(core)