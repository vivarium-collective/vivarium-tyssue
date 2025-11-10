from vivarium_tyssue.processes import register_processes

def tyssue_dset_update(schema, current, update, top_schema, top_state, path, core):
    return [i for i in update]

def behaviors_update(schema, current, update, top_schema, top_state, path, core):
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