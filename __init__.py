from vivarium_tyssue.processes import register_processes

def tyssue_dset_update(schema, current, update, top_schema, top_state, path, core):
    if len(current) == 0:
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