import vivarium_tyssue.models.crypt_gillespie.crypt_params as param
import numpy as np

rates_max = {}
K = {}
k = {}
regulations = {}
regulation_loc = {}
for i in param.cell_types:
    rates_max[i] = {}
    K[i] = {}
    k[i] = {}
    regulations[i] = {}

##############################
# Max rates: MUST define at least one per cell type
# Stem cell
rates_max['sc']['sc'] = 0.15  # 0.09
rates_max['sc']['pc'] = 0.2 # 0.04

# Progenitor cell
rates_max['pc']['pc'] = 0.22
rates_max['pc']['ent'] = 0.15 # 0.15
rates_max['pc']['gc'] = rates_max['pc']['ent'] * 0.33

# Goblet cell
rates_max['gc']['ex'] = 0.34 # 0.4

# Enterocyte
rates_max['ent']['ex'] = 0.34  # 0.4

# DCS
rates_max['dcs']['dcs'] = 0.


# Michaelis constant
K['ci'] = 41.

K['sc']['sc_wnt'] = 12*(44/200)
K['sc']['sc_density'] = 1
K['sc']['pc_wnt'] = 12*(44/200)

K['pc']['pc_wnt'] = 40*(44/200)
K['pc']['pc_density'] = 1
K['pc']['ent_wnt'] = 40.*(44/200)
K['pc']['gc_wnt'] = 40.*(44/200)

K['ent']['ex_wnt'] = 195.*(44/200)
K['ent']['ex_density'] = 1
K['gc']['ex_wnt'] = 195.*(44/200)
K['gc']['ex_density'] = 1


# k (width of transition in regulation function)
k['sc']['sc_wnt'] = 5.*(44/200)
k['sc']['sc_density'] = 0.3
k['sc']['pc_wnt'] = 5.*(44/200)

k['pc']['pc_wnt'] = 40.*(44/200)
k['pc']['pc_density'] = 0.3
k['pc']['gc_wnt'] = 15.*(44/200)
k['pc']['ent_wnt'] = 15.*(44/200)

k['ent']['ex_wnt'] = 15.*(44/200)
k['ent']['ex_density'] = 0.3
k['gc']['ex_wnt'] = 15.*(44/200)
k['gc']['ex_density'] = 0.3


# regulations (name and type of regulations for each jump)
regulations["sc"]["sc"] = {
    "wnt": "negative",
    "density": "negative"
}
regulations["sc"]["pc"] = {
    "wnt": "positive",
}

regulations["pc"]["ent"] = {
    "wnt": "positive"
}
regulations["pc"]["gc"] = {
    "wnt": "positive"
}
regulations["pc"]["pc"] = {
    "wnt": "negative",
    "density": "negative"
}

regulations["ent"]["ex"] = {
    "wnt": "positive",
    "density": "positive"
}
regulations["gc"]["ex"] = {
    "wnt": "positive",
    "density": "positive"
}

#define axis used for location of regulation value calculation
regulation_loc["wnt"] = "z"

#regulation value functions - calculates the x value that goes into the reg_pol function
def _face_pos(face_df, cell_uid, uid_pos):
    """Positional row of ``cell_uid`` in ``face_df``. ``uid_pos`` is an optional
    precomputed ``{unique_id: iloc}`` map (built once per Gillespie step) that
    turns the former O(Nfaces) boolean scan into an O(1) lookup; falls back to a
    scan when it isn't supplied (keeps the functions usable standalone)."""
    if uid_pos is not None:
        return uid_pos[int(cell_uid)]
    return int(np.flatnonzero(face_df["unique_id"].to_numpy() == cell_uid)[0])


def cell_to_wnt(face_df, cell_uid, uid_pos=None, axis="z"):
    pos = _face_pos(face_df, cell_uid, uid_pos)
    return float(np.asarray(face_df[axis])[pos])

# Floor on the area used for the density regulator. A face can be *momentarily*
# degenerate — area exactly 0.0 — right after a division, an extrusion's remove_face
# collapses its vertices, or a T1 swap, before geometry is refreshed and the remnant
# pruned; the Gillespie reads face_df through the shared store and can sample it in
# that window, where 1/area raised ZeroDivisionError and killed the whole run.
#
# Clamping is exact here, not an approximation: this value feeds
# reg_pol(x, K, k) with K=1, k=0.3 for every density-regulated transition, and that
# smoothstep saturates at 1 for x >= K+k, i.e. for any area <= ~0.77. So every area
# below the floor already yields the same regulation value as the area -> 0 limit
# (infinite density = maximal crowding).
_MIN_DENSITY_AREA = 1e-9


def cell_to_density(face_df, cell_uid, uid_pos=None):
    pos = _face_pos(face_df, cell_uid, uid_pos)
    area = float(np.asarray(face_df["area"])[pos])
    return 1 / max(area, _MIN_DENSITY_AREA)

#dict mapping regulation label to regulation value function
regulations_map = {
    "wnt": cell_to_wnt,
    "density": cell_to_density,
}

#maps the jump parameters dicts to their labels
JUMP_MAP = {
    "rates_max": rates_max,
    "K": K,
    "transition_lengths": k,
    "regulations": regulations,
}
