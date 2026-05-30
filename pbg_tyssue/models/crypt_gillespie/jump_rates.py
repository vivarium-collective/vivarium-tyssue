import pbg_tyssue.models.crypt_gillespie.crypt_params as param
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
def cell_to_wnt(face_df, cell_uid, axis="z"):

    loc = dict(face_df.loc[face_df["unique_id"] == cell_uid][["x", "y", "z"]])
    return loc[axis].values[0]

def cell_to_density(face_df, cell_uid):
    return 1 / face_df.loc[face_df["unique_id"]==cell_uid, 'area'].values[0]

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
