import vivarium_tyssue.models.crypt_gillespie.crypt_params as param
import numpy as np

rates_max = {}
K = {}
k = {}
regulations = {}
for i in param.cell_types:
    rates_max[i] = {}
    K[i] = {}
    k[i] = {}
    regulations[i] = {}

##############################
# Max rates: MUST define at least one per cell type
# Stem cell
rates_max['sc']['sc'] = 0.15  # 0.09
rates_max['sc']['pc'] = 0.2  # 0.04

# Progenitor cell
rates_max['pc']['pc'] = 0.22
rates_max['pc']['ent'] = 0.25  # 0.15
rates_max['pc']['gc'] = rates_max['pc']['ent'] * 0.33

# Goblet cell
rates_max['gc']['ex'] = 0.34  # 0.4

# Enterocyte
rates_max['ent']['ex'] = 0.34  # 0.4

# DCS
rates_max['dcs']['dcs'] = 0.


# Michaelis constant
K['ci'] = 41.

K['sc']['sc_z'] = 12.
K['sc']['sc_but'] = 2.
K['sc']['ci'] = 53.

K['pc']['pc_z'] = 40.

K['pc']['ent_z'] = 40.
K['pc']['ent_but'] = 1.5

K['ent']['ex_z'] = 190.
K['ent']['ex_ci'] = 20.


# k (width of transition in regulation function)
k['ci'] = 6.
k['sc']['sc_z'] = 5.
k['sc']['sc_but'] = 5.

k['pc']['pc_z'] = 40.

k['pc']['ent_z'] = 15.
k['pc']['ent_but'] = 5.

k['ent']['ex_z'] = 15.

# regulations (name and type of regulations for each jump)
regulations["sc"]["sc"] = {"wnt": "positive"}
regulations["sc"]["pc"] = {"wnt": "negative"}

regulations["pc"]["ent"] = {"wnt": "negative"}
regulations["pc"]["gc"] = {"wnt": "negative"}
regulations["pc"]["pc"] = {"wnt": "positive"}

regulations["ent"]["ex"] = {"wnt": "negative"}

regulations["gc"]["ex"] = {"wnt": "negative"}

def loc_to_wnt(loc):
    wnt = -0.67*loc + 10
    return wnt

regulations_map = {
    "wnt": loc_to_wnt,
}

JUMP_MAP = {
    "rates_max": rates_max,
    "K": K,
    "transition_lengths": k,
    "regulations": regulations,
}
