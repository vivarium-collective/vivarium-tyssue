from pprint import pprint
import time

from process_bigraph import Process, Composite
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from vivarium_tyssue.models.crypt_gillespie.crypt_params import *
from vivarium_tyssue.models.crypt_gillespie.jump_rates import *

test_rates_max = rates_max.copy()
del test_rates_max["dcs"]

def reg_pol(x, K, k):
    a = -1/(4*k**3)
    b = (3*K)/(4*k**3)
    c = -(3*K**2 - 3*k**2)/(4*k**3)
    d = ((K**3 + 2*k**3 - 3*K*k**2))/(4*k**3)

    if x < (K-k):
        y = 0
    elif x < (K+k):
        y = a*x**3 + b*x**2 + c*x + d
    else:
        y = 1

    return y

def f_rates_max(face_df, rates_max, valid_types=None):
    # Precompute total max rate per cell type
    rate_per_type = {
        ct: sum(transitions.values())
        for ct, transitions in rates_max.items()
    }

    cell_types = face_df["cell_type"].to_numpy()

    if valid_types is not None:
        valid_types = set(valid_types)

    rates = np.array([
        rate_per_type.get(ct, 0.0) if (valid_types is None or ct in valid_types) else 0.0
        for ct in cell_types
    ], dtype=float)

    return rates

def f_rate(face_df, cell_uid, cell_type, jump):
    """
    Parameters:
    cell: cell index
    jump:
    """
    rate_max = rates_max[cell_type][jump]
    rate = rate_max
    if regulations[cell_type]:
        regulators = regulations[cell_type][jump]
        loc = face_df.loc[cell_uid]["y"]

        for regulator, regulation in regulators.items():
            if (jump + "_" + regulator in K[cell_type]) & (jump + "_" + regulator in k[cell_type]):
                K_j = K[cell_type][jump + "_" + regulator]
                k_j = k[cell_type][jump + "_" + regulator]
                if regulation == "positive":
                    regulation_function = regulations_map[regulator]
                    regulation_term = reg_pol(regulation_function(loc), K_j, k_j)
                    rate *= regulation_term
                if regulation == "negative":
                    regulation_function = regulations_map[regulator]
                    regulation_term = 1 - reg_pol(regulation_function(loc), K_j, k_j)
                    rate *= regulation_term

    return rate

class Gillespie(Process):

    config_schema = {
        "cell_types": "list[string]",
        "rates_max": "map[map[float]]",
        "michaelis_constants": "map[map[float]]",
        "transition_lengths": "map[map[float]]",
        "geom": "string",
        "global_interval": "float",
        "growth_rate": "float",
        "shrink_rate": "float",
        "division_crit": "float",
        "apoptosis_crit": "float",
    }

    def initialize(self, config):
        self.cell_types = config["cell_types"]
        self.rates_max = config["rates_max"]
        self.michaelis_constants = config["michaelis_constants"]
        self.transition_lengths = config["transition_lengths"]
        self.geom = config["geom"]
        self.global_interval = config["global_interval"]
        self.growth_rate = config["growth_rate"]
        self.shrink_rate = config["shrink_rate"]
        self.division_crit = config["division_crit"]
        self.apoptosis_crit = config["apoptosis_crit"]

    def inputs(self):
        return {
            "datasets": "tyssue_data",
            "global_time": "float",
        }

    def outputs(self):
        return {
            "timestep": "float",
            "behaviors": "list[node]"
        }

    def update(self, inputs, interval):
        #calculate next time-step
        face_df = inputs["datasets"]["face_df"]
        u0 = np.random.random_sample()
        max_rates = f_rates_max(face_df, rates_max, valid_types=self.cell_types)
        face_df["max_rate"] = max_rates
        max_total = sum(max_rates)
        time_interval = -np.log(u0) / max_total

        #pick cell and event accept/reject
        probability = np.divide(max_rates, max_total)
        n_cells = len(face_df)

        #pick cell
        cell_id = np.random.choice(np.arange(0, n_cells), 1, p=probability)[0]
        cell_uid = face_df.loc[cell_id]["unique_id"]
        cell_type = face_df.loc[cell_id]["cell_type"]

        #pick event
        jumps, proba_j = list(zip(*self.rates_max[cell_type].items()))
        proba_j = np.asarray(proba_j)/sum(self.rates_max[cell_type].values())
        jump = np.random.choice(jumps, 1, p=proba_j)[0]

        #calculate rate of picked event for picked cell
        rate_event = f_rate(face_df, cell_uid, cell_type, jump)

        u1 = np.random.random_sample()

        event = []
        #accept or reject jump
        if rate_event/self.rates_max[cell_type][jump] >= u1:
            if jump == cell_type:
                event = [{
                    "func": "divide_crypt",
                    "geom": self.geom,
                    "cell_uid": cell_uid,
                    "dt": self.global_interval,
                    "cell_type": cell_type,
                    "crit_area": self.division_crit,
                    "growth_rate": self.growth_rate,
                }]

            elif jump == "ex":
                event = [{
                    "func": "apoptosis_extrusion",
                    "geom": self.geom,
                    "cell_uid": cell_uid,
                    "dt": self.global_interval,
                    "crit_area": self.apoptosis_crit,
                    "shrink_rate": self.shrink_rate,
                }]

            else:
                event = [{
                    "func": "differentiation",
                    "cell_uid": cell_uid,
                    "new_type": jump
                }]

        return {
            "timestep": time_interval,
            "behaviors": event,
        }

