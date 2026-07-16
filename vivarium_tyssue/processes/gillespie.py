from pprint import pprint
import time
import cProfile

from process_bigraph import Process, Composite, Step
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
        "regulations": "map[map[map[string]]]",
        "regulation_loc": "map[string]"
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
        self.regulations = config["regulations"]
        self.regulation_loc = config["regulation_loc"]
        # Total max rate per cell type — fixed for the run, so compute once here
        # instead of rebuilding the dict on every f_rates_max call (per step).
        self._rate_per_type = {
            ct: sum(transitions.values())
            for ct, transitions in self.rates_max.items()
        }

    def f_rates_max(self, face_df, valid_types=None):
        # Per-face max total rate: each face's cell_type -> its precomputed total
        # (0 for transient states like "dividing"/"extruding", absent from the
        # map), zeroed for any type outside valid_types. np.fromiter builds the
        # array in one pass without the intermediate Python list.
        rate_per_type = self._rate_per_type
        cell_types = face_df["cell_type"].to_numpy()
        n = len(cell_types)
        if valid_types is None:
            return np.fromiter(
                (rate_per_type.get(ct, 0.0) for ct in cell_types),
                dtype=float, count=n,
            )
        vt = valid_types if isinstance(valid_types, (set, frozenset)) else set(valid_types)
        return np.fromiter(
            (rate_per_type.get(ct, 0.0) if ct in vt else 0.0 for ct in cell_types),
            dtype=float, count=n,
        )

    def calculate_timestep(self, interval, state):
        # calculate next time-step
        face_df = state["datasets"]["face_df"]
        u0 = np.random.random_sample()
        max_rates = self.f_rates_max(face_df, valid_types=self.cell_types)
        face_df["max_rate"] = max_rates
        max_total = sum(max_rates)
        time_interval = -np.log(u0) / max_total
        return time_interval

    def f_rate(self, face_df, cell_uid, cell_type, jump, uid_pos=None):
        """
        Parameters:
        cell: cell index
        jump:
        uid_pos: optional {unique_id: iloc} map so the regulation functions look
            the cell up in O(1) instead of scanning face_df.
        """
        rate_max = self.rates_max[cell_type][jump]
        rate = rate_max
        regulators = self.regulations[cell_type].get(jump) if self.regulations[cell_type] else None
        if regulators:
            michaelis = self.michaelis_constants[cell_type]
            transition = self.transition_lengths[cell_type]
            for regulator, regulation in regulators.items():
                kwargs = {"uid_pos": uid_pos}
                if regulator in self.regulation_loc.keys():
                    kwargs["axis"] = self.regulation_loc[regulator]
                key = jump + "_" + regulator
                if key in michaelis and key in transition:
                    K_j = michaelis[key]
                    k_j = transition[key]
                    regulation_function = regulations_map[regulator]
                    regulation_value = reg_pol(regulation_function(face_df, cell_uid, **kwargs), K_j, k_j)
                    if regulation == "positive":
                        rate *= regulation_value
                    if regulation == "negative":
                        rate *= 1 - regulation_value
        return rate

    def inputs(self):
        return {
            "datasets": "tyssue_data",
            "behaviors": "list[node]",
            "global_time": "float",
        }

    def outputs(self):
        return {
            "behaviors": "list[node]",
            "gillespie_trigger": "float",
        }

    def update(self, inputs, interval):
        #calculate next time-step
        face_df = inputs["datasets"]["face_df"]
        max_rates = self.f_rates_max(face_df, valid_types=self.cell_types)
        face_df["max_rate"] = max_rates
        max_total = sum(max_rates)
        # time_interval = _time_interval - interval

        #pick cell and event accept/reject
        probability = np.divide(max_rates, max_total)

        # Cache the per-face columns as arrays once; the accept/reject loop and
        # the regulation lookups then index by position instead of paying a
        # label-based .loc / boolean scan every iteration.
        uid_arr = face_df["unique_id"].to_numpy()
        celltype_arr = face_df["cell_type"].to_numpy()
        uid_pos = {int(u): p for p, u in enumerate(uid_arr)}

        #gather cells already picked for events
        existing_uids = []
        if len(inputs["behaviors"]) > 0:
            existing_uids = {d["cell_uid"] for d in inputs["behaviors"] if "cell_uid" in d.keys()}

        while True:
            # Draw a face position from the rate distribution. choice(n, p=...)
            # consumes the RNG identically to choice(index_array, p=...) but hands
            # back the position directly, so no label->row lookup is needed.
            pos = np.random.choice(len(uid_arr), 1, p=probability)[0]
            cell_uid = int(uid_arr[pos])
            if (cell_uid not in existing_uids) and (celltype_arr[pos] in self.cell_types):
                break
        cell_type = celltype_arr[pos]

        #pick event
        jumps, proba_j = list(zip(*self.rates_max[cell_type].items()))
        proba_j = np.asarray(proba_j)/sum(self.rates_max[cell_type].values())
        jump = np.random.choice(jumps, 1, p=proba_j)[0]

        #calculate rate of picked event for picked cell
        rate_event = self.f_rate(face_df, cell_uid, cell_type, jump, uid_pos)

        u1 = np.random.random_sample()

        event = []
        #accept or reject jump
        if rate_event/self.rates_max[cell_type][jump] >= u1:
            if jump == cell_type:
                event = [{
                    "func": "division",
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
            "behaviors": event,
            "gillespie_trigger": 1,
        }