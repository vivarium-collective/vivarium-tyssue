import random

from process_bigraph import Process

from vivarium_tyssue.maps import *
import vivarium_tyssue.models.crypt_gillespie.crypt_params as crypt_params

class TestRegulations(Process):

    config_schema = {
        "period": "float",
        "geom": "string",
        "growth_rate": "float",
        "crit_area": "float",
        "double": "boolean",
    }

    def initialize(self, config):
        self.period = self.config["period"]

    def inputs(self):
        return {
            "global_time": "float",
            "datasets": "map[tyssue_dset]",
        }

    def outputs(self):
        return {
            "behaviors": "map"
        }

    def update(self, inputs, interval):
        print(inputs["global_time"])
        t = inputs["global_time"]
        on_period = (math.isclose(t % self.period, 0) or math.isclose(t % self.period, self.period)) and not math.isclose(t, 0)
        if on_period:
            faces = inputs["datasets"]["Face"]
            positive_y = [row for row in faces if row["y"] > 1]

            def pick():
                return random.choice(positive_y)["face"]

            base = {
                "crit_area": self.config["crit_area"],
                "growth_rate": self.config["growth_rate"],
                "dt": interval,
                "func": "division",
            }

            if self.config["double"]:
                update = {
                    "division1": {**base, "cell_id": pick()},
                    "division2": {**base, "cell_id": pick()},
                }
            else:
                update = {
                    "division": {**base, "cell_id": pick()}
                }
        else:
            update = {}

        return {"behaviors": update}

