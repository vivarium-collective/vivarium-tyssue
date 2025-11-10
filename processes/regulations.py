import logging
import warnings

import numpy as np
import math
from pprint import pprint

import random

from process_bigraph import Process, Composite, ProcessTypes
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from vivarium_tyssue.maps import *

class TestRegulations(Process):

    config_schema = {
        "period": "float",
        "geom": "string",
        "growth_rate": "float",
        "crit_area": "float",
    }

    def initialize(self, config):
        self.period = self.config["period"]

    def inputs(self):
        return {
            "global_time": "float",
            "datasets": "map[tyssue_dset]"
        }

    def outputs(self):
        return {
            "behaviors": "map"
        }

    def update(self, inputs, interval):
        print(inputs["global_time"])
        if (math.isclose(inputs["global_time"]%self.period, 0) or math.isclose(inputs["global_time"]%self.period, self.period)) and not math.isclose(inputs["global_time"], 0):
            cell_row = random.choice(inputs["datasets"]["Face"])
            cell_id = cell_row["face"]
            update = {
                "division": {
                    "cell_id": cell_id,
                    "crit_area": self.config["crit_area"],
                    "growth_rate": self.config["growth_rate"],
                    "dt": interval,
                }
            }
        else:
            update = {}

        return {"behaviors": update}

