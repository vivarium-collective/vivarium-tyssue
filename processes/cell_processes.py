import logging
import warnings

import numpy as np

from process_bigraph import Process, Step, Composite, ProcessTypes
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from vivarium_tyssue.maps import *


class Apoptosis(Step):

    config_schema = {

    }

    def initialize(self, config):
        pass
