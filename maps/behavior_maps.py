from tyssue.behaviors.sheet.apoptosis_events import apoptosis
from tyssue.behaviors.sheet.basic_events import contraction

from vivarium_tyssue.behaviors import *

BEHAVIOR_MAP = {
    "apoptosis_basic": apoptosis_cell,
    "apoptosis": apoptosis,
    "division": division,
    "contraction": contraction,
    "stochastic_tension": stochastic_tension,
}