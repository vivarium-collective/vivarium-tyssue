from pprint import pprint
import time

from process_bigraph import Process, Composite, ProcessTypes
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from vivarium_tyssue.models.crypt_gillespie.crypt_params import *
from vivarium_tyssue.models.crypt_gillespie.jump_rates import *

def generate_linear_gradient(dicts, axis, spatial_range, gradient, name):
    """
    Parameters:
    dicts: list[dict], tyssue_datasets type df in records format
    axis: str, spatial dimension for linear gradient
    spatial_range: tuple, upper and lower bounds of gradient
    gradient: float, strength of gradient
    name: string, name of linear gradient
    """
    concentrations = [{name: gradient * (_dict[axis] - spatial_range[0]), "unique_id": _dict["unique_id"]} for _dict in dicts]

    return concentrations


# =====
# Tests
# =====

dicts = [{'unique_id': 310,
  'unique_id_max': 1024,
  'vol': 0.0,
  'x': -2.2971073900456007,
  'y': -0.9514940661205715,
  'z': -9.848586232280129},
 {'unique_id': 311,
  'unique_id_max': 1024,
  'vol': 0.0,
  'x': 1.7575334321840888,
  'y': 1.757582033566359,
  'z': 7.83622455369009},
 {'unique_id': 312,
  'unique_id_max': 1024,
  'vol': 0.0,
  'x': -1.7594550252037042,
  'y': -1.7594875583588863,
  'z': -2.2754752568594987},
 {'unique_id': 313,
  'unique_id_max': 1024,
  'vol': 0.0,
  'x': -4.583203694163984e-05,
  'y': 2.4870403435627417,
  'z': 8.851422146186744},
 {'unique_id': 314,
  'unique_id_max': 1024,
  'vol': 0.0,
  'x': 0.9514670031854419,
  'y': 2.2970346481329007,
  'z': -9.848322205148774},
 {'unique_id': 315,
  'unique_id_max': 1024,
  'vol': 0.0,
  'x': 4.465552169726402e-06,
  'y': 2.4846369424825965,
  'z': -9.341955609477525},
 {'unique_id': 316,
  'unique_id_max': 1024,
  'vol': 0.0,
  'x': 0.950218960292827,
  'y': 2.293998407431126,
  'z': -8.836671514835482}]

if __name__=='__main__':
    start = time.time()
    print(start)
    concentrations = generate_linear_gradient(dicts, "z", (-10,10), 0.5, "butyrate_concentration")
    print(f"{time.time() - start} seconds")
    pprint(concentrations)