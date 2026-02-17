from vivarium_tyssue.processes.eulersolver import EulerSolver
from vivarium_tyssue.processes.regulations import *

def register_processes(core):
    core.register_link("EulerSolver", EulerSolver)
    core.register_link("TestRegulations", TestRegulations)
    core.register_link("StochasticLineTension", StochasticLineTension)
    core.register_link("CellJamming", CellJamming)
    return core