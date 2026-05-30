from pbg_tyssue.processes.eulersolver import EulerSolver
from pbg_tyssue.processes.regulations import *
from pbg_tyssue.processes.gillespie import *

def register_processes(core):
    core.register_link("EulerSolver", EulerSolver)
    core.register_link("TestRegulations", TestRegulations)
    core.register_link("StochasticLineTension", StochasticLineTension)
    core.register_link("CellJamming", CellJamming)
    core.register_link("ParameterGradient", ParameterGradient)
    core.register_link("AnisotropicTension", AnisotropicTension)
    core.register_link("Gillespie", Gillespie)
    return core