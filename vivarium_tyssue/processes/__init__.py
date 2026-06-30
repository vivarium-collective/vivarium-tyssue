from vivarium_tyssue.processes.eulersolver import EulerSolver
from vivarium_tyssue.processes.regulations import *
from vivarium_tyssue.processes.gillespie import *
from vivarium_tyssue.processes.tumor_coupling import TumorCoupling

def register_processes(core):
    core.register_link("EulerSolver", EulerSolver)
    core.register_link("TestRegulations", TestRegulations)
    core.register_link("StochasticLineTension", StochasticLineTension)
    core.register_link("CellJamming", CellJamming)
    core.register_link("ParameterGradient", ParameterGradient)
    core.register_link("AnisotropicTension", AnisotropicTension)
    core.register_link("Gillespie", Gillespie)
    core.register_link("TumorCoupling", TumorCoupling)
    return core