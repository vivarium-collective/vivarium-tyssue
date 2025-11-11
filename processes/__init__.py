from vivarium_tyssue.processes.eulersolver import EulerSolver
from vivarium_tyssue.processes.regulations import *

def register_processes(core):
    core.register_process("EulerSolver", EulerSolver)
    core.register_process("TestRegulations", TestRegulations)
    return core