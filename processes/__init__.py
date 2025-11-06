from vivarium_tyssue.processes.eulersolver import EulerSolver

def register_processes(core):
    core.register_process("EulerSolver", EulerSolver)
    return core