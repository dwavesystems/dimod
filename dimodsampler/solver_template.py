

class DiscreteModelSolver(object):
    """TODO"""
    def solve_qubo(self, Q, **solver_params):
        """TODO"""
        raise NotImplementedError

    def solve_ising(self, h, J, **solver_params):
        """TODO"""
        raise NotImplementedError

    def solve_structured_qubo(self, Q, **solver_params):
        """TODO"""
        raise NotImplementedError

    def solve_structured_ising(self, h, J, **solver_params):
        """TODO"""
        raise NotImplementedError

    @property
    def structure(self):
        raise NotImplementedError
