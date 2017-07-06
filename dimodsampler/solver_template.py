

class DiscreteModelSolver(object):
    def solve_qubo(self, Q, **solver_params):
        """Implemented solvers should overwrite this.

        Args:
            Q (dict): A dict encoding a qubo.

        Returns:


        """
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
