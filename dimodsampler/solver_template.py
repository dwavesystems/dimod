__all__ = ['DiscreteModelSampler']


class DiscreteModelSampler(object):
    """TODO"""
    def sample_qubo(self, Q, **solver_params):
        """TODO"""
        raise NotImplementedError

    def sample_ising(self, h, J, **solver_params):
        """TODO"""
        raise NotImplementedError

    def sample_structured_qubo(self, Q, **solver_params):
        """TODO"""
        raise NotImplementedError

    def sample_structured_ising(self, h, J, **solver_params):
        """TODO"""
        raise NotImplementedError

    @property
    def structure(self):
        raise NotImplementedError
