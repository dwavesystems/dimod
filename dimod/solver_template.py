"""TODO

Examples
--------
    Define a sampler:

    class MyLinearSampler(dimod.TemplateSampler):
        @dimod.decorators.qubo(1)
        def solve_qubo(self, Q):
            response = dimod.BinaryResponse()

            sample = {}
            for (u, v) in Q:
                if u != v:
                    pass

                if Q[(u, v)] > 0:
                    val = 0
                else:
                    val = 1

                sample[v] = val

"""


from dimod.decorators import ising, qubo
from dimod.utilities import qubo_to_ising

__all__ = ['TemplateSampler']


class TemplateSampler(object):
    """Serves as a template for samplers. Not intended to be used directly.

    The methods as provided are self-referential, trying to invoke them
    directly will lead to an infinite recursion. This is done so that users
    need only implement the methods that make sense.

    See module documentation for examples.
    """
    def __init__(self):
        self.structure = None

    @qubo(1)
    def sample_qubo(self, Q, **kwargs):
        """TODO"""
        h, J, offset = qubo_to_ising(Q)
        spin_response = self.sample_ising(h, J, **kwargs)
        return spin_response.as_binary(offset)

    @ising(1, 2)
    def sample_ising(self, h, J, **solver_params):
        """TODO"""
        Q, offset = ising_to_qubo(h, J)
        binary_response = self.sample_qubo(Q, **kwargs)
        return binary_response.as_spin(offset)

    @qubo(1)
    def sample_structured_qubo(self, Q, **kwargs):
        """TODO"""
        if self.structure is not None:
            raise NotImplementedError()
        return self.sample_qubo(Q, **kwargs)

    @ising(1, 2)
    def sample_structured_ising(self, h, J, **kwargs):
        """TODO"""
        if self.structure is not None:
            raise NotImplementedError()
        return self.sample_ising(h, J, **kwargs)

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, struct):
        self._structure = struct
