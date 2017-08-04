"""TODO, TODO again

Examples
--------
    Define a sampler that only operates on QUBO problems:

    >>> class MyLinearSampler(dimod.TemplateSampler):
    ...     @dimod.decorators.qubo(1)
    ...     def sample_qubo(self, Q):
    ...         response = dimod.BinaryResponse()
    ...
    ...         sample = {}
    ...         for (u, v) in Q:
    ...             if u != v:
    ...                 pass
    ...
    ...             if Q[(u, v)] > 0:
    ...                 val = 0
    ...             else:
    ...                 val = 1
    ...
    ...             sample[v] = val
    ...
    ...         response.add_sample(sample, Q=Q)
    ...         return response

    This will now behave as expected

    >>> Q = {(0, 0): 1, (1, 1): 0}
    >>> response = MyLinearSampler().sample_qubo(Q)
    >>> list(response.samples())
    [{0: 0, 1: 1}]

    Also, by implementing one of the methods, we now can use the others.

    >>> h = {0: -1, 1: 2}
    >>> J = {}
    >>> response = MyLinearSampler().sample_ising(h, J)
    >>> list(response.samples())
    [{0: 1, 1: -1}]

    Similarily for the structured methods.

    >>> h = {0: -1, 1: 2}
    >>> J = {}
    >>> response = MyLinearSampler().sample_structured_ising(h, J)
    >>> list(response.samples())
    [{0: 1, 1: -1}]
    >>> Q = {(0, 0): 1, (1, 1): 0}
    >>> response = MyLinearSampler().sample_structured_qubo(Q)
    >>> list(response.samples())
    [{0: 0, 1: 1}]

    However, if we assign a structure to our sampler, the structured
    methods will no longer work.

    >>> sampler = MyLinearSampler()
    >>> sampler.structure = 'linear'
    >>> try:
    ...     sampler.sample_structured_qubo({})
    ... except NotImplementedError:
    ...     print('not implemented')
    'not implemented'

"""


from dimod.decorators import ising, qubo
from dimod.utilities import qubo_to_ising, ising_to_qubo

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
    def sample_ising(self, h, J, **kwargs):
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
