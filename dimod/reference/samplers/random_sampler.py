"""
A random sampler for unit testing and debugging.
"""
from random import choice

from dimod.core.sampler import Sampler
from dimod.response import Response, SampleView

__all__ = ['RandomSampler']


class RandomSampler(Sampler):
    """A sampler that gives random samples for testing.

    Examples:
        This example provides random samples for a two-variable QUBO model.

        >>> import dimod
        >>> response = dimod.RandomSampler().sample_qubo({(0, 0): -1, (1, 1): -1, (0, 1): 2}, num_reads=5)
        >>> len(response)
        5
        >>> print(next(response.data()))      # doctest: +SKIP
        Sample(sample={0: 1, 1: 0}, energy=-1.0)

    """
    properties = None
    parameters = None

    def __init__(self):
        self.parameters = {'num_reads': []}
        self.properties = {}

    def sample(self, bqm, num_reads=10):
        """Give random samples for a binary quadratic model.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.
            num_reads (int, optional):
                Number of reads.

        Returns:
            :obj:`~dimod.Response`: A `dimod` :obj:`.~dimod.Response` object.

        Notes:
            For each variable in each sample, the value is chosen by a coin flip.

        Examples:
            This example provides samples for a two-variable Ising model.

            >>> import dimod
            >>> sampler = dimod.RandomSampler()
            >>> h = {0: -1, 1: -1}
            >>> J = {(0, 1): -1}
            >>> bqm = dimod.BinaryQuadraticModel(h, J, -0.5, dimod.SPIN)
            >>> response = sampler.sample(bqm, num_reads=3)
            >>> len(response)
            3
            >>> response.data_vectors['energy']        # doctest: +SKIP
            array([ 0.5, -3.5,  0.5])

        """
        values = tuple(bqm.vartype.value)

        def _itersample():
            for __ in range(num_reads):
                sample = {v: choice(values) for v in bqm.linear}
                energy = bqm.energy(sample)

                yield sample, energy

        samples, energies = zip(*_itersample())

        return Response.from_dicts(samples, {'energy': energies}, vartype=bqm.vartype)
