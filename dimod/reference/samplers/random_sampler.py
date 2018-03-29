"""
RandomSampler
-------------

A random sampler that can be used for unit testing and debugging.
"""
from random import choice

from dimod.core.sampler import Sampler
from dimod.response import Response, SampleView

__all__ = ['RandomSampler']


class RandomSampler(Sampler):
    """Gives random samples.

    Note that this sampler is intended for testing.

    """
    properties = None
    parameters = None

    def __init__(self):
        self.parameters = {'num_reads': []}
        self.properties = {}

    def sample(self, bqm, num_reads=10):
        """Gives random samples.

        Args:
            todo

        Returns:
            :obj:`.Response`: The vartype will match the given binary quadratic model.

        Notes:
            For each variable in each sample, the value is chosen by a coin flip.

        """
        values = tuple(bqm.vartype.value)

        def _itersample():
            for __ in range(num_reads):
                sample = {v: choice(values) for v in bqm.linear}
                energy = bqm.energy(sample)

                yield sample, energy

        samples, energies = zip(*_itersample())

        return Response.from_dicts(samples, {'energy': energies}, vartype=bqm.vartype)
