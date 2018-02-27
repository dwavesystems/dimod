"""
RandomSampler
-------------

A random sampler that can be used for unit testing and debugging.
"""
import numpy as np

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
        values = np.asarray(list(bqm.vartype.value), dtype='int8')
        samples = np.random.choice(values, (num_reads, len(bqm)))
        variable_labels = list(bqm.linear)
        label_to_idx = {v: idx for idx, v in enumerate(variable_labels)}

        energies = [bqm.energy(SampleView(idx, samples, label_to_idx)) for idx in range(num_reads)]

        return Response.from_matrix(samples, {'energy': energies},
                                    vartype=bqm.vartype, variable_labels=variable_labels)
