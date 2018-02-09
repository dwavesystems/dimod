"""
RandomSampler
-------------

A random sampler that can be used for unit testing and debugging.
"""
import numpy as np
import pandas as pd

from dimod.classes.sampler import Sampler
from dimod.response import Response

__all__ = ['RandomSampler']


class RandomSampler(Sampler):
    """Gives random samples.

    Note that this sampler is intended for testing.

    """
    def __init__(self):
        Sampler.__init__(self)
        self.sample_kwargs = {'num_reads': []}

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
        df_samples = pd.DataFrame(samples, columns=bqm.linear)

        energies = [bqm.energy(sample) for idx, sample in df_samples.iterrows()]

        response = Response(bqm.vartype)
        response.add_samples_from(df_samples, energies)

        return response
