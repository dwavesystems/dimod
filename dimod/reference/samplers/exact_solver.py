"""
An exact solver that calculates the energy of all possible samples.
"""
import itertools

import numpy as np
from six.moves import zip

from dimod.core.sampler import Sampler
from dimod.decorators import bqm_index_labels
from dimod.response import Response
from dimod.vartypes import Vartype

__all__ = ['ExactSolver']


class ExactSolver(Sampler):
    """A simple exact solver for testing and debugging.

    Notes:
        This solver becomes slow for problems with 18 or more
        variables.

    Examples:
        This example solves a two-variable Ising model.

        >>> import dimod
        >>> response = dimod.ExactSolver().sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
        >>> response.data_vectors['energy']
        array([-1.5, -0.5, -0.5,  2.5])

    """
    properties = None
    parameters = None

    def __init__(self):
        self.properties = {}
        self.parameters = {}

    @bqm_index_labels
    def sample(self, bqm):
        """Sample from a binary quadratic model.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

        Returns:
            :obj:`~dimod.Response`: A `dimod` :obj:`.~dimod.Response` object.


        Examples:
            This example provides samples for a two-variable Ising model.

            >>> import dimod
            >>> sampler = dimod.ExactSolver()
            >>> bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 1.0}, {(0, 1): 0.5}, -0.5, dimod.SPIN)
            >>> response = sampler.sample(bqm)
            >>> response.data_vectors['energy']
            array([-1., -2.,  1.,  0.])

        """
        M = bqm.binary.to_numpy_matrix()
        off = bqm.binary.offset

        if M.shape == (0, 0):
            return Response.empty(bqm.vartype)

        sample = np.zeros((len(bqm),), dtype=bool)

        # now we iterate, flipping one bit at a time until we have
        # traversed all samples. This is a Gray code.
        # https://en.wikipedia.org/wiki/Gray_code
        def iter_samples():
            sample = np.zeros((len(bqm)), dtype=bool)
            energy = 0.0

            yield sample.copy(), energy + off

            for i in range(1, 1 << len(bqm)):
                v = _ffs(i)

                # flip the bit in the sample
                sample[v] = not sample[v]

                # for now just calculate the energy, but there is a more clever way by calculating
                # the energy delta for the single bit flip, don't have time, pull requests
                # appreciated!
                energy = sample.dot(M).dot(sample.transpose())

                yield sample.copy(), float(energy) + off

        samples, energies = zip(*iter_samples())

        response = Response.from_matrix(np.matrix(samples, dtype='int8'), {'energy': energies},
                                        vartype=Vartype.BINARY)

        # make sure the response matches the given vartype, in-place.
        response.change_vartype(bqm.vartype, inplace=True)

        return response


def _ffs(x):
    """Gets the index of the least significant set bit of x."""
    return (x & -x).bit_length() - 1
