# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================
"""
A solver that calculates the energy of all possible samples.

Note:
    This sampler is designed for use in testing. Because it calculates the
    energy for every possible sample, it is very slow.

"""
import itertools

import numpy as np
from six.moves import zip

from dimod.core.sampler import Sampler
from dimod.decorators import bqm_index_labels
from dimod.sampleset import SampleSet
from dimod.vartypes import Vartype

__all__ = ['ExactSolver']


class ExactSolver(Sampler):
    """A simple exact solver for testing and debugging code using your local CPU.

    Notes:
        This solver becomes slow for problems with 18 or more
        variables.

    Examples:
        This example solves a two-variable Ising model.

        >>> h = {'a': -0.5, 'b': 1.0}
        >>> J = {('a', 'b'): -1.5}
        >>> sampleset = dimod.ExactSolver().sample_ising(h, J)
        >>> print(sampleset)
           a  b energy num_oc.
        0 -1 -1   -2.0       1
        2 +1 +1   -1.0       1
        1 +1 -1    0.0       1
        3 -1 +1    3.0       1
        ['SPIN', 4 rows, 4 samples, 2 variables]

        This example solves a two-variable QUBO.

        >>> Q = {('a', 'b'): 2.0, ('a', 'a'): 1.0, ('b', 'b'): -0.5}
        >>> sampleset = dimod.ExactSolver().sample_qubo(Q)


        This example solves a two-variable binary quadratic model

        >>> bqm = dimod.BinaryQuadraticModel({'a': 1.5}, {('a', 'b'): -1}, 0.0, 'SPIN')
        >>> sampleset = dimod.ExactSolver().sample(bqm)

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
            :obj:`~dimod.SampleSet`

        """
        M = bqm.binary.to_numpy_matrix()
        off = bqm.binary.offset

        if M.shape == (0, 0):
            return SampleSet.from_samples([], bqm.vartype, energy=[])

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

        response = SampleSet.from_samples(np.array(samples, dtype='int8'), Vartype.BINARY, energies)

        # make sure the response matches the given vartype, in-place.
        response.change_vartype(bqm.vartype, inplace=True)

        return response


def _ffs(x):
    """Gets the index of the least significant set bit of x."""
    return (x & -x).bit_length() - 1
