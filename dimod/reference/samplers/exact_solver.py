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
Solvers that calculate the energy of all possible samples.

Note:
    These samplers are designed for use in testing. Because they calculate
    energy for every possible sample, they are very slow.
"""
import numpy as np

from dimod.core.sampler import Sampler
from dimod.sampleset import SampleSet, as_samples
from dimod.core.polysampler import PolySampler
from dimod.vartypes import Vartype
from dimod.exceptions import SamplerUnknownArgWarning

__all__ = ['ExactSolver', 'ExactPolySolver', 'ExactDQMSolver']


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
        >>> print(sampleset)   # doctest: +SKIP
           a  b energy num_oc.
        0 -1 -1   -2.0       1
        2 +1 +1   -1.0       1
        1 +1 -1    0.0       1
        3 -1 +1    3.0       1
        ['SPIN', 4 rows, 4 samples, 2 variables]

        This example solves a two-variable QUBO.

        >>> Q = {('a', 'b'): 2.0, ('a', 'a'): 1.0, ('b', 'b'): -0.5}
        >>> sampleset = dimod.ExactSolver().sample_qubo(Q)
        >>> sampleset.first.sample
        {'a': 0, 'b': 1}

        This example solves a two-variable binary quadratic model.

        >>> bqm = dimod.BinaryQuadraticModel({'a': 1.5}, {('a', 'b'): -1}, 0.0, 'SPIN')
        >>> sampleset = dimod.ExactSolver().sample(bqm)
        >>> sampleset.first.energy
        -2.5

    """
    properties = None
    parameters = None

    def __init__(self):
        self.properties = {}
        self.parameters = {}

    def sample(self, bqm, **kwargs):
        """Sample from a binary quadratic model.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

        Returns:
            :obj:`~dimod.SampleSet`

        """
        kwargs = self.remove_unknown_kwargs(**kwargs)

        n = len(bqm.variables)
        if n == 0:
            return SampleSet.from_samples([], bqm.vartype, energy=[])

        samples = _graycode(bqm)

        if bqm.vartype is Vartype.SPIN:
            samples = 2*samples - 1

        response = SampleSet.from_samples_bqm((samples, list(bqm.variables)), bqm)
        return response


class ExactPolySolver(PolySampler):
    """A simple exact polynomial solver for testing/debugging code on your CPU.

    Notes:
        This solver becomes slow for problems with 18 or more
        variables.

    Examples:
        This example solves a three-variable hising model.

        >>> h = {'a': -0.5, 'b': 1.0, 'c': 0.}
        >>> J = {('a', 'b'): -1.5, ('a', 'b', 'c'): -1.0}
        >>> sampleset = dimod.ExactPolySolver().sample_hising(h, J)
        >>> print(sampleset)      # doctest: +SKIP
           a  b  c energy num_oc.
        1 -1 -1 +1   -3.0       1
        5 +1 +1 +1   -2.0       1
        0 -1 -1 -1   -1.0       1
        3 +1 -1 -1   -1.0       1
        4 +1 +1 -1    0.0       1
        2 +1 -1 +1    1.0       1
        7 -1 +1 -1    2.0       1
        6 -1 +1 +1    4.0       1
        ['SPIN', 8 rows, 8 samples, 3 variables]

        This example solves a three-variable HUBO.

        >>> Q = {('a', 'b'): 2.0, ('c',): 1.0, ('a', 'b', 'c'): -0.5}
        >>> sampleset = dimod.ExactPolySolver().sample_hubo(Q)
        >>> sampleset.first.energy
        0.0

        This example solves a three-variable binary polynomial

        >>> poly = dimod.BinaryPolynomial({('a',): 1.5, ('a', 'b'): -1, ('a', 'b', 'c'): 0.5}, 'SPIN')
        >>> sampleset = dimod.ExactPolySolver().sample_poly(poly)
        >>> sampleset.first.sample
        {'a': -1, 'b': -1, 'c': -1}

    """
    properties = None
    parameters = None

    def __init__(self):
        self.properties = {}
        self.parameters = {}

    def sample_poly(self, polynomial, **kwargs):
        """Sample from a binary polynomial.

        Args:
            polynomial (:obj:`~dimod.BinaryPolynomial`):
                Binary polynomial to be sampled from.

        Returns:
            :obj:`~dimod.SampleSet`

        """
        return ExactSolver().sample(polynomial, **kwargs)

class ExactDQMSolver():
    """A simple exact solver for testing and debugging code using your local CPU.

    Notes:
        This solver calculates the energy for every possible
        combination of variable cases. If variable i has
        k_i many cases, this will be k_1 * k_2 * ... * k_n
        which grows exponentially for constant k_i in the
        number of variables.

    """
    properties = None
    parameters = None

    def __init__(self):
        self.properties = {}
        self.parameters = {}

    def sample_dqm(self, dqm, **kwargs):
        """Sample from a discrete quadratic model.

        Args:
            dqm (:obj:`~dimod.DiscreteQuadraticModel`):
                Discrete quadratic model to be sampled from.

        Returns:
            :obj:`~dimod.SampleSet`

        """
        Sampler.remove_unknown_kwargs(self, **kwargs)
        
        n = dqm.num_variables()
        if n == 0:
            return SampleSet.from_samples([], 'DISCRETE', energy=[])

        possible_samples = as_samples((_all_cases_dqm(dqm), list(dqm.variables)))
        energies = dqm.energies(possible_samples)
        
        response = SampleSet.from_samples(possible_samples, 'DISCRETE', energies)
        return response


def _graycode(bqm):
    """Get a numpy array containing all possible samples in a gray-code order"""
    # developer note: there are better/faster ways to do this, but because
    # we're limited in performance by the energy calculation, this is probably
    # more readable and easier.
    n = len(bqm.variables)
    ns = 1 << n
    samples = np.empty((ns, n), dtype=np.int8)

    samples[0, :] = 0

    for i in range(1, ns):
        v = (i & -i).bit_length() - 1  # the least significant set bit of i
        samples[i, :] = samples[i - 1, :]
        samples[i, v] = not samples[i - 1, v]

    return samples


def _all_cases_dqm(dqm):
    """Get a numpy array containing all possible samples as lists of integers"""
    # developer note: there may be better ways to do this, but because we're
    # limited in performance by the energy calculation, this is probably fine

    cases = [range(dqm.num_cases(v)) for v in dqm.variables] 
    combinations = np.array(np.meshgrid(*cases)).T.reshape(-1,dqm.num_variables())
    
    return combinations