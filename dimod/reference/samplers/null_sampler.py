# Copyright 2019 D-Wave Systems Inc.
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
# =============================================================================
"""A sampler that always returns an empty sample set."""
try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import numpy as np

from dimod.core.sampler import Sampler
from dimod.sampleset import SampleSet

__all__ = ['NullSampler']


class NullSampler(Sampler):
    """A sampler that always returns an empty sample set.

    This sampler is useful for writing unit tests where the result is not
    important.

    Args:
        parameters (iterable/dict, optional):
            If provided, sets the parameters accepted by the sample methods.
            The values given in these parameters are ignored.

    Examples:
        >>> bqm = dimod.BinaryQuadraticModel.from_qubo({('a', 'b'): 1})
        >>> sampler = dimod.NullSampler()
        >>> sampleset = sampler.sample(bqm)
        >>> len(sampleset)
        0

        Setting additional parameters for the null sampler.

        >>> bqm = dimod.BinaryQuadraticModel.from_qubo({('a', 'b'): 1})
        >>> sampler = dimod.NullSampler(parameters=['a'])
        >>> sampleset = sampler.sample(bqm, a=5)

    """
    properties = None

    parameters = None
    """Keyword arguments accepted by the sampling methods"""

    def __init__(self, parameters=None):
        self.properties = {}

        self.parameters = {}
        if parameters is not None:
            if isinstance(parameters, abc.Mapping):
                self.parameters.update(parameters)
            else:
                self.parameters.update((param, []) for param in parameters)

    def sample(self, bqm, **kwargs):
        """Return an empty sample set.

        Args:
            bqm (:obj:`.BinaryQuadraticModel`):
                The binary quadratic model determines the variables labels in
                the sample set.

            kwargs:
                As specified when constructing the null sampler.

        Returns:
            :obj:`.SampleSet`: The empty sample set.


        """
        samples = np.empty((0, len(bqm)))
        labels = iter(bqm.variables)

        for kw in kwargs:
            if kw not in self.parameters:
                raise ValueError("unknown parameter {!r}".format(kw))

        return SampleSet.from_samples_bqm((samples, labels), bqm)
