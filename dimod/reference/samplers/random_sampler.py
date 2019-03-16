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
# =============================================================================
"""
A sampler that gives random samples.
"""
from random import choice

from dimod.core.sampler import Sampler
from dimod.sampleset import SampleSet

__all__ = ['RandomSampler']


class RandomSampler(Sampler):
    """A sampler that gives random samples for testing."""
    properties = None

    parameters = None
    """dict: Keyword arguments accepted by the sampling methods.

    Contents are exactly `{'num_reads': []}`
    """

    def __init__(self):
        self.parameters = {'num_reads': []}
        self.properties = {}

    def sample(self, bqm, num_reads=10):
        """Give random samples for a binary quadratic model.

        Variable assignments are chosen by coin flip.

        Args:
            bqm (:obj:`.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            num_reads (int, optional, default=10):
                Number of reads.

        Returns:
            :obj:`.SampleSet`

        """
        values = tuple(bqm.vartype.value)

        def _itersample():
            for __ in range(num_reads):
                sample = {v: choice(values) for v in bqm.linear}
                energy = bqm.energy(sample)

                yield sample, energy

        samples, energies = zip(*_itersample())

        return SampleSet.from_samples(samples, bqm.vartype, energies)
