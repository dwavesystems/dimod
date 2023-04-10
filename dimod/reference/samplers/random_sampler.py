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

"""
A sampler that gives random samples.
"""

from dimod.core.sampler import Sampler
from dimod.reference.samplers.identity_sampler import IdentitySampler

__all__ = ['RandomSampler']

class RandomSampler(Sampler):
    """A sampler that gives random samples for testing.

    Examples:
        This example produces 10 samples for a two-variable problem.

        >>> bqm = dimod.BinaryQuadraticModel.from_qubo({('a', 'b'): 1})
        >>> sampler = dimod.RandomSampler()
        >>> sampleset = sampler.sample(bqm, num_reads=10)
        >>> len(sampleset)
        10
    """
    properties = None

    parameters: dict = None
    """Keyword arguments accepted by the sampling methods.

    Contents are a dict with the following keys: ``{'num_reads': []}``.
    """

    def __init__(self):
        self.parameters = {'num_reads': []}
        self.properties = {}

    def sample(self, bqm, *, num_reads=10, seed=None, **kwargs):
        """Return random samples for a binary quadratic model.

        Variable assignments are chosen by coin flip.

        Args:
            bqm (:obj:`.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            num_reads (int, optional, default=10):
                Number of reads.

            seed (int (32-bit unsigned integer), optional):
                Seed to use for the PRNG. Specifying a particular seed with a
                constant set of parameters produces identical results. If not
                provided, a random seed is chosen.

        Returns:
            :obj:`.SampleSet`

        """
        # as an implementation detail, we can use IdentitySampler here, but
        # in order to save on future changes that decouple them, we won't
        # subclass
        return IdentitySampler().sample(bqm, num_reads=num_reads, seed=seed,
                                        initial_states_generator='random',
                                        **kwargs)
