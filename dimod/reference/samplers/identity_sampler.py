# Copyright 2020 D-Wave Systems Inc.
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
"""A sampler that returns the provided initial states."""
from dimod.core import Sampler, Initialized

__all__ = ['IdentitySampler']


class IdentitySampler(Sampler, Initialized):
    """A sampler that returns the provided initial states.

    Examples:

    >>> samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': +1}]
    >>> Q = {('a', 'b'): -1}
    >>> sampler = dimod.IdentitySampler()
    >>> sampleset = sampler.sample_qubo(Q, initial_states=samples)
    >>> print(sampleset)
       a  b energy num_oc.
    1  1  1   -1.0       1
    0  0  1    0.0       1
    ['BINARY', 2 rows, 2 samples, 2 variables]

    """

    parameters = {'initial_states': [],
                  'initial_states_generator': [],
                  'num_reads': [],
                  'seed': [],
                  }
    """dict: Keyword arguments accepted by the sampling methods.

    Contents are exactly `{'initial_states': [], 'initial_states_generator':
    [], 'num_reads': [], 'seed': []}`
    """

    properties = {}

    def sample(self, bqm, *args, **kwargs):
        """Return exactly the provided initial states.

        Args:
            bqm (:class:`~dimod.BinaryQuadraticModel`):
                The binary quadratic model to be sampled.

            num_reads (int, optional, default=len(initial_states) or 1):
                Number of reads. If `num_reads` is not explicitly given, it is
                selected to match the number of initial states given.
                If no initial states are given, it defaults to 1.

            initial_states (samples-like, optional, default=None):
                One or more samples, each defining an initial state for all the
                problem variables. Initial states are given one per read, but
                if fewer than `num_reads` initial states are defined,
                additional values are generated as specified by
                `initial_states_generator`. See func:`.as_samples` for a
                description of "samples-like".

            initial_states_generator ({'none', 'tile', 'random'}, optional, default='random'):
                Defines the expansion of `initial_states` if fewer than
                `num_reads` are specified:

                * "none":
                    If the number of initial states specified is smaller than
                    `num_reads`, raises ValueError.

                * "tile":
                    Reuses the specified initial states if fewer than
                    `num_reads` or truncates if greater.

                * "random":
                    Expands the specified initial states with randomly
                    generated states if fewer than `num_reads` or truncates if
                    greater.

            seed (int (32-bit unsigned integer), optional):
                Seed to use for the PRNG. Specifying a particular seed with a
                constant set of parameters produces identical results. If not
                provided, a random seed is chosen.

        Returns:
            :obj:`.SampleSet`: The initial states as provided, generated or
            augmented.


        """
        kwargs = self.remove_unknown_kwargs(**kwargs)
        parsed = self.parse_initial_states(bqm, *args, **kwargs)

        # and we're done, we just pass back the initial state we were handed
        return parsed.initial_states
