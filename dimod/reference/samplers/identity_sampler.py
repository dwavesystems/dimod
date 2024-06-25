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

"""A sampler that returns the provided initial states."""
import typing

from dimod.binary.binary_quadratic_model import BinaryQuadraticModel
from dimod.core import Sampler, Initialized
from dimod.sampleset import SampleSet

from dimod.typing import SamplesLike
try:
    InitialStateGenerator = typing.Literal['none', 'tile', 'random']
except AttributeError:
    InitialStateGenerator = str

__all__ = ['IdentitySampler']


class IdentitySampler(Sampler, Initialized):
    """A sampler that can return, expand, or truncate the provided initial states.

    Examples:

    >>> samples = [{'a': 1, 'b': 0}, {'a': 0, 'b': 1}]
    >>> Q = {('a', 'b'): -1}
    >>> sampler = dimod.IdentitySampler()
    >>> sampleset = sampler.sample_qubo(Q, initial_states=samples)
    >>> print(sampleset)
       a  b energy num_oc.
    0  1  0    0.0       1
    1  0  1    0.0       1
    ['BINARY', 2 rows, 2 samples, 2 variables]

    Add randomly generated samples:

    >>> sampleset = sampler.sample_qubo(Q, initial_states=samples, num_reads=4)
    >>> print(sampleset)                                  # doctest: +SKIP
       a  b energy num_oc.
    2  1  1   -1.0       1
    3  1  1   -1.0       1
    0  1  0    0.0       1
    1  0  1    0.0       1
    ['BINARY', 4 rows, 4 samples, 2 variables]

    Note that in the above output, the specified initial states are printed after
    the generated lower-energy samples but are the first samples in the
    :attr:`~dimod.SampleSet.record`.

    """

    parameters: dict = {'initial_states': [],
                  'initial_states_generator': [],
                  'num_reads': [],
                  'seed': [],
                  }
    """Keyword arguments accepted by the sampling methods.

    Contents are a dict with the following keys:
    ``{'initial_states': [], 'initial_states_generator': [], 'num_reads': [], 'seed': []}``
    """

    properties = {}

    def sample(self, bqm: BinaryQuadraticModel, *,
               initial_states: typing.Optional[SamplesLike] = None,
               initial_states_generator: InitialStateGenerator = 'random',
               num_reads: typing.Optional[int] = None,
               seed: typing.Optional[int] = None,
               **kwargs) -> SampleSet:
        """Return, expand, or truncate the provided initial states.

        Args:
            bqm: Binary quadratic model to be sampled.

            initial_states:
                One or more samples, each defining an initial state for all the
                problem variables. Initial states are given one per read, but
                if fewer than ``num_reads`` initial states are defined,
                additional values are generated as specified by
                ``initial_states_generator``. See :func:`.as_samples` for a
                description of `samples-like`.

            initial_states_generator:
                Defines the expansion of ``initial_states`` if fewer than
                ``num_reads`` are specified:

                * "none":
                    If the number of initial states specified is smaller than
                    ``num_reads``, raises ``ValueError``.

                * "tile":
                    Reuses the specified initial states if fewer than
                    ``num_reads``, or truncates if greater.

                * "random":
                    Expands the specified initial states with randomly
                    generated states if fewer than ``num_reads``, or truncates if
                    greater.

            num_reads:
                Number of reads. If not explicitly given, the number of reads is
                set to the number of initial states, or if initial states are
                not given, to the default of 1.

            seed:
                Seed (32-bit unsigned integer) to use for the PRNG. Specifying a
                particular seed with a constant set of parameters produces
                identical results. If not provided, a random seed is chosen.

        Returns:
            A :obj:`~dimod.SampleSet` with the specified initial states, optionally
            truncated or augmented.

        """
        kwargs = self.remove_unknown_kwargs(**kwargs)
        parsed = self.parse_initial_states(bqm, initial_states,
                   initial_states_generator, num_reads, seed, **kwargs)

        # and we're done, we just pass back the initial state we were handed
        return parsed.initial_states
