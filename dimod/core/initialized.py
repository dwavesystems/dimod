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

"""
Some samplers require or accept an initial starting point in sample-space.

See the source code for :class:`.IdentitySampler` for an example of using
this abstract base class in a sampler.

"""
from __future__ import annotations

import abc

from collections import namedtuple
from numbers import Integral

from typing import Optional, Tuple

import numpy as np

from dimod.sampleset import as_samples, infer_vartype, SampleSet
from dimod.vartypes import Vartype
from dimod.binary.binary_quadratic_model import BinaryQuadraticModel
from dimod.typing import SampleLike

__all__ = ['Initialized']


ParsedInputs = namedtuple('ParsedInputs',
                          ['initial_states',
                           'initial_states_generator',
                           'num_reads',
                           'seed'])


class Initialized(abc.ABC):

    # Allows new generators to be registered
    _generators = {}

    # dev note: if this function is updated, make sure to also update
    # IdentitySampler
    def parse_initial_states(self, bqm: BinaryQuadraticModel,
                             initial_states = None,
                             initial_states_generator: str = 'random',
                             num_reads: Optional[int] = None,
                             seed: Optional[int] = None,
                             copy_always: bool = False) -> Tuple[SampleLike, str, int, int]:
        """Parse or generate initial states for an initialized sampler.

        Args:
            bqm:
                Binary quadratic model.

            initial_states (samples-like):
                One or more samples, each defining an initial state for all the
                problem variables. Initial states are given one per read, but
                if fewer than ``num_reads`` initial states are defined,
                additional values are generated as specified by
                ``initial_states_generator``. See func:`dimod.as_samples` for a
                description of "samples-like".

            initial_states_generator:
                Defines the expansion of ``initial_states`` if fewer than
                ``num_reads`` are specified:

                * "none":
                    If the number of initial states specified is smaller than
                    ``num_reads``, raises ValueError.

                * "tile":
                    Reuses the specified initial states if fewer than
                    ``num_reads`` or truncates if greater.

                * "random":
                    Expands the specified initial states with randomly
                    generated states if fewer than ``num_reads`` or truncates if
                    greater.

            num_reads:
                Number of reads. Defaults to the number of initial states, if
                ``initial_states`` is specified, or to 1, if not.

            seed:
                32-bit unsigned integer seed to use for the PRNG. Specifying a
                particular seed with a constant set of parameters produces
                identical results. If not provided, a random seed is chosen.

            copy_always:
                If True, ``initial_states`` is always copied; otherwise it is
                copied only if necessary.

        Returns:
            A named tuple with ``['initial_states', 'initial_states_generator',
            'num_reads', 'seed']`` as generated by this function.

        """

        num_variables = len(bqm)

        # validate/initialize initial_states
        if initial_states is None:
            initial_states_array = np.empty((0, num_variables), dtype=np.int8)
            initial_states_variables = list(bqm.variables)
            initial_states_vartype = bqm.vartype
        else:
            # confirm that the vartype matches and/or make it match
            if isinstance(initial_states, SampleSet):
                initial_states_vartype = initial_states.vartype
            else:
                # check based on values, defaulting to match the current bqm
                initial_states_vartype = infer_vartype(initial_states) or bqm.vartype

            if not copy_always:
                # only copy if there's a vartype mismatch
                copy_always = initial_states_vartype != bqm.vartype

            initial_states_array, initial_states_variables = \
                as_samples(initial_states, copy=copy_always)

            # confirm that the variables match
            if bqm.variables ^ initial_states_variables:
                raise ValueError("mismatch between variables in "
                                 "'initial_states' and 'bqm'")

            # match the vartype of the initial_states to the bqm
            if initial_states_vartype is Vartype.SPIN and bqm.vartype is Vartype.BINARY:
                initial_states_array += 1
                initial_states_array //= 2
            elif initial_states_vartype is Vartype.BINARY and bqm.vartype is Vartype.SPIN:
                initial_states_array *= 2
                initial_states_array -= 1

        # validate num_reads and/or infer them from initial_states
        if num_reads is None:
            num_reads = len(initial_states_array) or 1
        if not isinstance(num_reads, Integral):
            raise TypeError("'num_reads' should be a positive integer")
        if num_reads < 1:
            raise ValueError("'num_reads' should be a positive integer")

        # fill/generate the initial states as needed
        if initial_states_generator not in self._generators:
            raise ValueError("unknown value for 'initial_states_generator'")

        extrapolate = self._generators[initial_states_generator]
        initial_states_array = extrapolate(initial_states=initial_states_array,
                                           num_reads=num_reads,
                                           num_variables=num_variables,
                                           seed=seed,
                                           vartype=bqm.vartype)
        initial_states_array = self._truncate_filter(initial_states_array, num_reads)

        sampleset = SampleSet.from_samples_bqm((initial_states_array,
                                                initial_states_variables),
                                               bqm)

        return ParsedInputs(sampleset, initial_states_generator, num_reads,
                            seed)

    @staticmethod
    def _truncate_filter(initial_states, num_reads):
        if len(initial_states) > num_reads:
            initial_states = initial_states[:num_reads]
        return initial_states


def _none_generator(initial_states, num_reads, *args, **kwargs):
    if len(initial_states) < num_reads:
        raise ValueError("insufficient number of initial states given")
    return initial_states


Initialized._generators.update(none=_none_generator)


def _tile_generator(initial_states, num_reads, *args, **kwargs):
    if len(initial_states) < 1:
        raise ValueError("cannot tile an empty sample set of initial states")

    if len(initial_states) >= num_reads:
        return initial_states

    reps, rem = divmod(num_reads, len(initial_states))

    initial_states = np.tile(initial_states, (reps, 1))
    initial_states = np.vstack((initial_states, initial_states[:rem]))

    return initial_states


Initialized._generators.update(tile=_tile_generator)


def _random_generator(initial_states, num_reads, num_variables, vartype, seed=None):
    rem = max(0, num_reads - len(initial_states))

    np_rand = np.random.RandomState(seed)

    # sort vartype so that seed is reproducable
    values = np.asarray(sorted(vartype.value), dtype=np.int8)

    # takes dtype from values
    random_states = np_rand.choice(values, size=(rem, num_variables))

    # handle zero-length array of input states
    if len(initial_states):
        initial_states = np.vstack((initial_states, random_states))
    else:
        initial_states = random_states

    return initial_states


Initialized._generators.update(random=_random_generator)
