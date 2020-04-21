# Copyright 2018 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS F ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A dimod :term:`sampler` that uses the MST2 multistart tabu search algorithm."""

from __future__ import division

import random
import warnings
import itertools
from functools import partial

import numpy
import dimod

from tabu import TabuSearch


class TabuSampler(dimod.Sampler):
    """A tabu-search sampler.

    Examples:
        This example solves a two-variable Ising model.

        >>> from tabu import TabuSampler
        >>> samples = TabuSampler().sample_ising({'a': -0.5, 'b': 1.0}, {'ab': -1})
        >>> list(samples.data()) # doctest: +SKIP
        [Sample(sample={'a': -1, 'b': -1}, energy=-1.5, num_occurrences=1)]
        >>> samples.first.energy
        -1.5

    """

    properties = None
    parameters = None

    def __init__(self):
        self.parameters = {'tenure': [],
                           'scale_factor': [],
                           'timeout': [],
                           'num_reads': [],
                           'init_solution': []}
        self.properties = {}

    def sample(self, bqm, initial_states=None, initial_states_generator='random',
               num_reads=None, tenure=None, timeout=20, scale_factor=1, **kwargs):
        # validate/initialize initial_states
        if initial_states is None:
            initial_states = dimod.SampleSet.from_samples(
                (np.empty((0, num_variables)), bqm.variables),
                energy=0, vartype=bqm.vartype)

        if not isinstance(initial_states, dimod.SampleSet):
            raise TypeError("'initial_states' is not 'dimod.SampleSet' instance")

        # validate num_reads and/or infer them from initial_states
        if num_reads is None:
            num_reads = len(initial_states) or 1
        if not isinstance(num_reads, Integral):
            raise TypeError("'num_reads' should be a positive integer")
        if num_reads < 1:
            raise ValueError("'num_reads' should be a positive integer")

        # initial states generators
        _generators = {
            'none': self._none_generator,
            'tile': self._tile_generator,
            'random': self._random_generator
        }

        if initial_states_generator not in _generators:
            raise ValueError("unknown value for 'initial_states_generator'")

        # unpack initial_states from sampleset to numpy array, label map and vartype
        initial_states_array = initial_states.record.sample
        init_label_map = dict(map(reversed, enumerate(initial_states.variables)))
        init_vartype = initial_states.vartype

        if set(init_label_map) ^ bqm.variables:
            raise ValueError("mismatch between variables in 'initial_states' and 'bqm'")

        # reorder initial states array according to label map
        identity = lambda i: i
        get_label = inverse_mapping.get if use_label_map else identity
        ordered_labels = [init_label_map[get_label(i)] for i in range(num_variables)]
        initial_states_array = initial_states_array[:, ordered_labels]

        numpy_initial_states = np.ascontiguousarray(initial_states_array, dtype=np.int8)

        # convert to ising, if provided in binary
        if init_vartype == dimod.BINARY:
            numpy_initial_states = 2 * numpy_initial_states - 1
        elif init_vartype != dimod.SPIN:
            raise TypeError("unsupported vartype")  # pragma: no cover

        # extrapolate and/or truncate initial states, if necessary
        extrapolate = _generators[initial_states_generator]
        numpy_initial_states = extrapolate(numpy_initial_states, num_reads, num_variables, seed)
        numpy_initial_states = self._truncate_filter(numpy_initial_states, num_reads)


    @staticmethod
    def _none_generator(initial_states, num_reads, *args, **kwargs):
        if len(initial_states) < num_reads:
            raise ValueError("insufficient number of initial states given")
        return initial_states

    @staticmethod
    def _tile_generator(initial_states, num_reads, *args, **kwargs):
        if len(initial_states) < 1:
            raise ValueError("cannot tile an empty sample set of initial states")

        if len(initial_states) >= num_reads:
            return initial_states

        reps, rem = divmod(num_reads, len(initial_states))

        initial_states = np.tile(initial_states, (reps, 1))
        initial_states = np.vstack((initial_states, initial_states[:rem]))

        return initial_states

    @staticmethod
    def _random_generator(initial_states, num_reads, num_variables, seed=None):
        rem = max(0, num_reads - len(initial_states))

        np_rand = np.random.RandomState(seed)
        random_states = 2 * np_rand.randint(2, size=(rem, num_variables)).astype(np.int8) - 1

        # handle zero-length array of input states
        if len(initial_states):
            initial_states = np.vstack((initial_states, random_states))
        else:
            initial_states = random_states

        return initial_states

    @staticmethod
    def _truncate_filter(initial_states, num_reads):
        if len(initial_states) > num_reads:
            initial_states = initial_states[:num_reads]
        return initial_states
