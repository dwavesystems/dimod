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

import unittest

import dimod
import numpy as np

from dimod import Initialized


class TestGenerators(unittest.TestCase):
    def setUp(self):
        self.bqm = dimod.BQM.from_ising({}, {'ab': -1, 'bc': 1, 'ac': 1})
        self.initial_states = [{'a': 1, 'b': 1, 'c': 1},
                               {'a': -1, 'b': -1, 'c': -1}]
        self.initial_sampleset = dimod.SampleSet.from_samples_bqm(
            self.initial_states, self.bqm)

    def test_2_fixed_8_random(self):
        init = Initialized().parse_initial_states(
            bqm=self.bqm,
            initial_states=self.initial_states,
            num_reads=10)
        self.assertEqual(len(init.initial_states), 10)
        self.assertEqual(init.num_reads, 10)
        self.assertEqual(list(init.initial_states.samples(sorted_by=None)[0:2]),
                         self.initial_states)

    def test_2_fixed_8_random_explicit(self):
        init = Initialized().parse_initial_states(
            bqm=self.bqm,
            initial_states=self.initial_states,
            num_reads=10,
            initial_states_generator='random')
        self.assertEqual(len(init.initial_states), 10)
        self.assertEqual(init.num_reads, 10)
        self.assertEqual(list(init.initial_states.samples(sorted_by=None)[0:2]),
                         self.initial_states)

    def test_all_random(self):
        init = Initialized().parse_initial_states(
            bqm=self.bqm,
            num_reads=10)
        self.assertEqual(len(init.initial_states), 10)
        self.assertEqual(init.num_reads, 10)

    def test_all_random_explicit(self):
        init = Initialized().parse_initial_states(
            bqm=self.bqm,
            num_reads=10,
            initial_states_generator='random')
        self.assertEqual(len(init.initial_states), 10)
        self.assertEqual(init.num_reads, 10)

    def test_truncated_to_num_reads(self):
        init = Initialized().parse_initial_states(
            bqm=self.bqm,
            initial_states=self.initial_states,
            num_reads=1
            )
        self.assertEqual(len(init.initial_states), 1)
        self.assertEqual(init.num_reads, 1)

        init = Initialized().parse_initial_states(
            bqm=self.bqm,
            initial_states=self.initial_states,
            num_reads=1,
            initial_states_generator='random',
            )
        self.assertEqual(len(init.initial_states), 1)
        self.assertEqual(init.num_reads, 1)
        self.assertEqual(list(init.initial_states.samples(sorted_by=None)[0:1]),
                         self.initial_states[:1])

        init = Initialized().parse_initial_states(
            bqm=self.bqm,
            initial_states=self.initial_states,
            num_reads=1,
            initial_states_generator='none',
            )
        self.assertEqual(len(init.initial_states), 1)
        self.assertEqual(init.num_reads, 1)
        self.assertEqual(list(init.initial_states.samples(sorted_by=None)[0:1]),
                         self.initial_states[:1])

        init = Initialized().parse_initial_states(
            bqm=self.bqm,
            initial_states=self.initial_states,
            num_reads=1,
            initial_states_generator='tile',
            )
        self.assertEqual(len(init.initial_states), 1)
        self.assertEqual(init.num_reads, 1)
        self.assertEqual(list(init.initial_states.samples(sorted_by=None)[0:1]),
                         self.initial_states[:1])

    def test_tile_2_to_5(self):
        init = Initialized().parse_initial_states(
            bqm=self.bqm,
            initial_states=self.initial_states,
            num_reads=5,
            initial_states_generator='tile',
            )

        self.assertEqual(len(init.initial_states), 5)
        self.assertEqual(init.num_reads, 5)
        self.assertEqual(list(init.initial_states.samples(sorted_by=None)[0:2]),
                         self.initial_states)
        np.testing.assert_array_equal(init.initial_states.record.sample[0:2, :],
                                      init.initial_states.record.sample[2:4, :])
        np.testing.assert_array_equal(init.initial_states.record.sample[0, :],
                                      init.initial_states.record.sample[4, :])

    def test_tile_2_to_10(self):
        init = Initialized().parse_initial_states(
            bqm=self.bqm,
            initial_states=self.initial_states,
            num_reads=10,
            initial_states_generator='tile',
            )
        self.assertEqual(len(init.initial_states), 10)
        self.assertEqual(init.num_reads, 10)
        self.assertEqual(list(init.initial_states.samples(sorted_by=None)[0:2]),
                         self.initial_states)

        np.testing.assert_array_equal(init.initial_states.record.sample[0:2, :],
                                      init.initial_states.record.sample[2:4, :])

    def test_x_vartype(self):
        samples = {'a': -1, 'b': 1}
        bqm = dimod.BQM.from_qubo({'ab': 1})

        init = Initialized().parse_initial_states(
            bqm=bqm, initial_states=samples, num_reads=10)

        self.assertIs(init.initial_states.vartype, dimod.BINARY)
        arr = init.initial_states.record.sample
        self.assertTrue(((arr == 1) ^ (arr == 0)).all())

        samples = {'a': 0, 'b': 1}
        bqm = dimod.BQM.from_ising({}, {'ab': 1})

        init = Initialized().parse_initial_states(
            bqm=bqm, initial_states=samples, num_reads=10)

        self.assertIs(init.initial_states.vartype, dimod.SPIN)
        arr = init.initial_states.record.sample
        self.assertTrue(((arr == 1) ^ (arr == -1)).all())

    def test_tile_empty(self):
        with self.assertRaises(ValueError):
            Initialized().parse_initial_states(
                bqm=self.bqm,
                initial_states_generator='tile',
                num_reads=10)

    def test_insufficent(self):
        with self.assertRaises(ValueError):
            Initialized().parse_initial_states(
                self.bqm,
                initial_states_generator='none', num_reads=3)

    def test_incompatible(self):
        with self.assertRaises(ValueError):
            Initialized().parse_initial_states(
                self.bqm,
                initial_states={'a': 1, 1: 1})

    def test_soft_num_reads(self):
        """Number of reads adapts to initial_states size, if provided."""

        # default num_reads == 1
        init = Initialized().parse_initial_states(bqm=self.bqm)
        self.assertEqual(len(init.initial_states), 1)
        self.assertEqual(init.num_reads, 1)

        # with initial_states, num_reads == len(initial_states)
        init = Initialized().parse_initial_states(
            bqm=self.bqm,
            initial_states=self.initial_states)
        self.assertEqual(len(init.initial_states), 2)
        self.assertEqual(init.num_reads, 2)

    def test_mismatched_vartype(self):
        """Input initial states are not modified when there is a mismatch between
        the vartypes of bqm and initial_states."""

        orig_sampleset = np.copy(self.initial_sampleset.record)
        result = Initialized().parse_initial_states(
            bqm=self.bqm.binary,
            initial_states=self.initial_sampleset)

        np.testing.assert_array_equal(orig_sampleset, self.initial_sampleset.record)
