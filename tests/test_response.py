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

from __future__ import division

import unittest
import concurrent.futures

from collections import OrderedDict

import numpy as np

import dimod

try:
    import pandas as pd
    _pandas = True
except ImportError:
    _pandas = False


class TestResponse(unittest.TestCase):

    def test_instantiation(self):
        samples_matrix = np.array([[0, 1, 0, 1],
                                   [1, 0, 1, 0],
                                   [0, 0, 0, 0],
                                   [1, 1, 1, 1]])
        energies = np.asarray([2, 2, 0, 4], dtype=float)

        response = dimod.Response(samples_matrix, {'energy': energies}, dimod.BINARY)

        np.testing.assert_equal(samples_matrix, response.record.sample)
        np.testing.assert_allclose(energies, response.record.energy)

    def test_data_vectors_are_arrays(self):
        samples_matrix = np.array([[0, 1, 0, 1],
                                   [1, 0, 1, 0],
                                   [0, 0, 0, 0],
                                   [1, 1, 1, 1]])
        energies = [2, 2, 0, 4]
        num_occurrences = [1, 1, 2, 1]
        objects = [object() for __ in range(4)]

        data_vectors = {'energy': energies, 'occurences': num_occurrences, 'objects': objects}

        response = dimod.Response(samples_matrix, data_vectors, dimod.BINARY)

        self.assertEqual(len(response.record.dtype.fields), 4)

        for key in data_vectors:
            self.assertIn(key, response.record.dtype.fields)

            vector = response.record[key]

            self.assertIsInstance(vector, np.ndarray)

            self.assertEqual(vector.shape, (4,))  # in this case they will all be 1D

    def test_data_vectors_wrong_length(self):
        samples_matrix = np.array([[0, 1, 0, 1],
                                   [1, 0, 1, 0],
                                   [0, 0, 0, 0],
                                   [1, 1, 1, 1]])
        energies = [2, 2, 0, 4]
        num_occurrences = [1, 1, 2, 1, 1]
        objects = [object() for __ in range(4)]

        data_vectors = {'energy': energies, 'occurences': num_occurrences, 'objects': objects}

        with self.assertRaises(ValueError):
            response = dimod.Response(samples_matrix, data_vectors, dimod.BINARY)

    def test_data_vectors_not_array_like(self):
        samples_matrix = np.array([[0, 1, 0, 1],
                                   [1, 0, 1, 0],
                                   [0, 0, 0, 0],
                                   [1, 1, 1, 1]])
        energies = [2, 2, 0, 4]
        num_occurrences = 'hi there'
        objects = [object() for __ in range(4)]

        data_vectors = {'energy': energies, 'occurences': num_occurrences, 'objects': objects}

        with self.assertRaises(ValueError):
            response = dimod.Response(samples_matrix, data_vectors, dimod.BINARY)

    def test_samples_num_limited(self):
        samples_matrix = np.array([[0, 1, 0, 1],
                                   [1, 0, 1, 0],
                                   [0, 0, 0, 0],
                                   [1, 1, 1, 1]])
        energies = [2, 2, 0, 4]
        num_occurrences = [1, 1, 2, 1]
        objects = [object() for __ in range(4)]

        data_vectors = {'energy': energies, 'occurences': num_occurrences, 'objects': objects}

        response = dimod.Response(samples_matrix, data_vectors, dimod.BINARY)

        samples_list = list(response.samples())

        self.assertEqual(len(samples_list), 4)

        shortened_samples_list = list(response.samples(3))

        self.assertEqual(len(shortened_samples_list), 3)
        self.assertEqual(shortened_samples_list, samples_list[0:3])

    def test_instantiation_without_energy(self):
        samples_matrix = np.array([[0, 1, 0, 1],
                                   [1, 0, 1, 0],
                                   [0, 0, 0, 0],
                                   [1, 1, 1, 1]])

        with self.assertRaises(ValueError):
            dimod.Response(samples_matrix, {}, dimod.BINARY)

    def test_from_matrix(self):
        # binary matrix
        samples_matrix = np.array([[0, 1, 0, 1],
                                   [1, 0, 1, 0]])
        energies = np.asarray([0, 1], dtype=float)

        response = dimod.Response.from_matrix(samples_matrix, {'energy': energies}, vartype=dimod.BINARY)

        self.assertTrue(np.all(samples_matrix == response.record.sample))
        self.assertEqual(response.record.sample.dtype, np.int8)
        self.assertIs(response.vartype, dimod.BINARY)

        # spin array
        samples_matrix = np.asarray([[-1, +1, -1, +1],
                                     [+1, -1, +1, -1]])
        energies = np.asarray([-.5, 1], dtype=float)

        response = dimod.Response.from_matrix(samples_matrix, {'energy': energies}, dimod.SPIN)

        self.assertTrue(np.all(samples_matrix == response.record.sample))
        self.assertEqual(response.record.sample.dtype, np.int8)
        self.assertIs(response.vartype, dimod.SPIN)

        # with array type
        samples_matrix = np.array([[0, 1, 0, 1],
                                   [1, 0, 1, 0]])
        energies = np.asarray([0, 1], dtype=float)

        response = dimod.Response.from_matrix(samples_matrix, {'energy': energies}, vartype=dimod.BINARY)

        self.assertTrue(np.all(samples_matrix == response.record.sample))
        self.assertEqual(response.record.sample.dtype, np.int8)
        self.assertIs(response.vartype, dimod.BINARY)

    def test_from_dicts(self):
        """Typical use"""

        samples = [{'a': -1, 'b': +1},
                   {'a': -1, 'b': -1}]
        energies = [sum(sample.values()) for sample in samples]

        response = dimod.Response.from_dicts(samples, {'energy': energies}, dimod.SPIN)

        self.assertTrue(np.all(response.record.sample == np.asarray([[-1, +1], [-1, -1]])))
        self.assertEqual(response.record.sample.dtype, np.int8)
        self.assertIs(response.vartype, dimod.SPIN)
        self.assertEqual(response.variable_labels, ['a', 'b'])  # sortable in py2 and py3

    def test_from_dicts_unsortable_labels(self):
        """Python3 cannot sort unlike types"""
        samples = [OrderedDict([('a', -1), (0, +1)]),  # ordered when cast to a list
                   OrderedDict([('a', -1), (0, -1)])]
        energies = [sum(sample.values()) for sample in samples]

        response = dimod.Response.from_dicts(samples, {'energy': energies}, dimod.SPIN)

        if response.variable_labels == [0, 'a']:
            np.testing.assert_equal(response.record.sample, np.array([[+1, -1], [-1, -1]]))
        else:
            self.assertEqual(response.variable_labels, ['a', 0])
            np.testing.assert_equal(response.record.sample, np.array([[-1, +1], [-1, -1]]))
        self.assertEqual(response.record.sample.dtype, np.int8)
        self.assertIs(response.vartype, dimod.SPIN)

    def test_from_dicts_unlike_labels(self):
        samples = [{'a': -1, 'b': +1},
                   {'a': -1, 'c': -1}]
        energies = [sum(sample.values()) for sample in samples]

        with self.assertRaises(ValueError):
            dimod.Response.from_dicts(samples, {'energy': energies}, dimod.SPIN)

    @unittest.skipUnless(_pandas, "no pandas installed")
    def test_from_pandas(self):
        samples_df = pd.DataFrame([[0, 1, 0], [1, 1, 1]], columns=['a', 'b', 'c'])

        response = dimod.Response.from_pandas(samples_df, {'energy': [-1, 1]}, dimod.BINARY)

    def test_update(self):
        samples = [{'a': -1, 'b': +1},
                   {'a': -1, 'b': -1}]
        energies = [sum(sample.values()) for sample in samples]

        response0 = dimod.Response.from_dicts(samples, {'energy': energies}, dimod.SPIN)
        response1 = dimod.Response.from_dicts(samples, {'energy': energies}, dimod.SPIN)

        response0.update(response1)

        self.assertEqual(len(response0), 4)
        for field in response0.record.dtype.fields:
            self.assertEqual(len(response0.record[field]), 4)

    def test_update_energy(self):

        def _spingen():
            while True:
                yield -1
                yield +1
                yield -1
        spingen = _spingen()

        h = {v: v * .2 for v in range(8)}
        J = {(u, u + 1): .3 * u * (u + 1) for u in range(7)}

        samples = [{v: next(spingen) for v in range(8)}
                   for __ in range(10)]
        energies = [dimod.ising_energy(sample, h, J) for sample in samples]

        response = dimod.Response.from_dicts(samples, {'energy': energies}, vartype=dimod.SPIN)

        # first make sure that all the energies are good
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, dimod.ising_energy(sample, h, J))

        # now do a bunch of updates
        for __ in range(10):
            samples = [{v: next(spingen) for v in range(8)}
                       for __ in range(10)]
            energies = [dimod.ising_energy(sample, h, J) for sample in samples]

            new_response = dimod.Response.from_dicts(samples, {'energy': energies}, vartype=dimod.SPIN)

            response.update(new_response)

        self.assertEqual(len(response), 110)

        # and check again
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, dimod.ising_energy(sample, h, J))

    def test_from_futures(self):
        def _futures():
            for __ in range(2):
                future = concurrent.futures.Future()

                future.set_result({'samples': [-1, -1, 1], 'energy': [.5]})

                yield future

        response = dimod.Response.from_futures(_futures(), vartype=dimod.SPIN, num_variables=3)

        self.assertTrue(response.done())

        matrix = response.record.sample

        np.testing.assert_array_equal(matrix, np.array([[-1, -1, 1], [-1, -1, 1]]))

    def test_from_futures_column_subset(self):
        def _futures():
            for __ in range(2):
                future = concurrent.futures.Future()

                future.set_result({'samples': [-1, -1, 1], 'energy': [.5]})

                yield future

        response = dimod.Response.from_futures(_futures(), vartype=dimod.SPIN, num_variables=2,
                                               active_variables=[0, 2])

        matrix = response.record.sample

        np.testing.assert_equal(matrix, np.array([[-1, 1], [-1, 1]]))

    def test_from_futures_typical(self):
        result = {'occurrences': [1],
                  'active_variables': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                       33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                       48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                       63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                                       78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                                       93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
                                       106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
                                       118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
                  'num_occurrences': [1],
                  'num_variables': 128,
                  'format': 'qp',
                  'timing': {},
                  'solutions': [[1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1,
                                 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1,
                                 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                                 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1,
                                 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1,
                                 -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]],
                  'energies': [-704.0],
                  'samples': [[1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                               -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1,
                               1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                               -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1,
                               1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1,
                               -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
                               1, 1, 1, 1, 1, 1, -1, -1, -1, -1]]}

        future = concurrent.futures.Future()

        response = dimod.Response.from_futures([future],
                                               vartype=dimod.SPIN,
                                               num_variables=result['num_variables'],
                                               samples_key='samples',
                                               data_vector_keys={'energies': 'energy',
                                                                 'occurrences': 'occurrences'},
                                               info_keys=['timing'],
                                               variable_labels=result['active_variables'],
                                               active_variables=result['active_variables'])

        future.set_result(result)

        matrix = response.record.sample

        np.testing.assert_equal(matrix, result['samples'])

    def test_from_futures_extra_keys(self):
        def _futures():
            for __ in range(2):
                future = concurrent.futures.Future()

                future.set_result({'samples': [-1, -1, 1], 'energy': [.5]})

                yield future

        response = dimod.Response.from_futures(_futures(), vartype=dimod.SPIN, num_variables=2,
                                               active_variables=[0, 2],
                                               data_vector_keys={'energy', 'other'},
                                               info_keys=['other'])

        matrix = response.record.sample

        np.testing.assert_equal(matrix, np.array([[-1, 1], [-1, 1]]))

        #

        response = dimod.Response.from_futures(_futures(), vartype=dimod.SPIN, num_variables=2,
                                               active_variables=[0, 2],
                                               data_vector_keys={'energy', 'other'},
                                               info_keys=['other'],
                                               ignore_extra_keys=False)

        with self.assertRaises(ValueError):
            matrix = response.record.sample

        #

        response = dimod.Response.from_futures(_futures(), vartype=dimod.SPIN, num_variables=2,
                                               active_variables=[0, 2],
                                               data_vector_keys={'energy'},
                                               info_keys=['other'],
                                               ignore_extra_keys=False)

        with self.assertRaises(ValueError):
            matrix = response.record.sample

    def test_empty(self):
        response = dimod.Response.empty(dimod.SPIN)
        self.assertFalse(response)
        self.assertIs(response.vartype, dimod.SPIN)

    def test__iter__(self):
        samples = [{'a': 0, 'b': 1}, {'a': 1, 'b': 0},
                   {'a': 1, 'b': 1}, {'a': 0, 'b': 0}]
        energies = [0, 1, 2, 3]
        response = dimod.Response.from_dicts(samples, {'energy': energies}, dimod.BINARY)

        for s0, s1 in zip(response, samples):
            self.assertEqual(s0, s1)

    def test_data_docstrings(self):
        samples = [{'a': 0, 'b': 1}, {'a': 1, 'b': 0}, {'a': 0, 'b': 0}]
        energies = [-1, 0, 1]
        response = dimod.Response.from_dicts(samples, {'energy': energies}, vartype=dimod.BINARY)

        for datum in response.data():
            sample, energy = datum  # this should work
            self.assertEqual(sample, datum.sample)  # named tuple
            self.assertEqual(energy, datum.energy)  # named tuple
            self.assertIn(datum.sample, samples)
            self.assertIn(datum.energy, energies)

            if energy == -1:
                self.assertEqual(sample, {'a': 0, 'b': 1})

        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1}, {'a': -1, 'b': -1}]
        energies = [-1.0, 0.0, 1.0]
        num_spin_up = [1, 1, 0]
        response = dimod.Response.from_dicts(samples, {'energy': energies, 'num_spin_up': num_spin_up},
                                             vartype=dimod.SPIN)
        for datum in response.data(['num_spin_up', 'energy']):
            nsu, energy = datum
            self.assertEqual(nsu, datum.num_spin_up)
            self.assertEqual(energy, datum.energy)

    def test_change_vartype_inplace(self):
        bqm = dimod.BinaryQuadraticModel({'a': .1}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': +1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [bqm.energy(sample) for sample in samples]
        response = dimod.Response.from_dicts(samples, {'energy': energies}, dimod.SPIN)

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.energy(sample), energy)

        response.change_vartype(dimod.SPIN)  # should do nothing

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.energy(sample), energy)

        response.change_vartype(dimod.BINARY)  # change to binary

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.binary.energy(sample), energy)

        response.change_vartype(dimod.SPIN)  # change to back to spin

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.spin.energy(sample), energy)

        # finally change with an offset
        response.change_vartype(dimod.BINARY, {'energy': 1})
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.binary.energy(sample) + 1, energy)

    def test_change_vartype_copy(self):
        bqm = dimod.BinaryQuadraticModel({'a': .1}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': +1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [bqm.energy(sample) for sample in samples]
        response = dimod.Response.from_dicts(samples, {'energy': energies}, dimod.SPIN)

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.energy(sample), energy)

        response_copy = response.change_vartype(dimod.SPIN, inplace=False)  # should do nothing

        self.assertIsNot(response, response_copy)

        for sample, energy in response_copy.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.energy(sample), energy)

    def test_relabel_copy(self):
        bqm = dimod.BinaryQuadraticModel({'a': .1}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': -1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [bqm.energy(sample) for sample in samples]
        response = dimod.Response.from_dicts(samples, {'energy': energies}, dimod.SPIN)

        new_response = response.relabel_variables({'a': 0, 'b': 1}, inplace=False)

        # original response should not change
        for sample in response:
            self.assertIn(sample, samples)

        for sample in new_response:
            self.assertEqual(set(sample), {0, 1})

    def test_relabel_docstring(self):
        response = dimod.Response.from_dicts([{'a': -1}, {'a': +1}], {'energy': [-1, 1]}, dimod.SPIN)
        response = dimod.Response.from_dicts([{'a': -1}, {'a': +1}], {'energy': [-1, 1]}, dimod.SPIN)
        new_response = response.relabel_variables({'a': 0}, inplace=False)

    def test_partial_relabel_inplace(self):
        mapping = {0: '3', 1: 4, 2: 5, 3: 6, 4: 7, 5: '1', 6: '2', 7: '0'}

        response = dimod.Response.from_matrix([[-1, +1, -1, +1, -1, +1, -1, +1]], {'energy': [-1]}, dimod.SPIN)

        new_response = response.relabel_variables(mapping, inplace=False)

        for new_sample, sample in zip(new_response, response):
            for v, val in sample.items():
                self.assertIn(mapping[v], new_sample)
                self.assertEqual(new_sample[mapping[v]], val)

            self.assertEqual(len(sample), len(new_sample))

    def test_partial_relabel(self):

        mapping = {0: '3', 1: 4, 2: 5, 3: 6, 4: 7, 5: '1', 6: '2', 7: '0'}

        response = dimod.Response.from_matrix([[-1, +1, -1, +1, -1, +1, -1, +1]], {'energy': [-1]}, dimod.SPIN)
        response2 = dimod.Response.from_matrix([[-1, +1, -1, +1, -1, +1, -1, +1]], {'energy': [-1]}, dimod.SPIN)

        response.relabel_variables(mapping, inplace=True)

        for new_sample, sample in zip(response, response2):
            for v, val in sample.items():
                self.assertIn(mapping[v], new_sample)
                self.assertEqual(new_sample[mapping[v]], val)

            self.assertEqual(len(sample), len(new_sample))


class TestSamplesStructuredArray(unittest.TestCase):
    def test_empty(self):
        data = dimod.response.data_struct_array([], energy=[])

        self.assertEqual(data.shape, (0,))

        self.assertEqual(len(data.dtype.fields), 2)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)

    def test_single_sample(self):
        data = dimod.response.data_struct_array([-1, 1, -1], energy=[1.5])

        self.assertEqual(data.shape, (1,))

        self.assertEqual(len(data.dtype.fields), 2)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)

    def test_single_sample_nested(self):
        data = dimod.response.data_struct_array([[-1, 1, -1]], energy=[1.5])

        self.assertEqual(data.shape, (1,))

        self.assertEqual(len(data.dtype.fields), 2)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)

    def test_multiple_samples(self):
        data = dimod.response.data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5])

        self.assertEqual(data.shape, (2,))

        self.assertEqual(len(data.dtype.fields), 2)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)

    def test_extra_data_vector(self):
        data = dimod.response.data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5], occurrences=np.asarray([1, 2]))

        self.assertEqual(data.shape, (2,))

        self.assertEqual(len(data.dtype.fields), 3)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)
        self.assertIn('occurrences', data.dtype.fields)

    def test_data_vector_higher_dimension(self):
        data = dimod.response.data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5], occurrences=[[0, 1], [1, 2]])

        self.assertEqual(data.shape, (2,))

        self.assertEqual(len(data.dtype.fields), 3)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)
        self.assertIn('occurrences', data.dtype.fields)

    def test_mismatched_vector_samples_rows(self):
        with self.assertRaises(ValueError):
            dimod.response.data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5, 5.6])

    def test_protected_sample_kwarg(self):
        with self.assertRaises(TypeError):
            dimod.response.data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5], sample=[5, 6])

    def test_missing_kwarg_energy(self):
        with self.assertRaises(TypeError):
            dimod.response.data_struct_array([[-1, +1, -1], [+1, -1, +1]], occ=[5, 6])
