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

import concurrent.futures
import itertools
import unittest

from collections import OrderedDict

import numpy as np

import dimod


class TestResponse(unittest.TestCase):
    def test__repr__(self):

        alpha = list('abcdefghijklmnopqrstuvwxyz')

        num_variables = 3  # needs to be small enough to not snip the matrix when printed
        num_samples = 2 ** num_variables

        samples = list(itertools.product((-1, 1), repeat=num_variables))
        energies = [0.0]*num_samples
        num_occurrences = [1] * num_samples
        record = dimod.response.data_struct_array(samples, energy=energies, num_occurrences=num_occurrences)

        labels = alpha[:num_variables]

        resp = dimod.Response(record, labels, {}, dimod.SPIN)

        from dimod import Response
        from numpy import rec
        new_resp = eval(resp.__repr__())

        np.testing.assert_array_equal(resp.record, new_resp.record)

    def test_from_samples_dicts(self):
        bqm = dimod.BinaryQuadraticModel({'a': .1}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': +1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [bqm.energy(sample) for sample in samples]

        resp = dimod.Response.from_samples(samples, {'energy': energies}, {}, dimod.SPIN)

    def test_from_future_default(self):

        future = concurrent.futures.Future()

        response = dimod.Response.from_future(future)

        self.assertIsInstance(response, dimod.Response)
        self.assertFalse(hasattr(response, '_record'))  # should not have a record yet

        self.assertFalse(response.done())

        # make future return a Response
        future.set_result(dimod.Response.from_samples([-1, -1, 1], {'energy': [.5]}, {'another_field': .5}, dimod.SPIN))

        self.assertTrue(response.done())

        # accessing response.record should resolve the future
        np.testing.assert_array_equal(response.record.sample,
                                      np.array([[-1, -1, 1]]))
        np.testing.assert_array_equal(response.record.energy,
                                      np.array([.5]))
        np.testing.assert_array_equal(response.record.num_occurrences,
                                      np.array([1]))
        self.assertEqual(response.info, {'another_field': .5})
        self.assertIs(response.vartype, dimod.SPIN)
        self.assertEqual(response.variable_labels, [0, 1, 2])

    def test__iter__(self):
        samples = [{'a': 0, 'b': 1}, {'a': 1, 'b': 0},
                   {'a': 1, 'b': 1}, {'a': 0, 'b': 0}]
        energies = [0, 1, 2, 3]
        response = dimod.Response.from_samples(samples, {'energy': energies}, {}, dimod.BINARY)

        for s0, s1 in zip(response, samples):
            self.assertEqual(s0, s1)

    def test_from_samples_unlike_labels(self):
        samples = [{'a': -1, 'b': +1},
                   {'a': -1, 'c': -1}]
        energies = [sum(sample.values()) for sample in samples]

        with self.assertRaises(ValueError):
            dimod.Response.from_samples(samples, {'energy': energies}, {}, dimod.SPIN)

    def test_from_future_typical(self):
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

        def result_to_response(future):
            result = future.result()
            return dimod.Response.from_samples(result['solutions'],
                                               {'energy': result['energies'], 'num_occurrences': result['occurrences']},
                                               {}, dimod.SPIN)

        response = dimod.Response.from_future(future, result_hook=result_to_response)

        future.set_result(result)

        matrix = response.record.sample

        np.testing.assert_equal(matrix, result['samples'])

    def test_data_docstrings(self):
        samples = [{'a': 0, 'b': 1}, {'a': 1, 'b': 0}, {'a': 0, 'b': 0}]
        energies = [-1, 0, 1]
        response = dimod.Response.from_samples(samples, {'energy': energies}, {}, vartype=dimod.BINARY)

        for datum in response.data():
            sample, energy, occ = datum  # this should work
            self.assertEqual(sample, datum.sample)  # named tuple
            self.assertEqual(energy, datum.energy)  # named tuple
            self.assertIn(datum.sample, samples)
            self.assertIn(datum.energy, energies)

            if energy == -1:
                self.assertEqual(sample, {'a': 0, 'b': 1})

        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1}, {'a': -1, 'b': -1}]
        energies = [-1.0, 0.0, 1.0]
        num_spin_up = [1, 1, 0]
        response = dimod.Response.from_samples(samples, {'energy': energies, 'num_spin_up': num_spin_up},
                                               {}, vartype=dimod.SPIN)
        for datum in response.data(['num_spin_up', 'energy']):
            nsu, energy = datum
            self.assertEqual(nsu, datum.num_spin_up)
            self.assertEqual(energy, datum.energy)

    def test_change_vartype_inplace(self):
        bqm = dimod.BinaryQuadraticModel({'a': .1}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': +1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [bqm.energy(sample) for sample in samples]
        response = dimod.Response.from_samples(samples, {'energy': energies}, {}, dimod.SPIN)

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
        response.change_vartype(dimod.BINARY, 1)
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.binary.energy(sample) + 1, energy)

    def test_change_vartype_copy(self):
        bqm = dimod.BinaryQuadraticModel({'a': .1}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': +1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [bqm.energy(sample) for sample in samples]
        response = dimod.Response.from_samples(samples, {'energy': energies}, {}, dimod.SPIN)

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
        response = dimod.Response.from_samples(samples, {'energy': energies}, {}, dimod.SPIN)

        new_response = response.relabel_variables({'a': 0, 'b': 1}, inplace=False)

        # original response should not change
        for sample in response:
            self.assertIn(sample, samples)

        for sample in new_response:
            self.assertEqual(set(sample), {0, 1})

    def test_relabel_docstring(self):
        response = dimod.Response.from_samples([{'a': -1}, {'a': +1}], {'energy': [-1, 1]}, {}, dimod.SPIN)
        new_response = response.relabel_variables({'a': 0}, inplace=False)

    def test_partial_relabel_inplace(self):
        mapping = {0: '3', 1: 4, 2: 5, 3: 6, 4: 7, 5: '1', 6: '2', 7: '0'}

        response = dimod.Response.from_samples([[-1, +1, -1, +1, -1, +1, -1, +1]], {'energy': [-1]}, {}, dimod.SPIN)

        new_response = response.relabel_variables(mapping, inplace=False)

        for new_sample, sample in zip(new_response, response):
            for v, val in sample.items():
                self.assertIn(mapping[v], new_sample)
                self.assertEqual(new_sample[mapping[v]], val)

            self.assertEqual(len(sample), len(new_sample))

    def test_partial_relabel(self):

        mapping = {0: '3', 1: 4, 2: 5, 3: 6, 4: 7, 5: '1', 6: '2', 7: '0'}

        response = dimod.Response.from_samples([[-1, +1, -1, +1, -1, +1, -1, +1]], {'energy': [-1]}, {}, dimod.SPIN)
        response2 = dimod.Response.from_samples([[-1, +1, -1, +1, -1, +1, -1, +1]], {'energy': [-1]}, {}, dimod.SPIN)

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

        self.assertEqual(len(data.dtype.fields), 3)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)

    def test_single_sample(self):
        data = dimod.response.data_struct_array([-1, 1, -1], energy=[1.5])

        self.assertEqual(data.shape, (1,))

        self.assertEqual(len(data.dtype.fields), 3)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)

    def test_single_sample_nested(self):
        data = dimod.response.data_struct_array([[-1, 1, -1]], energy=[1.5])

        self.assertEqual(data.shape, (1,))

        self.assertEqual(len(data.dtype.fields), 3)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)

    def test_multiple_samples(self):
        data = dimod.response.data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5])

        self.assertEqual(data.shape, (2,))

        self.assertEqual(len(data.dtype.fields), 3)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)

    def test_extra_data_vector(self):
        data = dimod.response.data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5],
                                                num_occurrences=np.asarray([1, 2]))

        self.assertEqual(data.shape, (2,))

        self.assertEqual(len(data.dtype.fields), 3)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)
        self.assertIn('num_occurrences', data.dtype.fields)

    def test_data_vector_higher_dimension(self):
        data = dimod.response.data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5],
                                                num_occurrences=[[0, 1], [1, 2]])

        self.assertEqual(data.shape, (2,))

        self.assertEqual(len(data.dtype.fields), 3)
        self.assertIn('sample', data.dtype.fields)
        self.assertIn('energy', data.dtype.fields)
        self.assertIn('num_occurrences', data.dtype.fields)

    def test_mismatched_vector_samples_rows(self):
        with self.assertRaises(ValueError):
            dimod.response.data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5, 5.6])

    def test_protected_sample_kwarg(self):
        with self.assertRaises(TypeError):
            dimod.response.data_struct_array([[-1, +1, -1], [+1, -1, +1]], energy=[1.5, 4.5], sample=[5, 6])

    def test_missing_kwarg_energy(self):
        with self.assertRaises(TypeError):
            dimod.response.data_struct_array([[-1, +1, -1], [+1, -1, +1]], occ=[5, 6])
