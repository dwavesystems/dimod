from __future__ import division

import unittest

from collections import OrderedDict

import numpy as np
import numpy.testing as npt

import dimod

try:
    import pandas as pd
    _pandas = True
except ImportError:
    _pandas = False


def iter_spins(avg=.4):
    """infinite iterator that yields samples converging to the expected average"""

    yield 1

    total = 1
    count = 1

    while True:
        val = 1 if total / count <= avg else -1
        total += val
        count += 1
        yield val


class TestResponse(unittest.TestCase):

    ##############################################################################################
    # Properties and overrides
    ##############################################################################################

    ##############################################################################################
    # Construction
    ##############################################################################################

    def test_instantiation(self):
        samples_matrix = np.matrix([[0, 1, 0, 1],
                                    [1, 0, 1, 0],
                                    [0, 0, 0, 0],
                                    [1, 1, 1, 1]])
        energies = np.asarray([2, 2, 0, 4], dtype=float)

        response = dimod.Response(samples_matrix, {'energy': energies}, dimod.BINARY)

        npt.assert_equal(samples_matrix, response.samples_matrix)
        npt.assert_allclose(energies, response.data_vectors['energy'])

    def test_data_vector_copy(self):
        samples_matrix = np.matrix([[0, 1, 0, 1],
                                    [1, 0, 1, 0],
                                    [0, 0, 0, 0],
                                    [1, 1, 1, 1]])
        energies = np.asarray([2, 2, 0, 4], dtype=float)

        data_vectors = {'energy': energies}
        response = dimod.Response(samples_matrix, data_vectors, dimod.BINARY)

        self.assertIsNot(response.data_vectors, data_vectors)

    def test_instantiation_without_energy(self):
        samples_matrix = np.matrix([[0, 1, 0, 1],
                                    [1, 0, 1, 0],
                                    [0, 0, 0, 0],
                                    [1, 1, 1, 1]])

        with self.assertRaises(ValueError):
            dimod.Response(samples_matrix, {}, dimod.BINARY)

    def test_from_matrix(self):
        # binary matrix
        samples_matrix = np.matrix([[0, 1, 0, 1],
                                    [1, 0, 1, 0]])
        energies = np.asarray([0, 1], dtype=float)

        response = dimod.Response.from_matrix(samples_matrix, {'energy': energies})

        self.assertTrue(np.all(samples_matrix == response.samples_matrix))
        self.assertEqual(response.samples_matrix.dtype, np.int8)
        self.assertIs(response.vartype, dimod.BINARY)

        # spin array
        samples_matrix = np.asarray([[-1, +1, -1, +1],
                                     [+1, -1, +1, -1]])
        energies = np.asarray([-.5, 1], dtype=float)

        response = dimod.Response.from_matrix(samples_matrix, {'energy': energies})

        self.assertTrue(np.all(samples_matrix == response.samples_matrix))
        self.assertEqual(response.samples_matrix.dtype, np.int8)
        self.assertIs(response.vartype, dimod.SPIN)

        # with array type
        samples_matrix = np.matrix([[0, 1, 0, 1],
                                    [1, 0, 1, 0]])
        energies = np.asarray([0, 1], dtype=float)

        response = dimod.Response.from_matrix(samples_matrix, {'energy': energies}, vartype=dimod.BINARY)

        self.assertTrue(np.all(samples_matrix == response.samples_matrix))
        self.assertEqual(response.samples_matrix.dtype, np.int8)
        self.assertIs(response.vartype, dimod.BINARY)

    def test_from_dicts(self):
        """Typical use"""

        samples = [{'a': -1, 'b': +1},
                   {'a': -1, 'b': -1}]
        energies = [sum(sample.values()) for sample in samples]

        response = dimod.Response.from_dicts(samples, {'energy': energies})

        self.assertTrue(np.all(response.samples_matrix == np.asarray([[-1, +1], [-1, -1]])))
        self.assertEqual(response.samples_matrix.dtype, np.int8)
        self.assertIs(response.vartype, dimod.SPIN)
        self.assertEqual(response.variable_labels, ['a', 'b'])  # sortable in py2 and py3

    def test_from_dicts_unsortable_labels(self):
        """Python3 cannot sort unlike types"""
        samples = [OrderedDict([('a', -1), (0, +1)]),  # ordered when cast to a list
                   OrderedDict([('a', -1), (0, -1)])]
        energies = [sum(sample.values()) for sample in samples]

        response = dimod.Response.from_dicts(samples, {'energy': energies})

        if response.variable_labels == [0, 'a']:
            npt.assert_equal(response.samples_matrix, np.matrix([[+1, -1], [-1, -1]]))
        else:
            self.assertEqual(response.variable_labels, ['a', 0])
            npt.assert_equal(response.samples_matrix, np.matrix([[-1, +1], [-1, -1]]))
        self.assertEqual(response.samples_matrix.dtype, np.int8)
        self.assertIs(response.vartype, dimod.SPIN)

    def test_from_dicts_unlike_labels(self):
        samples = [{'a': -1, 'b': +1},
                   {'a': -1, 'c': -1}]
        energies = [sum(sample.values()) for sample in samples]

        with self.assertRaises(ValueError):
            dimod.Response.from_dicts(samples, {'energy': energies})

    @unittest.skipUnless(_pandas, "no pandas installed")
    def test_from_pandas(self):
        samples_df = pd.DataFrame([[0, 1, 0], [1, 1, 1]], columns=['a', 'b', 'c'])

        response = dimod.Response.from_pandas(samples_df, {'energy': [-1, 1]})

    def test_update(self):
        samples = [{'a': -1, 'b': +1},
                   {'a': -1, 'b': -1}]
        energies = [sum(sample.values()) for sample in samples]

        response0 = dimod.Response.from_dicts(samples, {'energy': energies})
        response1 = dimod.Response.from_dicts(samples, {'energy': energies})

        response0.update(response1)

        self.assertEqual(len(response0), 4)
        for vector in response0.data_vectors.values():
            self.assertEqual(len(vector), 4)

    def test_update_energy(self):

        spingen = iter(iter_spins())

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

    ###############################################################################################
    # Viewing a Response
    ###############################################################################################

    def test__iter__(self):
        samples = [{'a': 0, 'b': 1}, {'a': 1, 'b': 0},
                   {'a': 1, 'b': 1}, {'a': 0, 'b': 0}]
        energies = [0, 1, 2, 3]
        response = dimod.Response.from_dicts(samples, {'energy': energies})

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

    ###############################################################################################
    # Transformations and Copies
    ###############################################################################################

    def test_change_vartype_inplace(self):
        bqm = dimod.BinaryQuadraticModel({'a': .1}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': +1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [bqm.energy(sample) for sample in samples]
        response = dimod.Response.from_dicts(samples, {'energy': energies})

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
        response = dimod.Response.from_dicts(samples, {'energy': energies})

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
        response = dimod.Response.from_dicts(samples, {'energy': energies})

        new_response = response.relabel_variables({'a': 0, 'b': 1}, inplace=False)

        # original response should not change
        for sample in response:
            self.assertIn(sample, samples)

        for sample in new_response:
            self.assertEqual(set(sample), {0, 1})

    def test_relabel_docstring(self):
        response = dimod.Response.from_dicts([{'a': -1}, {'a': +1}], {'energy': [-1, 1]})
        response = dimod.Response.from_dicts([{'a': -1}, {'a': +1}], {'energy': [-1, 1]})
        new_response = response.relabel_variables({'a': 0}, inplace=False)

    ##############################################################################################
    # Other
    ##############################################################################################

    def test_infer_vartype(self):
        samples_matrix = np.matrix(np.ones((100, 50)), dtype='int8')

        with self.assertRaises(ValueError):
            response = dimod.response.infer_vartype(samples_matrix)
