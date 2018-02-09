import unittest
import time
from collections import namedtuple

import numpy as np
import pandas as pd

import dimod

try:
    # py3
    import unittest.mock as mock
except ImportError:
    # py2
    import mock


Solver = namedtuple('Solver', ['nodes'])


class MockFuture():
    def __init__(self, waittime, variables, num_reads):
        self.end_time = time.time() + waittime

        self.samples = [[1 if v in variables else 3 for v in range(max(variables) + 1)]
                        for __ in range(num_reads)]
        self.energies = [sum([.1 * v for v in variables])] * num_reads
        self.occurrences = [1] * num_reads

        self.solver = Solver(variables)

    def done(self):
        return time.time() > self.end_time


class TestResponse(unittest.TestCase):
    def test_instantiation(self):
        # test empty spin bqm
        response = dimod.Response(dimod.SPIN)

    def test_add_samples_from_dict(self):
        # typical case
        response = dimod.Response(dimod.SPIN)

        bqm = dimod.BinaryQuadraticModel({'a': .1}, {('a', 'b'): -1}, 0.0, dimod.SPIN)

        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1}, {'a': +1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [bqm.energy(sample) for sample in samples]

        response.add_samples_from(samples, energies)
        response.add_samples_from(samples, energies)

        self.assertEqual(len(response), 8)

        self.assertTrue(response.df_samples.equals(pd.concat([pd.DataFrame(samples, dtype='int8', index=range(4)),
                                                              pd.DataFrame(samples, dtype='int8',
                                                                           index=range(4, 8))])))

        # try adding other fields

        response.add_samples_from(samples, energies, other=[1] * 4)

        self.assertEqual(len(response), 12)

        # try adding from pandas array
        samples_df = pd.DataFrame(samples)
        response.add_samples_from(samples_df, energies)

    def test_add_sample(self):
        response = dimod.Response(dimod.SPIN)

        sample = {'a': -1, 'b': 1}
        energy = 1

        response.add_sample(sample, energy)

        self.assertEqual(len(response), 1)

        # add a sample with additional info
        response.add_sample(sample, energy, num_occurences=1)
        self.assertEqual(len(response), 2)

        # add a pandas series
        response.add_sample(pd.Series(sample), energy)
        self.assertEqual(len(response), 3)

    def test_samples(self):
        # all defaults, given in energy order
        response = dimod.Response(dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': +1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [0, 1, 2, 3]

        response.add_samples_from(samples, energies)

        # should return itself, check twice because iterator
        self.assertEqual(list(response.samples()), samples)
        self.assertEqual(list(response.samples()), samples)
        self.assertTrue(all(isinstance(sample, dict) for sample in response.samples()))

        #

        # reverse energy ordering
        response = dimod.Response(dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': +1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [3, 2, 1, 0]

        response.add_samples_from(samples, energies)

        # should return itself reversed, check twice because iterator
        self.assertEqual(list(response.samples()), list(reversed(samples)))
        self.assertEqual(list(response.samples(sorted_by_energy=False)), samples)
        self.assertEqual(list(response.samples()), list(reversed(samples)))
        self.assertEqual(list(response.samples(sorted_by_energy=False)), samples)

        #

        response = dimod.Response(dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': +1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [0, 1, 2, 3]

        response.add_samples_from(samples, energies)

        for sample in response.samples(sample_type=pd.Series):
            self.assertIsInstance(sample, pd.Series)

    def test__iter__(self):
        response = dimod.Response(dimod.BINARY)
        samples = [{'a': 0, 'b': 1}, {'a': 1, 'b': 0},
                   {'a': 1, 'b': 1}, {'a': 0, 'b': 0}]
        energies = [0, 1, 2, 3]

        for s0, s1 in zip(response, samples):
            self.assertEqual(s0, s1)

    def test_data(self):
        response = dimod.Response(dimod.BINARY)
        response.add_sample({'a': +1, 'b': +1}, -1, num_spin_up=2)

        self.assertTrue(len(list(response.data())) == 1)

        for datum in response.data():
            self.assertEqual(datum.num_spin_up, 2)
            self.assertEqual(({'a': +1, 'b': +1}, -1, 2), datum)

        for datum in response.data(['sample', 'num_spin_up'], name=None):
            self.assertEqual(({'a': +1, 'b': +1}, 2), datum)

        for datum in response.data(['sample', 'num_spin_up'], sample_type=pd.Series):
            self.assertTrue(datum.sample.equals(pd.Series({'a': +1, 'b': +1}, dtype=object)))

        #

        # references a found bug
        response = dimod.Response(dimod.SPIN)
        response.add_sample({0: -1, 4: 1}, energy=0.0, num_occurences=1)

        for sample, energy in response.data(['sample', 'energy']):
            # there should be only one
            self.assertEqual(sample, {0: -1, 4: 1})
            self.assertEqual(energy, 0.0)

    def test_change_vartype_inplace(self):
        response = dimod.Response(dimod.SPIN)
        bqm = dimod.BinaryQuadraticModel({'a': .1}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': +1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [bqm.energy(sample) for sample in samples]

        response.add_samples_from(samples, energies)

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
        response.change_vartype(dimod.BINARY, offset=1)
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.binary.energy(sample) + 1, energy)

    def test_change_vartype_copy(self):
        response = dimod.Response(dimod.SPIN)
        bqm = dimod.BinaryQuadraticModel({'a': .1}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': +1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [bqm.energy(sample) for sample in samples]

        response.add_samples_from(samples, energies)

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.energy(sample), energy)

        response_copy = response.change_vartype(dimod.SPIN, inplace=False)  # should do nothing

        self.assertIsNot(response, response_copy)

        for sample, energy in response_copy.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.energy(sample), energy)

    @mock.patch('dimod.response.microclient')
    def test_add_samples_future(self, __):
        # don't need to do anything other than magic mock the micro client, wait_multiple just won't
        # block

        response = dimod.Response(dimod.SPIN)
        qubits = [0, 1, 2, 4, 5]

        self.assertTrue(response.done())

        # submit some problems and add the future objects to the response
        for __ in range(10):
            # wait .5s
            response.add_samples_future(MockFuture(.5, qubits, 10))

        # now try to read from it
        self.assertEqual(len(response), 100)

        for sample in response:
            for v, val in sample.items():
                self.assertIn(v, qubits)
                self.assertTrue(val in {-1, 1})

        self.assertTrue(response.done())

        for __ in range(1):
            # wait one second
            response.add_samples_future(MockFuture(.1, qubits, 10))

        # try to acces the data df
        response.df_data

        # now try to read from it
        self.assertEqual(len(response), 110)

    def test_retrieving_additional_fields(self):

        response = dimod.Response(dimod.SPIN)

        response.add_sample({'a': 1, 'b': -1}, 1.3, num_occurences=1)

        for sample, num_occurences in response.data(['sample', 'num_occurences']):
            self.assertEqual(sample, {'a': 1, 'b': -1})
            self.assertEqual(num_occurences, 1)

        response.add_sample({'a': 1, 'b': -1}, 1.3, num_occurences=1)

        self.assertEqual(len(response), 2)
        for sample, num_occurences in response.data(['sample', 'num_occurences']):
            self.assertEqual(sample, {'a': 1, 'b': -1})
            self.assertEqual(num_occurences, 1)

        #
        # index-labelled
        #

        response = dimod.Response(dimod.BINARY)

        response.add_sample({0: 1, 4: 0}, 1.3, num_occurences=1)

        for sample, num_occurences in response.data(['sample', 'num_occurences']):
            self.assertEqual(sample, {0: 1, 4: 0})
            self.assertEqual(num_occurences, 1)

        response.add_sample({0: 1, 4: 0}, 1.3, num_occurences=1)

        self.assertEqual(len(response), 2)
        for sample, num_occurences in response.data(['sample', 'num_occurences']):
            self.assertEqual(sample, {0: 1, 4: 0})
            self.assertEqual(num_occurences, 1)
