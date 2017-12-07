import unittest

try:
    import numpy as np
    _numpy = True
except ImportError:
    _numpy = False

import dimod
from dimod.responses.tests.test_response import ResponseGenericTests


@unittest.skipUnless(_numpy, "numpy not installed")
class TestNumpyResponse(unittest.TestCase, ResponseGenericTests):
    def setUp(self):
        self.response_factory = dimod.NumpyResponse
        self.one = 1
        self.zero = 0

    def test_response_creation_concatenation(self):
        num_samples = 100
        num_variables = 200
        samples = np.zeros((num_samples, num_variables))
        energies = np.zeros((num_samples,))

        response = dimod.NumpyResponse()

        # add samples to an empty response
        response.add_samples_from_array(samples, energies)

        # add samples when some are already present
        response.add_samples_from_array(samples, energies)

        self.assertEqual(len(response), 2 * num_samples)

    def test_response_creation_basic(self):

        samples = np.asarray([[-1, 1, -1, 1], [1, 1, 1, 1], [-1, -1, -1, -1]], dtype=int)
        energies = np.asarray([.5, -1, 1.4])
        sample_data = [{'a': 1}, {}, {'c': 'a'}]

        response = dimod.NumpyResponse()

        response.add_samples_from_array(samples, energies, datalist=sample_data)

        r_samples = []
        r_energies = []
        r_data = []
        for sample, energy, data in response.items(data=True):
            r_samples.append(sample)
            r_energies.append(energy)
            r_data.append(data)

        # should be sorted by energy and in the form defined by Dimod
        self.assertEqual(r_energies, [-1, .5, 1.4])
        self.assertEqual(r_samples, [{0: 1, 1: 1, 2: 1, 3: 1},
                                     {0: -1, 1: 1, 2: -1, 3: 1},
                                     {0: -1, 1: -1, 2: -1, 3: -1}])

    def test_input_checking(self):
        response = self.response_factory()

        with self.assertRaises(TypeError):
            response.add_samples_from_array([{}], np.asarray([0.0]))

        with self.assertRaises(ValueError):
            response.add_samples_from_array(np.asarray([0]), np.asarray([0.0]))

        with self.assertRaises(TypeError):
            response.add_samples_from_array(np.asarray([[0]]), [0.0])

        with self.assertRaises(ValueError):
            response.add_samples_from_array(np.asarray([[0]]), np.asarray([[0.0]]))

        with self.assertRaises(ValueError):
            response.add_samples_from_array(np.asarray([[0]]), np.asarray([0.0, -1.]))

        with self.assertRaises(ValueError):
            response.add_samples_from_array(np.asarray([[0]]), np.asarray([0.0]), datalist=[{}, {}])


@unittest.skipUnless(_numpy, "numpy not installed")
class TestSpinResponse(TestNumpyResponse):
    def setUp(self):
        self.one = 1
        self.zero = -1  # spin-valued
        self.response_factory = dimod.NumpySpinResponse

    def test_as_binary(self):

        h = {0: 0, 1: 0}
        J = {(0, 1): 1}

        sample0 = {0: -1, 1: 1}
        sample1 = {0: 1, 1: -1}
        sample2 = {0: 1, 1: 1}

        Q, offset = dimod.ising_to_qubo(h, J)

        response = self.response_factory()
        response.add_samples_from([sample0, sample1, sample2], [dimod.ising_energy(sample0, h, J),
                                                                dimod.ising_energy(sample1, h, J),
                                                                dimod.ising_energy(sample2, h, J)])

        bin_response = response.as_binary(-1 * offset)
        for sample, energy in bin_response.items():
            self.assertEqual(dimod.qubo_energy(sample, Q), energy)

        bin_response = response.as_binary(-1 * offset)
        data_ids = {id(data) for __, data in response.samples(data=True)}
        for __, data in bin_response.samples(data=True):
            self.assertNotIn(id(data), data_ids)


@unittest.skipUnless(_numpy, "numpy not installed")
class TestBinaryResponse(TestNumpyResponse):
    def setUp(self):
        self.response_factory = dimod.NumpyBinaryResponse
        self.zero = 0
        self.one = 1

    def test_as_spin_array(self):
        response = self.response_factory()

        # set up a BQM and some samples
        Q = {(0, 0): -1, (0, 1): 1, (1, 1): -1}
        h, J, offset = dimod.qubo_to_ising(Q)

        samples = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=int)
        energies = np.array([dimod.qubo_energy(row, Q) for row in samples])
        response.add_samples_from_array(samples, energies)

        # cast to spin
        spin_response = response.as_spin(-offset)

        # check that the energies are correcct
        for sample, energy in spin_response.items():
            self.assertEqual(dimod.ising_energy(sample, h, J), energy)

        # make a new spin response
        spin_response = response.as_spin(offset)
        data_ids = {id(data) for __, data in response.samples(data=True)}
        for __, data in spin_response.samples(data=True):
            self.assertNotIn(id(data), data_ids)
