import unittest

try:
    import numpy as np
    _numpy = True
except ImportError:
    _numpy = False

from dimod import NumpyResponse, NumpySpinResponse, NumpyBinaryResponse
from dimod.responses.tests.test_response import ResponseGenericTests
from dimod import ising_to_qubo, qubo_to_ising, qubo_energy, ising_energy


@unittest.skipUnless(_numpy, "numpy not installed")
class TestNumpyResponse(unittest.TestCase, ResponseGenericTests):
    response_factory = NumpyResponse
    relabel_allowed = False

    def test_response_creation_concatenation(self):
        num_samples = 100
        num_variables = 200
        samples = np.zeros((num_samples, num_variables))
        energies = np.zeros((num_samples,))

        response = NumpyResponse()

        # add samples to an empty response
        response.add_samples_from_array(samples, energies, sorted_by_energy=True)

        # add samples when some are already present
        response.add_samples_from_array(samples, energies, sorted_by_energy=True)

        self.assertEqual(len(response), 2 * num_samples)

    def test_response_creation_basic(self):

        samples = np.asarray([[-1, 1, -1, 1], [1, 1, 1, 1], [-1, -1, -1, -1]], dtype=int)
        energies = np.asarray([.5, -1, 1.4])
        sample_data = [{'a': 1}, {}, {'c': 'a'}]

        response = NumpyResponse()

        response.add_samples_from_array(samples, energies, sample_data)

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
        self.assertEqual(r_data, [{}, {'a': 1}, {'c': 'a'}])

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
            response.add_samples_from_array(np.asarray([[0]]), np.asarray([0.0]), [{}, {}])


@unittest.skipUnless(_numpy, "numpy not installed")
class TestSpinResponse(unittest.TestCase, ResponseGenericTests):
    one = 1
    zero = -1  # spin-valued
    response_factory = NumpySpinResponse
    relabel_allowed = False

    def test_as_binary(self):

        h = {0: 0, 1: 0}
        J = {(0, 1): 1}

        sample0 = {0: -1, 1: 1}
        sample1 = {0: 1, 1: -1}
        sample2 = {0: 1, 1: 1}

        Q, offset = ising_to_qubo(h, J)

        response = self.response_factory()
        response.add_samples_from([sample0, sample1, sample2], [ising_energy(h, J, sample0),
                                                                ising_energy(h, J, sample1),
                                                                ising_energy(h, J, sample2)])

        bin_response = response.as_binary(-1 * offset)
        for sample, energy in bin_response.items():
            self.assertEqual(qubo_energy(Q, sample), energy)

        bin_response = response.as_binary(-1 * offset, data_copy=True)
        data_ids = {id(data) for __, data in response.samples(data=True)}
        for __, data in bin_response.samples(data=True):
            self.assertNotIn(id(data), data_ids)

    def test_input_checking(self):
        response = self.response_factory()

        with self.assertRaises(ValueError):
            response.add_sample({0: 0}, 0.0)

        with self.assertRaises(TypeError):
            response.add_sample({0: 1})  # neither energy nor h, J

        with self.assertRaises(ValueError):
            response.add_samples_from([{0: 0}], [0.0])

        with self.assertRaises(TypeError):
            response.add_samples_from([{0: 1}])  # neither energy nor h, J

    def test_as_spin_response(self):
        response = self.response_factory()

        num_samples = 100
        num_variables = 200
        samples = np.triu(np.ones((num_samples, num_variables))) * 2 - 1
        energies = np.zeros((num_samples,))

        response.add_samples_from_array(samples, energies)

        dimod_response = response.as_spin_response()

        for s, t in zip(response, dimod_response):
            self.assertEqual(s, t)

        dimod_response = response.as_spin_response(data_copy=True)
        for (__, dat), (__, dat0) in zip(response.samples(data=True),
                                         dimod_response.samples(data=True)):
            self.assertNotEqual(id(dat), id(dat0))

    def test_sample_array_and_energy_array(self):

        # first try loading them up by matrices
        samples = np.asarray([[-1, -1, -1, -1],
                              [1, 1, 1, 1],
                              [-1, 1, -1, 1]])

        h = {0: 1, 1: -1, 2: 1, 3: 1}
        J = {(0, 2): -1, (2, 3): 1, (0, 2): .25}

        energies = np.asarray([ising_energy(h, J, row) for row in samples])

        response = self.response_factory()
        response.add_samples_from_array(samples, energies)

        r_samples = response.samples_array()
        r_energies = response.energies_array()

        # assert that samples are sorted by energy and that they agree
        for idx, row in enumerate(r_samples):
            self.assertEqual(ising_energy(h, J, row), r_energies[idx])

        for idx in range(len(r_energies) - 1):
            self.assertLessEqual(r_energies[idx], r_energies[idx + 1])
        self.assertEqual(r_samples.ndim, 2)
        self.assertEqual(r_energies.ndim, 1)

        # test the empty case
        response = self.response_factory()
        r_samples = response.samples_array()
        r_energies = response.energies_array()

        self.assertIsInstance(r_energies, np.ndarray)
        self.assertIsInstance(r_samples, np.ndarray)

        self.assertEqual(r_samples.ndim, 2)
        self.assertEqual(r_energies.ndim, 1)


@unittest.skipUnless(_numpy, "numpy not installed")
class TestBinaryResponse(unittest.TestCase, ResponseGenericTests):
    response_factory = NumpyBinaryResponse
    relabel_allowed = False

    def test_as_spin(self):
        # add_sample with no energy specified, but Q given

        Q = {(0, 0): -1, (0, 1): 1, (1, 1): -1}

        sample0 = {0: 0, 1: 1}
        sample1 = {0: 1, 1: 1}
        sample2 = {0: 0, 1: 0}

        h, J, offset = qubo_to_ising(Q)

        response = self.response_factory()
        response.add_samples_from([sample0, sample1, sample2], [qubo_energy(Q, sample0),
                                                                qubo_energy(Q, sample1),
                                                                qubo_energy(Q, sample2)])

        response.add_sample(sample0, qubo_energy(Q, sample0))

        spin_response = response.as_spin(-1 * offset)

        for sample, energy in spin_response.items():
            self.assertEqual(ising_energy(h, J, sample), energy)

        spin_response = response.as_spin(-1 * offset, data_copy=True)
        data_ids = {id(data) for __, data in response.samples(data=True)}
        for __, data in spin_response.samples(data=True):
            self.assertNotIn(id(data), data_ids)

    def test_input_checking(self):
        response = self.response_factory()

        with self.assertRaises(ValueError):
            response.add_sample({0: -1}, 0.0)

        with self.assertRaises(TypeError):
            response.add_sample({0: 0})  # neither energy nor Q

        with self.assertRaises(ValueError):
            response.add_samples_from([{0: -1}], [0.0])

        with self.assertRaises(TypeError):
            response.add_samples_from([{0: 0}])  # neither energy nor Q

    def test_as_binary_response(self):
        response = self.response_factory()

        num_samples = 100
        num_variables = 200
        samples = np.triu(np.ones((num_samples, num_variables)))
        energies = np.zeros((num_samples,))

        response.add_samples_from_array(samples, energies)

        dimod_response = response.as_binary_response()

        for s, t in zip(response, dimod_response):
            self.assertEqual(s, t)

        dimod_response = response.as_binary_response(data_copy=True)
        for (__, dat), (__, dat0) in zip(response.samples(data=True),
                                         dimod_response.samples(data=True)):
            self.assertNotEqual(id(dat), id(dat0))
