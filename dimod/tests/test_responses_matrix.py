import unittest

try:
    import numpy as np
    _numpy = True
except ImportError:
    _numpy = False

from dimod import NumpyResponse


@unittest.skipUnless(_numpy, "numpy not installed")
class TestNumpyResponse(unittest.TestCase):
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
