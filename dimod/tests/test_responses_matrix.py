import unittest

try:
    import numpy as np
    _numpy = True
except ImportError:
    _numpy = False

from dimod import NumpyResponse


@unittest.skipUnless(_numpy, "numpy not installed")
class TestNumpyResponse(unittest.TestCase):
    def test_response_creation_typical(self):
        num_samples = 100
        num_variables = 200
        samples = np.zeros((num_samples, num_variables))
        energies = np.zeros((num_samples, 1))

        response = NumpyResponse()

        # add samples to an empty response
        response.add_samples_from_array(samples, energies, sorted_by_energy=True)

        # add samples when some are already present
        response.add_samples_from_array(samples, energies, sorted_by_energy=True)
