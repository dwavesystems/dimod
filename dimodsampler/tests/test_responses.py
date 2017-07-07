import unittest

import itertools
import random
import itertools

from dimodsampler import SpinResponse, BinaryResponse
from dimodsampler import DiscreteModelResponse


class TestDiscreteModelResponse(unittest.TestCase):
    """Tests on the DiscreteModelResponse"""
    def test_empty_object(self):
        response = DiscreteModelResponse()

        # should be empty
        self.assertEqual(list(response.samples()), [])
        self.assertEqual(list(response.energies()), [])
        self.assertEqual(len(response), 0)

    def test_samples(self):
        response = DiscreteModelResponse()
        response.add_sample({0: -1}, 1, data={'n': 5})
        response.add_sample({0: 1}, -1, data={'n': 1})
        self.assertEqual(list(response.samples()),
                         [{0: 1}, {0: -1}])
        self.assertEqual(list(response.samples(data=True)),
                         [({0: 1}, {'n': 1}), ({0: -1}, {'n': 5})])

    def test_energies(self):
        response = DiscreteModelResponse()
        response.add_sample({0: -1}, 1, data={'n': 5})
        response.add_sample({0: 1}, -1, data={'n': 1})
        self.assertEqual(list(response.energies()),
                         [-1, 1])
        self.assertEqual(list(response.energies(data=True)),
                         [(-1, {'n': 1}), (1, {'n': 5})])

    def test_items(self):
        response = DiscreteModelResponse()
        response.add_sample({0: -1}, 1, data={'n': 5})
        response.add_sample({0: 1}, -1, data={'n': 1})
        self.assertEqual(list(response.items()), [({0: 1}, -1), ({0: -1}, 1)])
        self.assertEqual(list(response.items(data=True)),
                         [({0: 1}, -1, {'n': 1}), ({0: -1}, 1, {'n': 5})])

    def test_add_samples_from(self):
        response = DiscreteModelResponse()

        sample0 = {0: -1}
        energy0 = 1
        data0 = {'n': 107}

        samples = itertools.repeat(sample0, 10)
        energies = itertools.repeat(energy0, 10)
        sample_data = itertools.repeat(data0, 10)

        response.add_samples_from(samples, energies, sample_data)

        samples = itertools.repeat(sample0, 10)
        energies = itertools.repeat(energy0, 10)

        response.add_samples_from(samples, energies)

        items = itertools.repeat((sample0, energy0, data0), 10)
        response.add_samples_from(items)

        items = itertools.repeat((sample0, energy0), 10)
        response.add_samples_from(items)
