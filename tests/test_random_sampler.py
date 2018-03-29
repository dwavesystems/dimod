import unittest

import dimod
import dimod.testing as dtest


class TestRandomSampler(unittest.TestCase):
    def test_initialization(self):
        sampler = dimod.RandomSampler()

        dtest.assert_sampler_api(sampler)
        self.assertEqual(sampler.properties, {})
        self.assertEqual(sampler.parameters, {'num_reads': []})

    def test_energies(self):
        bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0},
                                         {(0, 1): -1.0, (1, 2): 1.0, (0, 2): 1.0},
                                         1.0,
                                         dimod.SPIN)
        sampler = dimod.RandomSampler()
        response = sampler.sample(bqm, num_reads=10)

        # check their energies
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, bqm.energy(sample))
