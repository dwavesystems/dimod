import unittest

import itertools

from dimod import ExactSolver, ising_energy, qubo_energy
from dimod.samplers.tests.generic_sampler_tests import TestSamplerAPI


class TestExactSolver(unittest.TestCase, TestSamplerAPI):
    def setUp(self):
        self.sampler = ExactSolver()

    def test_all_samples(self):
        """Check that every sample is included and that they all have the correct energy."""

        n = 10

        # create a qubo
        Q = {(v, v): (v % 3) for v in range(n)}
        Q[(0, n - 1)] = 1.3
        Q[(3, n - 2)] = -.26666666

        response = self.sampler.sample_qubo(Q)

        self.assertEqual(len(response), 2**n, "incorrect number of samples returned")

        sample_tuples = set()
        for sample in response.samples():
            stpl = tuple(sample[v] for v in range(n))
            sample_tuples.add(stpl)

        for tpl in itertools.product((0, 1), repeat=n):
            self.assertIn(tpl, sample_tuples)

        # let's also double check the enegy
        for sample, energy in response.items():
            self.assertTrue(abs(energy - qubo_energy(Q, sample)) < .000001)
