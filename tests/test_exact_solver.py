import unittest
import itertools

import dimod


class TestExactSolver(unittest.TestCase):
    def setUp(self):
        self.sampler = dimod.ExactSolver()
        self.sampler_factory = dimod.ExactSolver

    def test_all_samples(self):
        """Check that every sample is included and has the correct energy."""

        n = 4

        # create a qubo
        Q = {(v, v): (v % 3) for v in range(n)}
        Q[(0, n - 1)] = 1.3
        Q[(3, n - 2)] = -.26666666

        response = self.sampler.sample_qubo(Q)

        # print(response)

        self.assertEqual(len(response), 2**n, "incorrect number of samples returned")

        sample_tuples = set()
        for sample in response.samples():
            stpl = tuple(sample[v] for v in range(n))
            sample_tuples.add(stpl)

        for tpl in itertools.product((0, 1), repeat=n):
            self.assertIn(tpl, sample_tuples)

        # let's also double check the energy
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, dimod.qubo_energy(sample, Q))
