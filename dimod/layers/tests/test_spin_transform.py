import unittest
import itertools

import dimod
from dimod.samplers.tests.generic_sampler_tests import TestSamplerAPI


class MockSampler(dimod.TemplateSampler):
    def sample_ising(self, h, J, orig_h=None):

        if orig_h is not None:
            assert h != orig_h

        return dimod.ExactSolver().sample_ising(h, J)


class TestSpinTransform(unittest.TestCase):
    def test_function_typical(self):
        """Check that applying the transform twice returns the original
        form
        """
        # the form of the Ising problem is not important
        h = {v: v for v in range(100)}
        J = {(u, v): u + v for u, v in itertools.combinations(range(100), 2)}

        # apply transform
        h_spin, J_spin, transform = dimod.apply_spin_transform(h, J)

        # unapply the transform
        h_orig, J_orig, transform = dimod.apply_spin_transform(h_spin, J_spin, transform)

        self.assertEqual(h, h_orig)
        self.assertEqual(J, J_orig)


class TestSpinTransformComposition(unittest.TestCase, TestSamplerAPI):

    def setUp(self):
        ComposedSampler = dimod.SpinTransformation(MockSampler)
        self.sampler = ComposedSampler()

    def test_spin_transform_composition(self):
        sampler = self.sampler

        # let's get a problem that we know the answer to
        h = {v: .1 for v in range(10)}
        J = {(u, v): -1. for (u, v) in itertools.combinations(h, 2)}

        response = sampler.sample_ising(h, J, orig_h=h)

        # lowest energy sample should be all -1
        sample = next(iter(response))
        self.assertTrue(all(s == -1 for s in sample.values()))

        # also energy should still be preserved
        for sample, energy in response.items():
            self.assertLessEqual(abs(dimod.ising_energy(h, J, sample) - energy), 10**-5)

        Q, __ = dimod.ising_to_qubo(h, J)

        response = sampler.sample_qubo(Q)

        # also energy should still be preserved
        for sample, energy in response.items():
            self.assertLessEqual(abs(dimod.qubo_energy(Q, sample) - energy), 10**-5)
