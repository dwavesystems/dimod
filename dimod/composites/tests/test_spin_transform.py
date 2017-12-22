import unittest
import itertools
import random

import dimod
from dimod.samplers.tests.generic_sampler_tests import SamplerAPITest

# import functions and classes we wish to test, from the 'topmost' location
from dimod.composites.spin_transform import apply_spin_reversal_transform


class MockSampler(dimod.TemplateSampler):
    def sample_ising(self, h, J, orig_h=None):

        if orig_h is not None:
            assert h != orig_h

        return dimod.ExactSolver().sample_ising(h, J)

        return dimod.ExactSolver().sample_qubo(Q)


class TestSpinTransformComposition(unittest.TestCase, SamplerAPITest):

    def setUp(self):
        self.sampler = dimod.SpinReversalTransform(MockSampler())

    def test_spin_transform_composition_basic(self):
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
            self.assertLessEqual(abs(dimod.ising_energy(sample, h, J) - energy), 10**-5)

        for __, data in response.samples(data=True):
            self.assertIn('spin_reversal_variables', data)

        Q, __ = dimod.ising_to_qubo(h, J)

        response = sampler.sample_qubo(Q)

        # lowest energy sample should be all 0
        sample = next(iter(response))
        self.assertTrue(all(s == 0 for s in sample.values()))

        # also energy should still be preserved
        for sample, energy in response.items():
            self.assertLessEqual(abs(dimod.qubo_energy(sample, Q) - energy), 10**-5)

        for __, data in response.samples(data=True):
            self.assertIn('spin_reversal_variables', data)

        response = sampler.sample_ising(h, J, orig_h=h)

        # lowest energy sample should be all -1
        sample = next(iter(response))
        self.assertTrue(all(s == -1 for s in sample.values()))

        # also energy should still be preserved
        for sample, energy in response.items():
            self.assertLessEqual(abs(dimod.ising_energy(sample, h, J) - energy), 10**-5)

        for __, data in response.samples(data=True):
            self.assertIn('spin_reversal_variables', data)

    def test_input_checking(self):
        sampler = self.sampler

        with self.assertRaises(TypeError):
            sampler.sample_ising({}, {}, num_spin_reversal_transforms=.5)

    def test_double_stack(self):

        # nested spin reversal transforms
        sampler = dimod.SpinReversalTransform(self.sampler)

        h = {v: .1 for v in range(10)}
        J = {(u, v): -1. for (u, v) in itertools.combinations(h, 2)}

        response = sampler.sample_ising(h, J, orig_h=h)

        for __, data in response.samples(data=True):
            # should be two spin-reversal-transform reports in addition to the three other fields
            self.assertEqual(len(data), 5)

    def test_multiple_spin_transforms(self):
        sampler = dimod.SpinReversalTransform(self.sampler)

        h = {v: .1 for v in range(10)}
        J = {(u, v): -1. for (u, v) in itertools.combinations(h, 2)}

        response = sampler.sample_ising(h, J)

        response2 = sampler.sample_ising(h, J, num_spin_reversal_transforms=2)

        # should be twice as many samples
        self.assertEqual(len(response2), 2 * len(response))

    def test_kwarg_propogation_composite(self):
        sampler = dimod.SpinReversalTransform(self.sampler)

        # kwargs should have propogated through, correctness is tested elsewhere
        self.assertIn('orig_h', sampler.accepted_kwargs)


class TestSpinTransform(unittest.TestCase):
    def test_function_typical(self):
        """Check that applying the transform twice returns the original
        form
        """
        # the form of the Ising problem is not important
        h = {v: v for v in range(100)}
        J = {(u, v): u + v for u, v in itertools.combinations(range(100), 2)}

        # apply transform
        h_spin, J_spin, transform = apply_spin_reversal_transform(h, J)

        # unapply the transform
        h_orig, J_orig, transform = apply_spin_reversal_transform(h_spin, J_spin, transform)

        self.assertEqual(h, h_orig)
        self.assertEqual(J, J_orig)

    def test_specifying_variables(self):
        """"""
        h = {v: v for v in range(100)}
        J = {(u, v): u + v for u, v in itertools.combinations(range(100), 2)}

        transform = {1, 5, 10, 15}

        h_spin, J_spin, transform = apply_spin_reversal_transform(h, J, transform)

        # check that the variables in transform and no others are negated
        for v, bias in h_spin.items():
            if v in transform:
                self.assertEqual(bias, -v)
            else:
                self.assertEqual(bias, v)

        for (u, v), bias in J_spin.items():
            if v in transform and u in transform:
                self.assertEqual(bias, v + u)
            elif v in transform:
                self.assertEqual(bias, -(v + u))
            elif u in transform:
                self.assertEqual(bias, -(v + u))
            else:
                self.assertEqual(bias, v + u)

    def test_specifying_variables_mismatch(self):
        # this should do nothing
        h = {0: 0, 1: 1, 2: 3, 4: 16}
        transform = {'a', 'b', 'c'}

        h_spin, J_spin, transform = apply_spin_reversal_transform(h, {}, transform)

        self.assertEqual(h, h_spin)
        self.assertEqual({}, J_spin)
