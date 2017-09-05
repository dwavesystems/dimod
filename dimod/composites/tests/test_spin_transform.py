import unittest
import itertools

import dimod
from dimod.samplers.tests.generic_sampler_tests import TestSamplerAPI

# import functions and classes we wish to test, from the 'topmost' location
from dimod.composites.spin_transform import apply_spin_reversal_transform
from dimod.composites.spin_transform import apply_spin_reversal_transform_qubo


class MockSampler(dimod.TemplateSampler):
    def sample_ising(self, h, J, orig_h=None):

        if orig_h is not None:
            assert h != orig_h

        return dimod.ExactSolver().sample_ising(h, J)

        return dimod.ExactSolver().sample_qubo(Q)


# class TestSpinTransformComposition(unittest.TestCase, TestSamplerAPI):

#     def setUp(self):
#         ComposedSampler = dimod.SpinTransform(MockSampler)
#         self.sampler = ComposedSampler()

#     def test_spin_transform_composition_basic(self):
#         sampler = self.sampler

#         # let's get a problem that we know the answer to
#         h = {v: .1 for v in range(10)}
#         J = {(u, v): -1. for (u, v) in itertools.combinations(h, 2)}

#         response = sampler.sample_ising(h, J, orig_h=h)

#         # lowest energy sample should be all -1
#         sample = next(iter(response))
#         self.assertTrue(all(s == -1 for s in sample.values()))

#         # also energy should still be preserved
#         for sample, energy in response.items():
#             self.assertLessEqual(abs(dimod.ising_energy(h, J, sample) - energy), 10**-5)

#         for __, data in response.samples(data=True):
#             self.assertIn('spin_reversal_variables', data)

#         Q, __ = dimod.ising_to_qubo(h, J)

#         response = sampler.sample_qubo(Q)

#         # lowest energy sample should be all 0
#         sample = next(iter(response))
#         self.assertTrue(all(s == 0 for s in sample.values()))

#         # also energy should still be preserved
#         for sample, energy in response.items():
#             self.assertLessEqual(abs(dimod.qubo_energy(Q, sample) - energy), 10**-5)

#         for __, data in response.samples(data=True):
#             self.assertIn('spin_reversal_variables', data)


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

    def test_qubo(self):
        Q = {(0, 0): 1, (1, 1): -1, (0, 1): 1}  # minimized for [0, 1]

        Q_spin, transform, offset = apply_spin_reversal_transform_qubo(Q, {1})

        response = dimod.ExactSolver().sample_qubo(Q_spin)
        sample = next(iter(response))

        self.assertEqual(sample, {0: 0, 1: 0})  # expect 1 to be reversed

        Q.update({(0, 2): .4, (2, 2): 1.6, (3, 4): .1, (0, 4): -.5})

        for __ in range(10):
            # apply a random transform
            Q_spin, transform, offset = apply_spin_reversal_transform_qubo(Q)

            sample = next(iter(dimod.ExactSolver().sample_qubo(Q_spin)))

            sample_orig = {v: 1 - s if v in transform else s for v, s in sample.items()}

            self.assertLessEqual(abs(dimod.qubo_energy(Q_spin, sample) -
                                     dimod.qubo_energy(Q, sample_orig) + offset), 10**-5)
