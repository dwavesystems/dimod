import unittest
import itertools
import random

import dimod
import dimod.testing as dit


class TestSpinTransformComposite(unittest.TestCase):
    def test_instantiation(self):
        for factory in [dimod.ExactSolver, dimod.RandomSampler, dimod.SimulatedAnnealingSampler]:

            sampler = dimod.SpinReversalTransformComposite(factory())

            dit.assert_sampler_api(sampler)
            dit.assert_composite_api(sampler)
