import unittest

import dimod
import dimod.testing as dit


class MockSampler(unittest.TestCase):
    def test_instantiation(self):
        for factory in [dimod.ExactSolver, dimod.RandomSampler, dimod.SimulatedAnnealingSampler]:
            sampler = dimod.StructureComposite(factory(), [0, 1, 2], [(0, 1), (0, 2), (1, 2)])

            dit.assert_sampler_api(sampler)
            dit.assert_composite_api(sampler)
            dit.assert_structured_api(sampler)
