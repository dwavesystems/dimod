import unittest

import dimod


class MockSampler(unittest.TestCase, dimod.test.CompositeAPITest):
    def setUp(self):
        self.sampler_factories = [dimod.ExactSolver, dimod.RandomSampler, dimod.SimulatedAnnealingSampler]

        def composite_factory(sampler):
            return dimod.StructureComposite(sampler, [0, 1, 2], [(0, 1), (0, 2), (1, 2)])
        self.composite_factory = composite_factory
