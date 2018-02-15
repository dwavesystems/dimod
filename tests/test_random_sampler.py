import unittest

import dimod


class TestRandomSampler(unittest.TestCase, dimod.test.SamplerAPITestCaseMixin):
    def setUp(self):
        self.sampler = dimod.RandomSampler()
        self.sampler_factory = dimod.RandomSampler
