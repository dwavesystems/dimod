import unittest

import dimod
from dimod.test import SamplerAPITest


class TestRandomSampler(unittest.TestCase, SamplerAPITest):
    def setUp(self):
        self.sampler = dimod.RandomSampler()
        self.sampler_factory = dimod.RandomSampler
