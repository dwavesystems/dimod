import unittest

from dimod import RandomSampler
from dimod.samplers.tests.generic_sampler_tests import SamplerAPITest


class TestRandomSampler(unittest.TestCase, SamplerAPITest):
    # this inherits tests from TestSamplerAPI

    def setUp(self):
        self.sampler = RandomSampler()
