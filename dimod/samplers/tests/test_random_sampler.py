import unittest

from dimod import RandomSampler
from dimod.samplers.tests.generic_sampler_tests import TestSamplerAPI


class TestRandomSampler(unittest.TestCase, TestSamplerAPI):
    # this inherits tests from TestSamplerAPI

    def setUp(self):
        self.sampler = RandomSampler()
