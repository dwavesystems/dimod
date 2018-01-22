import unittest

import dimod
from tests.generic_sampler_tests import SamplerAPITest


class TestRandomSampler(unittest.TestCase, SamplerAPITest):
    # this inherits tests from TestSamplerAPI

    def setUp(self):
        self.sampler = dimod.RandomSampler()

    def test_keyword_propogation_random_sampler(self):
        sampler = self.sampler

        # no extra args
        self.assertEqual(set(sampler.accepted_kwargs), {'h', 'J', 'Q', 'num_samples'})
