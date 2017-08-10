import unittest

from dimod import RandomSampler
from dimod.samplers.tests.generic_sampler_tests import TestSolverAPI


class TestRandomSampler(unittest.TestCase, TestSolverAPI):
    # this inherits tests from TestSolverAPI

    def setUp(self):
        self.sampler = RandomSampler()
