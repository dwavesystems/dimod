import unittest

import itertools
import random

from dimod import ExactSolver, ising_energy, qubo_energy
from dimod.samplers.simulated_annealing import ising_simulated_annealing, greedy_coloring
from dimod.samplers.tests.generic_sampler_tests import TestSolverAPI


class TestExactSolver(unittest.TestCase, TestSolverAPI):
    def setUp(self):
        self.sampler = ExactSolver()
