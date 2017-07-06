import unittest

import time


from dwave_qasolver import SimulatedAnnealingSolver
from dwave_qasolver.solvers.simulated_annealing_solver import solve_ising_simulated_annealing
from dwave_qasolver.solvers.tests.generic_solver_tests import TestSolverAPI


class TestSASolver(unittest.TestCase, TestSolverAPI):
    def setUp(self):
        self.solver = SimulatedAnnealingSolver()

    def test_multiprocessing(self):

        solver = self.solver

        h = {0: -.5, 1: 0, 2: 1, 3: -.5}
        J = {(0, 2): -1, (1, 2): -1, (0, 3): .5, (1, 3): -1}

        solver.solve_ising(h, J, samples=100, multiprocessing=False)
        solver.solve_ising(h, J, samples=100, multiprocessing=True)


class TestSimulatedAnnealingAlgorithm(unittest.TestCase):
    def test_solve_basic(self):
        """really simple test"""

        # AND gate
        h = {0: -.5, 1: 0, 2: 1, 3: -.5}
        J = {(0, 2): -1, (1, 2): -1, (0, 3): .5, (1, 3): -1}

        solve_ising_simulated_annealing(h, J)
