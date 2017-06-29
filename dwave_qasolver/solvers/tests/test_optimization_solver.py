import unittest

from dwave_qasolver.solvers.tests.generic_solver_tests import TestSolverAPI

from dwave_qasolver.solvers import SoftwareOptimizer


class TestOptimizationSolver(unittest.TestCase, TestSolverAPI):

    def setUp(self):
        """Need to overwrite the setup and save the solver to be tested"""
        self.solver = SoftwareOptimizer()
