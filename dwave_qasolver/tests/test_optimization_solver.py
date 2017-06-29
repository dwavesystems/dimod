import unittest

from dwave_qasolver.optimization_solver import SoftwareOptimizer


class TestOptimizationSolver(unittest.TestCase):

    def test_single_node_qubo(self):
        solver = SoftwareOptimizer()

        Q = {(0, 0): 1}

        solutions = solver.solve_qubo(Q)

        self.assertTrue(solutions[0] == {0: 0})

    def test_single_node_ising(self):
        solver = SoftwareOptimizer()

        h = {0: -1}
        J = {}

        solutions = solver.solve_ising(h, J)

        for soln in solutions:
            self.assertEqual(soln[0], 1)
