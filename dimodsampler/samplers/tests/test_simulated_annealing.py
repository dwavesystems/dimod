import unittest

import itertools
import random

from dimodsampler import SimulatedAnnealingSolver
from dimodsampler.samplers.simulated_annealing import ising_simulated_annealing, greedy_coloring
# from dwave_qasolver.solvers.simulated_annealing_solver import solve_ising_simulated_annealing
# from dwave_qasolver.solvers.tests.generic_solver_tests import TestSolverAPI


# class TestSASolver(unittest.TestCase, TestSolverAPI):
#     def setUp(self):
#         self.solver = SimulatedAnnealingSolver()

#     def test_multiprocessing(self):

#         solver = self.solver

#         h = {0: -.5, 1: 0, 2: 1, 3: -.5}
#         J = {(0, 2): -1, (1, 2): -1, (0, 3): .5, (1, 3): -1}

#         solver.solve_ising(h, J, samples=100, multiprocessing=False)
#         solver.solve_ising(h, J, samples=100, multiprocessing=True)


class TestSimulatedAnnealingAlgorithm(unittest.TestCase):
    def test_solve_basic(self):
        """really simple test"""

        # AND gate
        h = {0: -.5, 1: 0, 2: 1, 3: -.5}
        J = {(0, 2): -1, (1, 2): -1, (0, 3): .5, (1, 3): -1}

        ising_simulated_annealing(h, J)

    def test_greedy_coloring(self):
        # set up an adjacency matrix

        N = 100  # number of nodes

        adj = {node: set() for node in range(N)}

        # add randomly approximately 5% of the edges
        for u, v in itertools.combinations(range(N), 2):
            if random.random() < .05:
                adj[u].add(v)
                adj[v].add(u)

        # add one disconnected node
        adj[N] = set()

        coloring, colors = greedy_coloring(adj)

        # we want to check that coloring and colors agree
        for v, c in coloring.items():
            self.assertIn(v, colors[c])
        for c, nodes in colors.items():
            for v in nodes:
                self.assertEqual(c, coloring[v])

        # next we want to make sure that no two neighbors share the same color
        for v in adj:
            for u in adj[v]:
                self.assertNotEqual(coloring[u], coloring[v])
