import unittest


from dwave_qasolver.solvers.simulated_annealing_solver import solve_ising_simulated_annealing


class TestSimulatedAnnealingAlgorithm(unittest.TestCase):
    def test_solve_basic(self):
        """really simple test"""

        # AND gate
        h = {0: -.5, 1: 0, 2: 1, 3: -.5}
        J = {(0, 2): -1, (1, 2): -1, (0, 3): .5, (1, 3): -1}

        solve_ising_simulated_annealing(h, J)
