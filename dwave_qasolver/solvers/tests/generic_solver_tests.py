class TestSolverAPI:
    """Provides a series of generic API tests that all solvers should pass.
    """

    def test_single_node_qubo(self):
        """Create a QUBO that has only a single variable."""

        solver = self.solver

        Q = {(0, 0): 1}
        variables = [0]

        response = solver.solve_qubo(Q)
        self.check_response_form(response, variables)

    def test_single_node_ising(self):
        solver = self.solver

        h = {0: -1}
        J = {}
        variables = [0]

        response = solver.solve_ising(h, J)
        self.check_response_form(response, variables)

    def test_single_edge_qubo(self):
        solver = self.solver

        Q = {(0, 0): 0, (1, 1): 0, (0, 1): 1}
        variables = [0, 1]

        response = solver.solve_qubo(Q)
        self.check_response_form(response, variables)

    def test_single_edge_ising(self):
        solver = self.solver

        h = {0: 0, 1: 1}
        J = {(0, 1): 1}
        variables = [0, 1]

        response = solver.solve_ising(h, J)
        self.check_response_form(response, variables)

    def check_response_form(self, response, variables):
        for var in variables:
            for soln in response:
                self.assertIn(var, soln)
