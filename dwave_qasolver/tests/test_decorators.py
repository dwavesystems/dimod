import unittest

from dwave_qasolver.solver_template import DiscreteModelSolver
from dwave_qasolver.solution_templates import SpinResponse, BooleanResponse
from dwave_qasolver.decorators import solve_qubo_api, qubo_index_labels


# we need a dummy solver to use to check the decorators
class DummySolver(DiscreteModelSolver):

    @solve_qubo_api()
    @qubo_index_labels()
    def solve_qubo(self, Q, detailed_solution=False, testcase=None):
        nodes = reduce(set.union, ({n0, n1} for n0, n1 in Q))

        # check that all of the nodes are index labelled
        testcase.assertTrue(all(isinstance(n, int) for n in nodes))

        return BooleanResponse([{node: 0 for node in nodes}])


class TestDecorators(unittest.TestCase):

    def test_index_relabel(self):
        Q = {('a', 'b'): 0}

        solns = DummySolver().solve_qubo(Q, testcase=self)
