import unittest

import itertools
import random
import math

from dwave_qasolver import SpinResponse, BinaryResponse
from dwave_qasolver.solution_templates import DiscreteModelResponse


class TestDiscreteModelResponse(unittest.TestCase):
    """Tests on the DiscreteModelResponse"""

    def test_add_solution(self):
        """Tests for the add_solution method and the various retrieval methods."""
        response = DiscreteModelResponse()

        # ok, if we add a solution by itself, it should have an energy of NaN
        soln0 = {0: 0, 0: 1}
        response.add_solution(soln0, 1)

        # ok, we should have length 1 now, and the solution should be nan
        self.assertEqual(response[soln0], 1)
        self.assertEqual(len(response), 1)
        self.assertEqual(response.solutions(), [soln0])
        self.assertEqual(response.energies(), [1])
        self.assertTrue(all(en == 1 for en in response.energies_iter()))

        # now another solution
        soln1 = {0: 1, 1: 0}
        response.add_solution(soln1, -1)

        # so the energy for soln1 should be -1
        self.assertEqual(response[soln1], -1)
        self.assertEqual(len(response), 2)

        # now check the order of the solutions
        _check_solution_energy_order(self, response)

    def test_add_solutions_from(self):
        """Adding multiple solutions at once."""

        response = DiscreteModelResponse()

        soln0 = {0: 0, 0: 1}
        soln1 = {0: 1, 1: 0}

        response.add_solutions_from([soln0, soln1], [1, -1])

        self.assertEqual(response[soln1], -1)

        # now check the order of the solutions
        _check_solution_energy_order(self, response)


class TestBinaryResponse(unittest.TestCase):

    def test_add_solution(self):
        response = BinaryResponse()

        # TODO


def _check_solution_energy_order(testcase, response):
    """Check that the order of the solutions is from lowest to highest energy.
    """
    previous = -1 * float('inf')
    for en in response.energies():
        testcase.assertLessEqual(previous, en)
        previous = en


def _random_spin_solution(n):
    return {idx: random.choice((-1, 1)) for idx in range(n)}


def _random_bin_solution(n):
    return {idx: random.choice((0, 1)) for idx in range(n)}
