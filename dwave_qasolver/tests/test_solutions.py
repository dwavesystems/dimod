import unittest

import itertools
import random

from dwave_qasolver import SpinResponse, BinaryResponse
from dwave_qasolver.solution_templates import DiscreteModelResponse


class TestSolutionsTemplates(unittest.TestCase):
    def test_spin_self_sorting(self):

        response = SpinResponse()

        nV = 10

        # add 100 random solutions one by one
        for __ in range(100):
            soln = _random_spin_solution(nV)
            en = random.random()
            response.add_solution(soln, en)

        self.assertEqual(len(response), 100,
                         'We added {} solutions, length should be {}'.format(100, len(response)))

        previous = -1 * float('inf')
        for en in response.energies():
            self.assertLessEqual(previous, en)
            previous = en


    def test_spin_calc_energy(self):
        response = SpinResponse()

        nV = 10

        h = {idx: random.uniform(-2, 2) for idx in range(nV)}
        J = {(n0, n1): random.uniform(-1, 1) for (n0, n1) in itertools.combinations(range(nV), 2)}

        for __ in range(100):
            soln = _random_spin_solution(nV)

            response.add_solution(soln, h=h, J=J)




def _random_spin_solution(n):
    return {idx: random.choice((-1, 1)) for idx in range(n)}


def _random_bin_solution(n):
    return {idx: random.choice((0, 1)) for idx in range(n)}
