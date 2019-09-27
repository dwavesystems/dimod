import numpy as np
import dimod
import orang
import unittest

from dimod.roof_duality.extended_fix_variables import find_contractible_variables_naive, \
    find_contractible_variables_roof_duality, uncontract_solution, find_and_contract_all_variables_roof_duality, \
    find_and_contract_all_variables_naive


class TestLockableDict(unittest.TestCase):

    def test_find_contractible_variables_naive(self):

        J = {(u, v): 1 for (u, v) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]}
        J[(0, 3)] = -10
        bqm = dimod.BinaryQuadraticModel.from_ising({}, J)

        contractible_variables = find_contractible_variables_naive(bqm)
        self.assertEqual(contractible_variables, {(0, 3): True})

    def test_find_contractible_variables_roof_duality(self):

        J = {(u, v): 1 for (u, v) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]}
        J[(0, 3)] = -10
        bqm = dimod.BinaryQuadraticModel.from_ising({}, J)

        contractible_variables, fixed_variables = find_contractible_variables_roof_duality(bqm)
        self.assertEqual(contractible_variables, {(0, 3): True})
        self.assertEqual(fixed_variables, {})


    def test_find_and_contract_all_variables_roof_duality(self):

        bqm = dimod.BinaryQuadraticModel.from_ising({}, {(0, 1): -1, (1, 2): 3, (2, 3): 2})

        bqm2, variable_map, _ = find_and_contract_all_variables_roof_duality(bqm)
        variable_map_check = {0: (1, True), 1: (1, True), 2: (1, False), 3: (1, True)}
        self.assertEqual(variable_map, variable_map_check)
        bqm_check = dimod.BinaryQuadraticModel.from_ising({1: 0}, {}, -6)
        self.assertEqual(bqm2, bqm_check)

    def test_find_and_contract_all_variables_naive(self):

        J = {(u, v): 1 for (u, v) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]}
        J[(0, 3)] = -10
        bqm = dimod.BinaryQuadraticModel.from_ising({}, J)

        bqm2, variable_map = find_and_contract_all_variables_naive(bqm)
        variable_map_check = {i: (i, True) for i in [0, 1, 2, 4, 5]}
        variable_map_check[3] = (0, True)
        self.assertEqual(variable_map, variable_map_check)
        bqm_check = dimod.BinaryQuadraticModel.from_ising({},
                    {(u, v): 1 for (u, v) in [(0, 1), (0, 2), (1, 2), (0, 4), (0, 5), (4, 5)]},
                    -10)
        self.assertEqual(bqm2, bqm_check)

    def test_uncontract_solution(self):

        J = {(u, v): 1 for (u, v) in [(0, 1), (0, 2), (1, 2)]}
        J[(0, 3)] = -10
        bqm = dimod.BinaryQuadraticModel.from_ising({}, J)

        bqm2, variable_map, _ = find_and_contract_all_variables_roof_duality(bqm, sampling_mode=True)

        sampleset2 = dimod.ExactSolver().sample(bqm2)
        sampleset = uncontract_solution(sampleset2, variable_map)

        # check that energies in uncontracted and contracted bqms match up:
        dimod.testing.assert_response_energies(sampleset, bqm)

        # check that lowest energies satisfy variable map conditions:
        sampleset = dimod.ExactSolver().sample(bqm)
        min_energy = min(sampleset.data(['energy']))
        for sample, energy in sampleset.data(['sample', 'energy']):
            if energy == min_energy:
                for u, val in variable_map.items():
                    (v, uv_equal) = val
                    assert uv_equal == (sample[u] == sample[v])