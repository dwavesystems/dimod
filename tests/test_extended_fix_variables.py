import dimod
import unittest

try:
    from dimod import fix_variables
    from dimod.roof_duality.extended_fix_variables import find_contractible_variables_naive, \
        find_contractible_variables_roof_duality, uncontract_solution, find_and_contract_all_variables_roof_duality, \
        find_and_contract_all_variables_naive
except ImportError:
    cpp = False
else:
    cpp = True

@unittest.skipUnless(cpp, "no cpp extensions built")
class TestExtendedFixVariables(unittest.TestCase):

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


        bqm = dimod.BinaryQuadraticModel.from_ising({}, {(0, 1): -1, (1, 2): 1, (2, 3): 1})

        bqm2, variable_map, fixed_variables = find_and_contract_all_variables_roof_duality(bqm, sampling_mode=True)

        # 0, 1, and 3 should be contracted to the same value, 2 to the opposite value
        self.assertEqual(variable_map[1], variable_map[0])
        self.assertEqual(variable_map[3], variable_map[0])
        self.assertEqual(variable_map[2], (variable_map[0][0], False))
        bqm_check = dimod.BinaryQuadraticModel.from_ising({variable_map[0][0]: 0}, {}, -3)
        self.assertEqual(bqm2, bqm_check)

    def test_find_and_contract_all_variables_naive(self):

        J = {(u, v): 1 for (u, v) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]}
        J[(0, 3)] = -10
        bqm = dimod.BinaryQuadraticModel.from_ising({}, J)

        bqm2, variable_map = find_and_contract_all_variables_naive(bqm)

        # only 0 and 3 should be contracted
        self.assertEqual(variable_map[0], variable_map[3])
        for i in [1, 2, 4, 5]:
            self.assertEqual(variable_map[i], (i, True))
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