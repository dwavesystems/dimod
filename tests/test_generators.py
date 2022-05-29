# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import unittest
import unittest.mock

import dimod
import itertools

import numpy as np

try:
    import networkx as nx
except ImportError:
    _networkx = False
else:
    _networkx = True


class TestRandomGNMRandomBQM(unittest.TestCase):
    def test_bias_generator(self):
        def gen(n):
            return np.full(n, 6)

        bqm = dimod.generators.gnm_random_bqm(10, 20, 'SPIN',
                                              bias_generator=gen)

        self.assertTrue(all(b == 6 for b in bqm.linear.values()))
        self.assertTrue(all(b == 6 for b in bqm.quadratic.values()))
        self.assertEqual(bqm.offset, 6)
        self.assertEqual(sorted(bqm.variables), list(range(10)))

    def test_labelled(self):
        bqm = dimod.generators.gnm_random_bqm('abcdef', 1, 'SPIN')
        self.assertEqual(bqm.shape, (6, 1))
        self.assertEqual(list(bqm.variables), list('abcdef'))

    def test_shape(self):
        n = 10
        for m in range(n*(n-1)//2):
            with self.subTest(shape=(n, m)):
                bqm = dimod.generators.gnm_random_bqm(n, m, 'SPIN')
                self.assertEqual(bqm.shape, (n, m))

    def test_deprecation(self):
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.generators.gnm_random_bqm(2, 1, "SPIN", cls=dimod.BinaryQuadraticModel)

class TestRandomGNPRandomBQM(unittest.TestCase):
    def test_bias_generator(self):
        def gen(n):
            return np.full(n, 6)

        bqm = dimod.generators.gnp_random_bqm(10, 1, 'SPIN',
                                              bias_generator=gen)

        self.assertTrue(all(b == 6 for b in bqm.linear.values()))
        self.assertTrue(all(b == 6 for b in bqm.quadratic.values()))
        self.assertEqual(bqm.offset, 6)
        self.assertEqual(sorted(bqm.variables), list(range(10)))

    def test_disconnected(self):
        for n in range(10):
            bqm = dimod.generators.gnp_random_bqm(n, 0, 'BINARY')
            self.assertEqual(bqm.shape, (n, 0))

        # p < 0 treated as 0
        for n in range(10):
            bqm = dimod.generators.gnp_random_bqm(n, -100, 'BINARY')
            self.assertEqual(bqm.shape, (n, 0))

    def test_empty(self):
        bqm = dimod.generators.gnp_random_bqm(0, 1, 'SPIN')
        self.assertEqual(bqm.shape, (0, 0))

    def test_fully_connected(self):
        for n in range(10):
            bqm = dimod.generators.gnp_random_bqm(n, 1, 'BINARY')
            self.assertEqual(bqm.shape, (n, n*(n-1)//2))

        # p > 1 treated as 1
        for n in range(10):
            bqm = dimod.generators.gnp_random_bqm(n, 100, 'BINARY')
            self.assertEqual(bqm.shape, (n, n*(n-1)//2))

    def test_labelled(self):
        bqm = dimod.generators.gnp_random_bqm('abcdef', 1, 'SPIN')
        self.assertEqual(bqm.shape, (6, 15))
        self.assertEqual(list(bqm.variables), list('abcdef'))

    def test_random_state(self):
        r = np.random.RandomState(16)

        bqm0 = dimod.generators.gnp_random_bqm(10, .6, 'SPIN', random_state=r)
        bqm1 = dimod.generators.gnp_random_bqm(10, .6, 'SPIN', random_state=r)

        # very small probability this returns True
        self.assertNotEqual(bqm0, bqm1)

        r = np.random.RandomState(16)

        bqm2 = dimod.generators.gnp_random_bqm(10, .6, 'SPIN', random_state=r)
        bqm3 = dimod.generators.gnp_random_bqm(10, .6, 'SPIN', random_state=r)

        self.assertNotEqual(bqm2, bqm3)

        self.assertEqual(bqm0, bqm2)
        self.assertEqual(bqm1, bqm3)

    def test_seed(self):
        bqm0 = dimod.generators.gnp_random_bqm(10, .6, 'SPIN', random_state=5)
        bqm1 = dimod.generators.gnp_random_bqm(10, .6, 'SPIN', random_state=5)

        self.assertEqual(bqm0, bqm1)

    def test_singleton(self):
        bqm = dimod.generators.gnp_random_bqm(1, 1, 'SPIN')
        self.assertEqual(bqm.shape, (1, 0))

        bqm = dimod.generators.gnp_random_bqm(1, 0, 'SPIN')
        self.assertEqual(bqm.shape, (1, 0))

    def test_deprecation(self):
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.generators.gnp_random_bqm(2, 1, "SPIN", cls=dimod.BinaryQuadraticModel)

class TestRandomUniform(unittest.TestCase):
    def test_singleton(self):
        bqm = dimod.generators.random.uniform(1, dimod.BINARY)

        # should have a single node
        self.assertEqual(len(bqm), 1)
        self.assertIn(0, bqm.variables)

    def test_empty(self):
        bqm = dimod.generators.random.uniform(0, dimod.BINARY)

        # should have a single node
        self.assertEqual(len(bqm), 0)

    def test_seed(self):
        bqm0 = dimod.generators.random.uniform(1, dimod.BINARY, seed=506)
        bqm1 = dimod.generators.random.uniform(1, dimod.BINARY, seed=506)

        self.assertEqual(bqm0, bqm1)

        bqm2 = dimod.generators.random.uniform(1, dimod.BINARY, seed=123)

        self.assertNotEqual(bqm2, bqm1)

    def test_deprecation(self):
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.generators.random.uniform(2, "SPIN", cls=dimod.BinaryQuadraticModel)

class TestRandomRandint(unittest.TestCase):
    def test_singleton(self):
        bqm = dimod.generators.random.randint(1, dimod.BINARY)

        # should have a single node
        self.assertEqual(len(bqm), 1)
        self.assertIn(0, bqm.variables)

    def test_empty(self):
        bqm = dimod.generators.random.randint(0, dimod.BINARY)

        # should have a single node
        self.assertEqual(len(bqm), 0)

    def test_seed(self):
        bqm0 = dimod.generators.random.randint(100, dimod.BINARY, seed=506)
        bqm1 = dimod.generators.random.randint(100, dimod.BINARY, seed=506)

        self.assertEqual(bqm0, bqm1)

        bqm2 = dimod.generators.random.randint(100, dimod.BINARY, seed=123)

        self.assertNotEqual(bqm2, bqm1)

    def test_deprecation(self):
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.generators.random.randint(3, dimod.BINARY, cls=dimod.BinaryQuadraticModel)

class TestRandomRanR(unittest.TestCase):
    def test_singleton(self):
        bqm = dimod.generators.random.ran_r(1, 1)

        # should have a single node
        self.assertEqual(len(bqm), 1)
        self.assertIn(0, bqm.variables)

    def test_empty(self):
        bqm = dimod.generators.random.ran_r(1, 0)

        # should have no nodes
        self.assertEqual(len(bqm), 0)

    def test_seed(self):
        bqm0 = dimod.generators.random.ran_r(3, 100, seed=506)
        bqm1 = dimod.generators.random.ran_r(3, 100, seed=506)

        self.assertEqual(bqm0, bqm1)

        bqm2 = dimod.generators.random.ran_r(3, 100, seed=123)

        self.assertNotEqual(bqm2, bqm1)

    def test_values(self):
        bqm = dimod.generators.random.ran_r(5, 10)

        self.assertFalse(all(bqm.linear.values()))
        self.assertTrue(all(val != 0 for val in bqm.quadratic.values()))
        self.assertTrue(all(val <= 5 for val in bqm.quadratic.values()))
        self.assertTrue(all(val >= -5 for val in bqm.quadratic.values()))

    def test_deprecation(self):
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.generators.random.ran_r(2, 3, cls=dimod.BinaryQuadraticModel)

class TestChimeraAnticluster(unittest.TestCase):
    def test_singletile(self):
        bqm = dimod.generators.chimera_anticluster(1)

        self.assertEqual(len(bqm), 8)
        self.assertEqual(len(bqm.quadratic), 16)
        for i in range(4):
            for j in range(4, 8):
                self.assertIn(i, bqm.adj)
                self.assertIn(j, bqm.adj[i])
                self.assertIn(bqm.adj[i][j], (-1, 1))

    @unittest.skipUnless(_networkx, "no networkx installed")
    def test_multitile(self):
        bqm = dimod.generators.chimera_anticluster(2, multiplier=4)

        self.assertEqual(set(bqm.linear.values()), {0})

        for (u, v), bias in bqm.quadratic.items():
            if u // 8 == v // 8:
                self.assertIn(bias, {-1, 1})
            else:
                self.assertIn(bias, {-4, 4})

    def test_singletile_subgraph(self):
        subgraph = ([0, 1, 2, 3, 4, 5, 6],
                    [(0, 4), (0, 5), (0, 6),
                     (1, 4), (1, 5), (1, 6),
                     (2, 4), (2, 5),
                     (3, 4), (3, 5), (3, 6)])
        bqm = dimod.generators.chimera_anticluster(1, subgraph=subgraph)

        self.assertEqual(len(bqm), 7)
        self.assertEqual(len(bqm.quadratic), 11)

        nodes, edges = subgraph
        for v in nodes:
            self.assertEqual(bqm.linear[v], 0)
        for u, v in edges:
            self.assertIn(bqm.quadratic[(u, v)], {-1, 1})

    def test_singletile_not_subgraph(self):
        subgraph = ([0, 'a'], [(0, 1)])

        with self.assertRaises(ValueError):
            dimod.generators.chimera_anticluster(1, subgraph=subgraph)

    def test_seed(self):
        bqm0 = dimod.generators.chimera_anticluster(2, 1, 3, seed=506)
        bqm1 = dimod.generators.chimera_anticluster(2, 1, 3, seed=506)

        self.assertEqual(bqm0, bqm1)

        bqm2 = dimod.generators.chimera_anticluster(2, 1, 3, seed=123)

        self.assertNotEqual(bqm2, bqm1)

    def test_empty(self):
        bqm = dimod.generators.chimera_anticluster(0)

        # should have a single node
        self.assertEqual(len(bqm), 0)

    def test_subgraph_edgelist(self):
        # c2 edgelist with some random ones removed
        edgelist = [(0, 4), (0, 5), (0, 6), (0, 7), (0, 16), (4, 1), (4, 2),
                    (4, 3), (4, 12), (5, 1), (5, 2), (5, 3), (5, 13), (6, 1),
                    (6, 2), (6, 3), (6, 14), (7, 1), (7, 2), (7, 3), (7, 15),
                    (1, 17), (2, 18), (3, 19), (16, 20), (16, 21), (16, 22),
                    (16, 23), (20, 17), (20, 18), (20, 19), (20, 28), (21, 17),
                    (21, 18), (21, 19), (21, 29), (22, 17), (22, 18), (22, 19),
                    (22, 30), (23, 17), (23, 18), (23, 19), (23, 31), (8, 12),
                    (8, 13), (8, 14), (8, 15), (12, 9), (12, 10),
                    (12, 11), (13, 9), (13, 10), (13, 11), (14, 9), (14, 10),
                    (14, 11), (15, 9), (15, 10), (15, 11), (9, 25), (10, 26),
                    (11, 27), (24, 28), (24, 29), (24, 30), (24, 31), (28, 25),
                    (28, 26), (28, 27), (29, 25), (29, 26), (29, 27), (30, 25),
                    (30, 26), (30, 27), (31, 25), (31, 27)]

        bqm = dimod.generators.chimera_anticluster(2, subgraph=edgelist)

    def test_deprecation(self):
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.generators.chimera_anticluster(0, cls=6)


@unittest.skipUnless(_networkx, "no networkx installed")
class TestFCL(unittest.TestCase):

    def setUp(self):
        self.G = nx.Graph()
        for u in range(4):
            for v in range(4, 8):
                self.G.add_edge(u, v)

    def test_singletile(self):
        G = self.G
        bqm = dimod.generators.frustrated_loop(G, 10)

        self.assertEqual(len(bqm), 8)
        self.assertEqual(len(bqm.quadratic), 16)
        for i in range(4):
            for j in range(4, 8):
                self.assertIn(i, bqm.adj)
                self.assertIn(j, bqm.adj[i])

    def test_seed(self):
        G = self.G

        bqm0 = dimod.generators.frustrated_loop(G, 10, seed=506)
        bqm1 = dimod.generators.frustrated_loop(G, 10, seed=506)

        self.assertEqual(bqm0, bqm1)

        bqm2 = dimod.generators.frustrated_loop(G, 10, seed=123)

        self.assertNotEqual(bqm2, bqm1)

    def test_too_few_good_cycles(self):

        with self.assertRaises(RuntimeError):
            dimod.generators.frustrated_loop(4, 3, R=1, seed=1)

    def test_planted_solution(self):
        G = self.G

        planted = {v:v%2*2-1 for v in G}
        bqm = dimod.generators.frustrated_loop(G, 10, planted_solution=planted)

        inv_solution = {k:-v for k,v in planted.items()}
        self.assertEqual(bqm.energy(planted),bqm.energy(inv_solution))

        all_ones = {v:1 for v in G}
        self.assertNotEqual(bqm.energy(planted),bqm.energy(all_ones))

class TestCombinations(unittest.TestCase):

    def check_combinations(self, variables, k, bqm, strength):
        self.assertEqual(len(bqm), len(variables))

        sampleset = dimod.ExactSolver().sample(bqm)

        for sample, energy in sampleset.data(['sample', 'energy']):
            if sum(val == 1 for val in sample.values()) == k:
                self.assertEqual(energy, 0)
            else:
                self.assertGreaterEqual(energy, strength)

    def test_2_choose_1(self):
        bqm = dimod.generators.combinations(2, 1)

        self.assertIs(bqm.vartype, dimod.BINARY)
        self.check_combinations(range(2), 1, bqm, 1)

    def test_5_choose_3(self):
        bqm = dimod.generators.combinations('abcde', 3)

        self.assertIs(bqm.vartype, dimod.BINARY)
        self.check_combinations('abcde', 3, bqm, 1)

    def test_5_choose_3_spin(self):
        bqm = dimod.generators.combinations('abcde', 3, vartype='SPIN')

        self.assertIs(bqm.vartype, dimod.SPIN)
        self.check_combinations('abcde', 3, bqm, 1)

    def test_5_choose_3_strength_4(self):
        bqm = dimod.generators.combinations('abcde', 3, strength=4.)

        self.assertIs(bqm.vartype, dimod.BINARY)
        self.check_combinations('abcde', 3, bqm, 4)

    def test_3_choose_0(self):
        bqm = dimod.generators.combinations(3, 0)

        self.assertIs(bqm.vartype, dimod.BINARY)
        self.check_combinations(range(3), 0, bqm, 1)


class TestAntiCrossing(unittest.TestCase):

    def test_wrong_size(self):
        with self.assertRaises(ValueError):
            bqm = dimod.generators.anti_crossing_clique(4)

        with self.assertRaises(ValueError):
            bqm = dimod.generators.anti_crossing_clique(7)

        with self.assertRaises(ValueError):
            bqm = dimod.generators.anti_crossing_loops(6)

        with self.assertRaises(ValueError):
            bqm = dimod.generators.anti_crossing_loops(9)

    def test_fixed_size(self):
        bqm = dimod.generators.anti_crossing_loops(8)
        bqm_fixed = dimod.BinaryQuadraticModel.from_ising({0: 0, 2: 0, 4: -1, 6: -1, 1: 1, 3: 1, 5: -1, 7: -1},
                                               {(0, 1): -1, (2, 3): -1, (0, 4): -1,
                                                (2, 6): -1, (1, 3): -1, (1, 5): -1, (3, 7): -1})
        self.assertEqual(bqm, bqm_fixed)

        bqm = dimod.generators.anti_crossing_clique(8)
        bqm_fixed = dimod.BinaryQuadraticModel.from_ising({0: 1, 4: -1, 1: 0, 5: -1, 2: 1, 6: -1, 3: 1, 7: -1},
                                               {(0, 1): -1, (0, 2): -1, (0, 3): -1, (0, 4): -1, (1, 2): -1, (1, 3): -1,
                                                (1, 5): -1, (2, 3): -1, (2, 6): -1, (3, 7): -1})
        self.assertEqual(bqm, bqm_fixed)


class TestDoped(unittest.TestCase):

    def test_wrong_doping(self):
        with self.assertRaises(ValueError):
            bqm0 = dimod.generators.random.doped(3, 100)

    def test_correct_seed(self):
        bqm0 = dimod.generators.random.doped(0.5, 100, seed=506)
        bqm1 = dimod.generators.random.doped(0.5, 100, seed=506)

        self.assertEqual(bqm0, bqm1)

        bqm2 = dimod.generators.random.doped(0.5, 100, seed=123)

        self.assertNotEqual(bqm2, bqm1)

    def test_correct_ratio(self):
        bqm = dimod.generators.random.doped(0.3, 100, seed=506)
        total = len(bqm.quadratic)
        afm = sum([val == 1 for val in bqm.quadratic.values()])
        self.assertAlmostEqual(afm / total, 0.3, places=2)

    def test_correct_ratio_fm(self):
        bqm = dimod.generators.random.doped(0.3, 100, seed=506, fm=False)
        total = len(bqm.quadratic)
        fm = sum([val == -1 for val in bqm.quadratic.values()])
        self.assertAlmostEqual(fm / total, 0.3, places=2)

    def test_deprecation(self):
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.generators.random.doped(0.3, 10, cls=dimod.BinaryQuadraticModel)


class TestKnapsack(unittest.TestCase):

    def test_model(self):
        num_items = 10
        cqm = dimod.generators.random_knapsack(num_items=num_items)
        self.assertEqual(len(cqm.variables), num_items)
        self.assertEqual(len(cqm.constraints), 1)

    def test_infeasible(self):
        num_items = 10
        cqm = dimod.generators.random_knapsack(num_items=num_items)

        # create an infeasible state, by selecting all the items
        x = {i: 1 for i in cqm.variables}
        lhs = cqm.constraints['capacity'].lhs.energy(x)
        self.assertGreater(lhs, 0)

    def test_feasible(self):
        num_items = 10
        cqm = dimod.generators.random_knapsack(num_items=num_items)

        # create feasible state, by not selecting any item
        x = {i: 0 for i in cqm.variables}
        lhs = cqm.constraints['capacity'].lhs.energy(x)
        self.assertLessEqual(lhs, 0)


class TestBinPacking(unittest.TestCase):

    def test_model(self):
        num_items = 10
        cqm = dimod.generators.random_bin_packing(num_items=num_items)
        self.assertEqual(len(cqm.variables), num_items*(num_items+1))
        self.assertEqual(len(cqm.constraints), 2*num_items)

    def test_infeasible(self):
        num_items = 10
        cqm = dimod.generators.random_bin_packing(num_items=num_items)

        for i in range(num_items):
            x = {'x_{}_{}'.format(i, j): 1 for j in range(num_items)}
            lhs = cqm.constraints['item_placing_{}'.format(i)].lhs.energy(x)
            self.assertGreater(lhs, 0)

        for i in range(num_items):
            x = {'x_{}_{}'.format(j, i): 1 for j in range(num_items)}
            x['y_{}'.format(i)] = 1
            lhs = cqm.constraints['capacity_bin_{}'.format(i)].lhs.energy(x)
            self.assertGreater(lhs, 0)

    def test_feasible(self):
        num_items = 10
        cqm = dimod.generators.random_bin_packing(num_items=num_items)

        for i in range(num_items):
            x = {'x_{}_{}'.format(i, j): 0 for j in range(1, num_items)}
            x['x_{}_0'.format(i)] = 1
            lhs = cqm.constraints['item_placing_{}'.format(i)].lhs.energy(x)
            self.assertLessEqual(lhs, 0)

        for i in range(num_items):
            x = {'x_{}_{}'.format(j, i): 0 for j in range(num_items)}
            x['y_{}'.format(i)] = 1
            lhs = cqm.constraints['capacity_bin_{}'.format(i)].lhs.energy(x)
            self.assertLessEqual(lhs, 0)


class TestMultiKnapsack(unittest.TestCase):

    def test_model(self):
        num_items = 20
        num_bins = 10
        cqm = dimod.generators.random_multi_knapsack(num_items=num_items, num_bins=num_bins)
        self.assertEqual(len(cqm.variables), num_items*num_bins)
        self.assertEqual(len(cqm.constraints), num_bins+num_items)


class TestGates(unittest.TestCase):
    def test_gates_no_aux(self):

        def halfadder(a, b, sum_, carry):
            return sum_ == (a ^ b) and carry == (a and b)

        def fulladder(a, b, c, sum_, carry):
            t = a + b + c
            return carry == (t >= 2) and sum_ == (t % 2)

        # the gates with no auxiliary variable are simple
        gates = dict(AND=(dimod.generators.and_gate, lambda a, b, p: (a and b) == p, 3),
                     OR=(dimod.generators.or_gate, lambda a, b, p: (a or b) == p, 3),
                     HA=(dimod.generators.halfadder_gate, halfadder, 4),
                     FA=(dimod.generators.fulladder_gate, fulladder, 5),
                     )

        for label, (generator, func, num_vars) in gates.items():
            # test with default strength
            with self.subTest(gate=label):
                bqm = generator(*range(num_vars))

                sampleset = dimod.ExactSolver().sample(bqm)
                for sample, energy in sampleset.data(['sample', 'energy']):
                    if func(*(sample[v] for v in range(num_vars))):
                        self.assertEqual(energy, 0)
                    else:
                        self.assertGreaterEqual(energy, 1)

                # also check that the gap is 1
                self.assertIn(1, set(sampleset.record.energy))

            with self.subTest(gate=label, strength=7.5):
                bqm = generator(*range(num_vars), strength=7.5)

                sampleset = dimod.ExactSolver().sample(bqm)
                for sample, energy in sampleset.data(['sample', 'energy']):
                    if func(*(sample[v] for v in range(num_vars))):
                        self.assertEqual(energy, 0)
                    else:
                        self.assertGreaterEqual(energy, 7.5)

                # also check that the gap is 7.5
                self.assertIn(7.5, set(sampleset.record.energy))

    def test_xor_gate(self):
        bqm = dimod.generators.xor_gate('a', 'b', 'p', 'x')
        self.assertEqual(list(bqm.variables), list('abpx'))
        sampleset = dimod.ExactSolver().sample(bqm)

        # we have an aux variable, so let's just check the ground states
        # explicitly
        self.assertEqual(bqm.energy({'a': 0, 'b': 0, 'p': 0, 'x': 0}), 0)
        self.assertEqual(bqm.energy({'a': 0, 'b': 1, 'p': 1, 'x': 0}), 0)
        self.assertEqual(bqm.energy({'a': 1, 'b': 0, 'p': 1, 'x': 0}), 0)
        self.assertEqual(bqm.energy({'a': 1, 'b': 1, 'p': 0, 'x': 1}), 0)

        for sample, energy in sampleset.data(['sample', 'energy']):
            if sample['p'] != (sample['a'] ^ sample['b']):
                self.assertGreaterEqual(energy, 1)

        # also check that the gap is 1
        self.assertIn(1, set(sampleset.record.energy))

    def test_xor_gate_strength(self):
        bqm = dimod.generators.xor_gate('a', 'b', 'p', 'x', strength=7.5)
        self.assertEqual(list(bqm.variables), list('abpx'))
        sampleset = dimod.ExactSolver().sample(bqm)

        # we have an aux variable, so let's just check the ground states
        # explicitly
        self.assertEqual(bqm.energy({'a': 0, 'b': 0, 'p': 0, 'x': 0}), 0)
        self.assertEqual(bqm.energy({'a': 0, 'b': 1, 'p': 1, 'x': 0}), 0)
        self.assertEqual(bqm.energy({'a': 1, 'b': 0, 'p': 1, 'x': 0}), 0)
        self.assertEqual(bqm.energy({'a': 1, 'b': 1, 'p': 0, 'x': 1}), 0)

        for sample, energy in sampleset.data(['sample', 'energy']):
            if sample['p'] != (sample['a'] ^ sample['b']):
                self.assertGreaterEqual(energy, 7.5)

        # also check that the gap is 7.5
        self.assertIn(7.5, set(sampleset.record.energy))

    def test_multiplication_circuit(self):

        # Verify correct variables for 3x3 circuit
        mc3_vars = {
            'a0', 'a1', 'a2',
            'b0', 'b1', 'b2',
            'and0,1', 'and0,2', 'and1,0', 'and1,1', 'and1,2', 'and2,0', 'and2,1', 'and2,2',
            'sum1,1', 'sum1,2',
            'carry1,0', 'carry1,1', 'carry1,2', 'carry2,0', 'carry2,1',
            'p0', 'p1', 'p2', 'p3', 'p4', 'p5',
        }
        bqm = dimod.generators.multiplication_circuit(3)
        self.assertEqual(set(bqm.variables), mc3_vars)

        # Verify correct variables for 2x3 circuit
        mc2_3_vars = {
            'a0', 'a1',
            'b0', 'b1', 'b2',
            'and0,1', 'and0,2', 'and1,0', 'and1,1', 'and1,2',
            'carry1,0', 'carry1,1',
            'p0', 'p1', 'p2', 'p3', 'p4'
        }
        bqm = dimod.generators.multiplication_circuit(2, 3)
        self.assertEqual(set(bqm.variables), mc2_3_vars)

        # Verify correct variables for 3x2 circuit
        mc3_2_vars = {
            'a0', 'a1', 'a2',
            'b0', 'b1',
            'and0,1', 'and1,0', 'and1,1', 'and2,0', 'and2,1',
            'sum1,1',
            'carry1,0', 'carry1,1', 'carry2,0',
            'p0', 'p1', 'p2', 'p3', 'p4',
        }
        bqm = dimod.generators.multiplication_circuit(3, 2)
        self.assertEqual((set(bqm.variables)), mc3_2_vars)

        bqm_9 = bqm.copy()
        for fixed_var, fixed_val in {'a0': 1, 'a1': 1, 'a2':0, 'b0': 1, 'b1': 1 }.items():
            bqm_9.fix_variable(fixed_var, fixed_val)
        best = dimod.ExactSolver().sample(bqm_9).first
        p = [best.sample['p0'], best.sample['p1'], best.sample['p2'],
             best.sample['p3']]
        self.assertEqual(p, [1, 0, 0, 1])

        # Verify correct factoring/multiplication for 2x2 circuit
        bqm = dimod.generators.multiplication_circuit(2)

        bqm_6 = bqm.copy()
        for fixed_var, fixed_val in {'p0': 0, 'p1': 1, 'p2':1}.items():
            bqm_6.fix_variable(fixed_var, fixed_val)
        best = dimod.ExactSolver().sample(bqm_6).first
        ab = [best.sample['a0'], best.sample['a1'], best.sample['b0'], best.sample['b1']]
        self.assertTrue(ab == [0, 1, 1, 1] or ab == [1, 1, 0, 1])

        bqm_4 = bqm.copy()
        for fixed_var, fixed_val in {'p0': 0, 'p1': 0, 'p2':1}.items():
            bqm_4.fix_variable(fixed_var, fixed_val)
        best = dimod.ExactSolver().sample(bqm_4).first
        ab = [best.sample['a0'], best.sample['a1'], best.sample['b0'], best.sample['b1']]
        self.assertEqual(ab, [0, 1, 0, 1])

        for fixed_var, fixed_val in {'a0': 1, 'a1': 1, 'b0':1, 'b1':1}.items():
            bqm.fix_variable(fixed_var, fixed_val)
        best = dimod.ExactSolver().sample(bqm).first
        p = [best.sample['p0'], best.sample['p1'], best.sample['p2'],
             best.sample['p3']]
        self.assertEqual(p, [1, 0, 0, 1])

class TestInteger(unittest.TestCase):
    def test_exceptions(self):
        with self.assertRaises(ValueError):
            dimod.generators.binary_encoding('v', .9)

    def test_values(self):
        sampler = dimod.ExactSolver()
        for ub in [3, 7, 15, 2, 4, 8, 13, 11, 5]:
            with self.subTest(upper_bound=ub):
                bqm = dimod.generators.binary_encoding('v', ub)

                sampleset = sampler.sample(bqm)

                # we see all possible energies in range
                self.assertEqual(set(sampleset.record.energy), set(range(ub+1)))

                # the variable labels correspond to the energy
                for sample, energy in sampleset.data(['sample', 'energy']):
                    self.assertEqual(sum(v[1]*val for v, val in sample.items()), energy)


class TestIndependentSet(unittest.TestCase):
    def test_edges(self):
        bqm = dimod.generators.independent_set([(0, 1), (1, 2), (0, 2)])
        self.assertEqual(bqm.linear, {0: 0.0, 1: 0.0, 2: 0.0})
        self.assertEqual(bqm.quadratic, {(1, 0): 1.0, (2, 0): 1.0, (2, 1): 1.0})
        self.assertEqual(bqm.offset, 0)
        self.assertIs(bqm.vartype, dimod.BINARY)

    @unittest.skipUnless(_networkx, "no networkx installed")
    def test_edges_networkx(self):
        G = nx.complete_graph(3)
        bqm = dimod.generators.independent_set(G.edges)
        self.assertEqual(bqm.linear, {0: 0.0, 1: 0.0, 2: 0.0})
        self.assertEqual(bqm.quadratic, {(1, 0): 1.0, (2, 0): 1.0, (2, 1): 1.0})
        self.assertEqual(bqm.offset, 0)
        self.assertIs(bqm.vartype, dimod.BINARY)

    def test_edges_and_nodes(self):
        bqm = dimod.generators.independent_set([(0, 1), (1, 2)], [0, 1, 2, 3])
        self.assertEqual(bqm.linear, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
        self.assertEqual(bqm.quadratic, {(1, 0): 1.0, (2, 1): 1.0})
        self.assertEqual(bqm.offset, 0)
        self.assertIs(bqm.vartype, dimod.BINARY)

    @unittest.skipUnless(_networkx, "no networkx installed")
    def test_edges_and_nodes_networkx(self):
        G = nx.complete_graph(3)
        G.add_node(3)
        bqm = dimod.generators.independent_set(G.edges, G.nodes)
        self.assertEqual(bqm.linear, {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0})
        self.assertEqual(bqm.quadratic, {(1, 0): 1.0, (2, 0): 1.0, (2, 1): 1.0})
        self.assertEqual(bqm.offset, 0)
        self.assertIs(bqm.vartype, dimod.BINARY)

    def test_empty(self):
        self.assertEqual(dimod.generators.independent_set([]).shape, (0, 0))
        self.assertEqual(dimod.generators.independent_set([], []).shape, (0, 0))

    @unittest.skipUnless(_networkx, "no networkx installed")
    def test_empty_networkx(self):
        G = nx.Graph()
        self.assertEqual(dimod.generators.independent_set(G.edges).shape, (0, 0))
        self.assertEqual(dimod.generators.independent_set(G.edges, G.nodes).shape, (0, 0))

    def test_functional(self):
        edges = [(1, 0), (3, 0), (3, 1), (4, 1), (4, 3), (5, 1), (5, 3),
                 (6, 1), (7, 0), (7, 1), (7, 5), (7, 6), (8, 1), (8, 6),
                 (8, 7), (9, 2), (9, 4), (9, 5), (9, 8)]

        bqm = dimod.generators.independent_set(edges)

        self.assertEqual(set(frozenset(e) for e in edges),
                         set(frozenset(i) for i in bqm.quadratic))

        energies = set()
        for sample, energy in dimod.ExactSolver().sample(bqm).data(['sample', 'energy']):
            if energy:
                self.assertGreaterEqual(energy, 1)
                self.assertTrue(any(sample[u] and sample[v] for u, v in edges))
            else:
                self.assertFalse(any(sample[u] and sample[v] for u, v in edges))
            energies.add(energy)
        self.assertIn(1, energies)


class TestMaximumIndependentSet(unittest.TestCase):
    def test_edges(self):
        bqm = dimod.generators.maximum_independent_set([(0, 1), (1, 2), (0, 2)])
        self.assertEqual(bqm.linear, {0: -1, 1: -1, 2: -1})
        self.assertEqual(bqm.quadratic, {(1, 0): 2, (2, 0): 2, (2, 1): 2})
        self.assertEqual(bqm.offset, 0)
        self.assertIs(bqm.vartype, dimod.BINARY)

    @unittest.skipUnless(_networkx, "no networkx installed")
    def test_edges_networkx(self):
        G = nx.complete_graph(3)
        bqm = dimod.generators.maximum_independent_set(G.edges)
        self.assertEqual(bqm.linear, {0: -1, 1: -1, 2: -1})
        self.assertEqual(bqm.quadratic, {(1, 0): 2, (2, 0): 2, (2, 1): 2})
        self.assertEqual(bqm.offset, 0)
        self.assertIs(bqm.vartype, dimod.BINARY)

    def test_edges_and_nodes(self):
        bqm = dimod.generators.maximum_independent_set([(0, 1), (1, 2)], [0, 1, 2, 3])
        self.assertEqual(bqm.linear, {0: -1, 1: -1, 2: -1, 3: -1})
        self.assertEqual(bqm.quadratic, {(1, 0): 2, (2, 1): 2})
        self.assertEqual(bqm.offset, 0)
        self.assertIs(bqm.vartype, dimod.BINARY)

    @unittest.skipUnless(_networkx, "no networkx installed")
    def test_edges_and_nodes_networkx(self):
        G = nx.complete_graph(3)
        G.add_node(3)
        bqm = dimod.generators.maximum_independent_set(G.edges, G.nodes)
        self.assertEqual(bqm.linear, {0: -1, 1: -1, 2: -1, 3: -1})
        self.assertEqual(bqm.quadratic, {(1, 0): 2, (2, 0): 2, (2, 1): 2})
        self.assertEqual(bqm.offset, 0)
        self.assertIs(bqm.vartype, dimod.BINARY)

    def test_empty(self):
        self.assertEqual(dimod.generators.maximum_independent_set([]).shape, (0, 0))
        self.assertEqual(dimod.generators.maximum_independent_set([], []).shape, (0, 0))

    @unittest.skipUnless(_networkx, "no networkx installed")
    def test_empty_networkx(self):
        G = nx.Graph()
        self.assertEqual(dimod.generators.maximum_independent_set(G.edges).shape, (0, 0))
        self.assertEqual(dimod.generators.maximum_independent_set(G.edges, G.nodes).shape, (0, 0))

    def test_functional(self):
        edges = [(1, 0), (3, 0), (3, 1), (4, 1), (4, 3), (5, 1), (5, 3),
                 (6, 1), (7, 0), (7, 1), (7, 5), (7, 6), (8, 1), (8, 6),
                 (8, 7), (9, 2), (9, 4), (9, 5), (9, 8)]

        bqm = dimod.generators.maximum_independent_set(edges)

        self.assertEqual(set(frozenset(e) for e in edges),
                         set(frozenset(i) for i in bqm.quadratic))

        data = dimod.ExactSolver().sample(bqm).data(['sample', 'energy'])

        first = next(data)

        self.assertFalse(any(first.sample[u] and first.sample[v] for u, v in edges))

        for sample, energy in data:
            if energy == first.energy:
                # it's unique
                self.assertEqual(sum(sample.values()), sum(first.sample.values()))
            elif any(sample[u] and sample[v] for u, v in edges):
                self.assertGreaterEqual(energy, first.energy)
            else:
                self.assertLess(sum(sample.values()), sum(first.sample.values()))


class TestMaximumWeightIndependentSet(unittest.TestCase):
    def test_default_weight(self):
        edges = [(0, 1), (1, 2)]
        nodes = [(1, .5)]

        bqm = dimod.generators.maximum_weight_independent_set(edges, nodes)

        self.assertEqual(bqm.linear, {0: -1, 1: -.5, 2: -1})

    def test_functional(self):
        edges = [(0, 1), (1, 2)]
        nodes = [(0, .25), (1, .5), (2, .25)]

        bqm = dimod.generators.maximum_weight_independent_set(edges, nodes)

        sampleset = dimod.ExactSolver().sample(bqm)

        configs = {tuple(sample[v] for v in range(3)) for sample in sampleset.lowest().samples()}
        self.assertEqual(configs, {(0, 1, 0), (1, 0, 1)})

    @unittest.skipUnless(_networkx, "no networkx installed")
    def test_functional_networkx(self):
        G = nx.complete_graph(3)
        G.add_nodes_from([0, 2], weight=.5)
        G.add_node(1, weight=1)

        bqm = dimod.generators.maximum_weight_independent_set(G.edges, G.nodes('weight'))

        self.assertEqual(bqm.linear, {0: -0.5, 1: -1.0, 2: -0.5})
        self.assertEqual(bqm.quadratic, {(1, 0): 2.0, (2, 0): 2.0, (2, 1): 2.0})
        self.assertEqual(bqm.offset, 0)
        self.assertIs(bqm.vartype, dimod.BINARY)

    def test_empty(self):
        self.assertEqual(dimod.generators.maximum_weight_independent_set([]).shape, (0, 0))
        self.assertEqual(dimod.generators.maximum_weight_independent_set([], []).shape, (0, 0))

    @unittest.skipUnless(_networkx, "no networkx installed")
    def test_empty_networkx(self):
        G = nx.Graph()
        self.assertEqual(dimod.generators.maximum_weight_independent_set(G.edges).shape, (0, 0))
        self.assertEqual(
            dimod.generators.maximum_weight_independent_set(G.edges, G.nodes('weight')).shape,
            (0, 0))


class TestSatisfiability(unittest.TestCase):
    def test_empty(self):
        for n in range(3):
            with self.assertRaises(ValueError):
                dimod.generators.random_nae3sat(n, 0)
        for n in range(4):
            with self.assertRaises(ValueError):
                dimod.generators.random_2in4sat(n, 0)

    def test_nae3sat(self):

        # we want the clause to have no negations
        class MyGen(np.random.Generator):
            def __init__(self):
                super().__init__(np.random.PCG64(5))

            def integers(self, *args, **kwargs):
                return np.asarray([1, 1, 1])

        seed = MyGen()

        for trial in range(10):
            with self.subTest(trial=trial):
                bqm = dimod.generators.random_nae3sat(3, 1, seed=seed)

                # in the ground state they should not be all equal
                ss = dimod.ExactSolver().sample(bqm)
                self.assertEqual(set(ss.first.sample.values()), {-1, +1})

    def test_2in4sat(self):

        # we want the clause to have no negations
        class MyGen(np.random.Generator):
            def __init__(self):
                super().__init__(np.random.PCG64(5))

            def integers(self, *args, **kwargs):
                return np.asarray([1, 1, 1, 1])

        seed = MyGen()

        for trial in range(10):
            with self.subTest(trial=trial):
                bqm = dimod.generators.random_2in4sat(4, 1, seed=seed)

                # in the ground state they should not be all equal
                ss = dimod.ExactSolver().sample(bqm)
                self.assertEqual(sum(ss.first.sample.values()), 0)
                
    def test_planting_sat(self):
        # NP-hard to prove successful planting (rule out lower energies), this
        # test could fail coincidentally, but unlikely.        

        # test run time is O(2^num_var), keep num_var small.
        num_var = 6
        all_spin_assignments = np.array(list(itertools.product([-1, 1], repeat=num_var)))

        # NAE3SAT
        # Deep in UNSAT phase (num_clause/num_var>>2.1), very unlikely to be
        # SAT by chance.
        num_clauses = 24 
        bqm = dimod.generators.random_nae3sat(num_var, num_clauses, planted_solution=True)
        E_SAT = - num_clauses
        all_energies = bqm.energies((all_spin_assignments, bqm.variables))
        self.assertEqual(np.min(all_energies), E_SAT)
        self.assertEqual(all_energies[0], E_SAT) #all -1 state
        self.assertEqual(all_energies[-1], E_SAT) #all 1 state
        
        # 2in4SAT
        # Deep in UNSAT phase (num_clause/num_var>>0.9), very unlikely to be
        # SAT by chance.
        num_clauses = 12
        bqm = dimod.generators.random_2in4sat(num_var, num_clauses, planted_solution=True)
        E_SAT = - 2*num_clauses
        all_energies = bqm.energies((all_spin_assignments, bqm.variables))
        self.assertEqual(np.min(all_energies), E_SAT)
        self.assertEqual(all_energies[0], E_SAT) #all -1 state
        self.assertEqual(all_energies[-1], E_SAT) #all 1 state
        
    def test_labels(self):
        self.assertEqual(dimod.generators.random_2in4sat(10, 1).variables, range(10))
        self.assertEqual(dimod.generators.random_2in4sat('abdef', 1).variables, 'abdef')
