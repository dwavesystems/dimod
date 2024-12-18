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

    def test_frustrated_loop_clique(self):
        # More stringent test of special case:
        
        num_var = 6
        all_spin_assignments = np.array(list(itertools.product([-1, 1], repeat=num_var)))
        # Per loop added, atleast 1/4 chance all 1 is no longer a ground state
        # (of energy -num_loops). Add a bunch so that coincidental test pass is
        # highly unlikely:
        num_loops = 24
        
        # Use of a clique means all loops are atleast triangles, contribute at
        # most -1 to energy:
        clique = nx.Graph()
        clique.add_edges_from({(u, v) for u in range(num_var) for v in range(u)})
                             
        bqm = dimod.generators.frustrated_loop(clique, num_loops) 
        all_energies = bqm.energies((all_spin_assignments,bqm.variables))
        E_SAT = np.min(all_energies)
        self.assertLessEqual(E_SAT,-num_loops)
        self.assertEqual(all_energies[0],E_SAT) #All -1 planted
        self.assertEqual(all_energies[-1],E_SAT) #All 1 planted
        # Could add biclique graph to slightly extend concept.
    
    def test_plant_solution(self):
        G = self.G
        bqm = dimod.generators.frustrated_loop(G, 10, plant_solution=True)
        planted1 = {v:1 for v in G}
        planted2 = {v:-1 for v in G}
        self.assertEqual(bqm.energy(planted1),bqm.energy(planted2))
        # Check plant_solution=True option non-trivial:
        bqm = dimod.generators.frustrated_loop(G, 10, plant_solution=False)
        # Check that the loop is actually frustrated (in planted case):
        num_var = 3
        triangle = nx.Graph()
        triangle.add_edges_from({(u, v) for u in range(num_var) for v in range(u)})
        all_spin_assignments = np.array(list(itertools.product([-1, 1], repeat=num_var)))
        for plant_solution in [True, False]:
            bqm = dimod.generators.frustrated_loop(triangle, 1, plant_solution=plant_solution)
            all_energies = bqm.energies((all_spin_assignments,bqm.variables))
            E_SAT = -1
            self.assertLessEqual(E_SAT,np.min(all_energies))
            self.assertEqual(all_energies[0],E_SAT) # All -1 planted
            self.assertEqual(all_energies[-1],E_SAT) # All 1 planted
        
        
                        
    def test_planted_solution(self):
        # This tests a deprecated workflow,
        # test_plant_solution provides a generalized test for successful planting.
        G = self.G

        planted = {v:v%2*2-1 for v in G}
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.generators.frustrated_loop(G, 10, planted_solution=planted)

        inv_solution = {k:-v for k,v in planted.items()}
        self.assertEqual(bqm.energy(planted),bqm.energy(inv_solution))
        
        all_ones = {v:1 for v in G}
        self.assertNotEqual(bqm.energy(planted),bqm.energy(all_ones))

    def test_smoke_tuple_labels(self):
        # https://github.com/dwavesystems/dimod/issues/1342
        g = nx.erdos_renyi_graph(100, 0.5)
        tuple_map = dict()
        for node in g.nodes:
            tuple_map[node] = (node, node, node)
        g = nx.relabel_nodes(g, tuple_map)
        bqm = dimod.generators.frustrated_loop(g, R=3, num_cycles=100)


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

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            dimod.generators.knapsack([1], [], 1)

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

    def test_quadratic_exceptions(self):
        with self.assertRaises(ValueError):
            dimod.generators.quadratic_knapsack([1], [], [], 1)

        with self.assertRaises(ValueError):
            dimod.generators.quadratic_knapsack([1], [1], [[1, 1]], 1)

        with self.assertRaises(ValueError):
            dimod.generators.quadratic_knapsack([1], [1], [[1, 1], [1, 1]], 1)

    def test_quadratic_model(self):
        values = [1, 2]
        weights = [2, 1]
        profits = [[1, 2], [2, 1]]
        capacity = 1
        cqm = dimod.generators.quadratic_knapsack(values, weights, profits, capacity)
        self.assertEqual(len(cqm.variables), len(values))
        self.assertEqual(len(cqm.constraints), 1)

    def test_quadratic_infeasible(self):
        values = [1, 2]
        weights = [2, 1]
        profits = [[1, 2], [2, 1]]
        capacity = 1
        cqm = dimod.generators.quadratic_knapsack(values, weights, profits, capacity)

        # create an infeasible state, by selecting all the items
        x = {i: 1 for i in cqm.variables}
        lhs = cqm.constraints['capacity'].lhs.energy(x)
        self.assertGreater(lhs, 0)

    def test_quadratic_feasible(self):
        values = [1, 2]
        weights = [2, 1]
        profits = [[1, 2], [2, 1]]
        capacity = 1
        cqm = dimod.generators.quadratic_knapsack(values, weights, profits, capacity)

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

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            dimod.generators.multi_knapsack([1], [], [])

    def test_model(self):
        num_items = 20
        num_bins = 10
        cqm = dimod.generators.random_multi_knapsack(num_items=num_items, num_bins=num_bins)
        self.assertEqual(len(cqm.variables), num_items*num_bins)
        self.assertEqual(len(cqm.constraints), num_bins+num_items)

    def test_quadratic_exceptions(self):
        with self.assertRaises(ValueError):
            dimod.generators.quadratic_multi_knapsack([1], [], [], [])

        with self.assertRaises(ValueError):
            dimod.generators.quadratic_multi_knapsack([1], [1], [[1, 1]], [])

        with self.assertRaises(ValueError):
            dimod.generators.quadratic_multi_knapsack([1], [1], [[1, 1], [1, 1]], [])

    def test_quadratic_model(self):
        values = [1, 2, 3]
        weights = [3, 2, 1]
        profits = [[1, 2, 3], [2, 2, 2], [3, 2, 1]]
        capacity = [1, 2]
        cqm = dimod.generators.quadratic_multi_knapsack(values, weights, profits, capacity)
        self.assertEqual(len(cqm.variables), len(values)*len(capacity))
        self.assertEqual(len(cqm.constraints), len(weights)+len(capacity))


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
        bqm = dimod.generators.random_nae3sat(num_var, num_clauses, plant_solution=True)
        E_SAT = - num_clauses
        all_energies = bqm.energies((all_spin_assignments, bqm.variables))
        self.assertEqual(np.min(all_energies), E_SAT)
        self.assertEqual(all_energies[0], E_SAT) #all -1 state
        self.assertEqual(all_energies[-1], E_SAT) #all 1 state
        
        # 2in4SAT
        # Deep in UNSAT phase (num_clause/num_var>>0.9), very unlikely to be
        # SAT by chance.
        num_clauses = 12
        bqm = dimod.generators.random_2in4sat(num_var, num_clauses, plant_solution=True)
        E_SAT = - 2*num_clauses
        all_energies = bqm.energies((all_spin_assignments, bqm.variables))
        self.assertEqual(np.min(all_energies), E_SAT)
        self.assertEqual(all_energies[0], E_SAT) #all -1 state
        self.assertEqual(all_energies[-1], E_SAT) #all 1 state
        
    def test_labels(self):
        self.assertEqual(dimod.generators.random_2in4sat(10, 1).variables, range(10))
        self.assertEqual(dimod.generators.random_2in4sat('abdef', 1).variables, 'abdef')


class TestMagicSquares(unittest.TestCase):
    def test_invalid_power(self):
        for power in [-1, 0, 3, 5, 10]:
            with self.assertRaises(ValueError):
                dimod.generators.magic_square(1, power)

    def test_size(self):
        for size in [2**i for i in range(5)]:
            magic_square = dimod.generators.magic_square(size)
            
            self.assertEqual(len(magic_square.constraints), 2*size + 3)
            self.assertEqual(magic_square.num_variables(), size**2 + 1)
    
    def test_constraints(self):
        
        magic_square = dimod.generators.magic_square(5)
        
        constraint_names = magic_square.constraints.keys()
        
        for name in constraint_names:
            if name != "uniqueness":
                self.assertEqual(magic_square.constraints[name].rhs, 0)
                
                constraint_lhs = magic_square.constraints[name].lhs
                
                for name, term in constraint_lhs.linear.items():
                    if name != "sum":
                        self.assertEqual(term, 1)
                    else:
                        self.assertEqual(term, -1)
                
                for name, term in constraint_lhs.quadratic.items():
                    self.assertEqual(term, 1)
                
            else:
                self.assertEqual(magic_square.constraints[name].rhs, 300)
                
                constraint_lhs = magic_square.constraints[name].lhs
                
                for name, term in constraint_lhs.linear.items():
                    self.assertEqual(term, 0)
                    
                self_squares = [(i, i) for i in magic_square.variables]
                
                for name, term in constraint_lhs.quadratic.items():
                    if name not in self_squares:
                        self.assertEqual(term, -2)
                    else:
                        self.assertEqual(term, 24)
    
    def test_constraints_squares(self):
        
        magic_square = dimod.generators.magic_square(5, power=2)
        
        constraint_names = magic_square.constraints.keys()
        
        for name in constraint_names:
            if name != "uniqueness":
                self.assertEqual(magic_square.constraints[name].rhs, 0)
                
                constraint_lhs = magic_square.constraints[name].lhs
                
                for name, term in constraint_lhs.linear.items():
                    if name != "sum":
                        self.assertEqual(term, 0)
                    else:
                        self.assertEqual(term, -1)
                  
                for name, term in constraint_lhs.quadratic.items():
                    self.assertEqual(term, 1)
                
            else:
                self.assertEqual(magic_square.constraints[name].rhs, 300)
                
                constraint_lhs = magic_square.constraints[name].lhs

                for name, term in constraint_lhs.linear.items():
                    self.assertEqual(term, 0)
                    
                self_squares = [(i, i) for i in magic_square.variables]
                
                for name, term in constraint_lhs.quadratic.items():
                    if name not in self_squares:
                        self.assertEqual(term, -2)
                    else:
                        self.assertEqual(term, 24)

@unittest.skipUnless(_networkx, "no networkx installed")
class TestMIMO(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.symbols_bpsk = np.asarray([[-1, 1]])
        self.symbols_qam = lambda a: np.array([[complex(i, j)] \
            for i in range(-a, a + 1, 2) for j in range(-a, a + 1, 2)])
        
        def _make_honeycomb(L: int):    
            """Generate 2L by 2L triangular lattice. 
            
            The generated lattice has open boundaries and cut corners to make a hexagon. 

            Args:
                L: Length of lattice.

            Returns:
                :class:`networkx.Graph`.
            """
            G = nx.Graph()

            G.add_edges_from([((x, y), (x, y + 1)) for x in range(2*L + 1) for y in range(2*L)])
            G.add_edges_from([((x, y), (x + 1, y)) for x in range(2*L) for y in range(2*L + 1)])
            G.add_edges_from([((x, y), (x + 1, y +1 )) for x in range(2*L) for y in range(2*L)])
            G.remove_nodes_from([(i, j) for j in range(L) for i in range(L + 1 + j, 2*L + 1) ])
            G.remove_nodes_from([(i, j) for i in range(L) for j in range(L + 1 + i, 2*L + 1)])

            return G
        
        self._make_honeycomb = lambda L: _make_honeycomb(L)
            
    def _effective_fields(self, bqm):
        num_var = bqm.num_variables
        effFields = np.zeros(num_var)
        for key in bqm.quadratic:
            effFields[key[0]] += bqm.adj[key[0]][key[1]]
            effFields[key[1]] += bqm.adj[key[0]][key[1]]
        for key in bqm.linear:
            effFields[key] += bqm.linear[key]
        return effFields
        
    def test_filter_marginal_estimators(self):
        # Tested but so far this function is unused
        fme = dimod.generators.wireless.filter_marginal_estimator
        
        filtered_signal = self.rng.random(20) + np.arange(-20, 20, 2)
        estimated_source = fme(filtered_signal, 'BPSK')
        self.assertTrue(0 == len(set(estimated_source).difference(np.arange(-1, 3, 2))))
        self.assertTrue(np.all(estimated_source[:-1] <= estimated_source[1:]))
        
        filtered_signal = filtered_signal + 1j*(-self.rng.random(20) + np.arange(20, -20, -2))
        
        for modulation in ['QPSK','16QAM','64QAM']:
            estimated_source = fme(filtered_signal, modulation=modulation)
            self.assertTrue(np.all(np.flip(estimated_source.real) == estimated_source.imag))
    
    def test_linear_filter(self):

        Nt = 5
        Nr = 7
        # linear_filter(F, method='zero_forcing', PoverNt=1, SNRoverNt = 1)
        F = self.rng.normal(size=(Nr,Nt)) + 1j*self.rng.normal(size=(Nr,Nt))
        Fsimple = np.identity(Nt) # Nt=Nr

        #BPSK, real channel:
        transmitted_symbolsQAM = dimod.generators.wireless._create_transmitted_symbols(Nt, 
            amps=[-3, -1, 1, 3], quadrature=True)

        y = np.matmul(F, transmitted_symbolsQAM)

        # Defaults
        W = dimod.generators.wireless.linear_filter(F=F)
        self.assertEqual(W.shape,(Nt,Nr))

        # Check arguments:
        W = dimod.generators.wireless.linear_filter(F=F, 
            method='matched_filter', PoverNt=0.5, SNRoverNt=1.2)
        self.assertEqual(W.shape,(Nt,Nr))

        # Over-constrained noiseless channel by default, zero_forcing and MMSE are perfect:
        for method in ['zero_forcing', 'MMSE']:
            W = dimod.generators.wireless.linear_filter(F=F, method=method)
            reconstructed_symbols = np.matmul(W,y)
            self.assertTrue(np.all(np.abs(reconstructed_symbols - transmitted_symbolsQAM) < 1e-8))

        # matched_filter and MMSE (non-zero noise) are erroneous given interfered signal:
        W = dimod.generators.wireless.linear_filter(F=F, method='MMSE', PoverNt=0.5, SNRoverNt=1)
        reconstructed_symbols = np.matmul(W, y)
        self.assertTrue(np.all(np.abs(reconstructed_symbols - transmitted_symbolsQAM) > 1e-8))
            
    def test_quadratic_forms(self):
        # Quadratic form must evaluate to match original objective:
        num_var = 3
        num_receivers = 5
        F = self.rng.normal(0, 1, size=(num_receivers, num_var)) + \
            1j*self.rng.normal(0, 1, size=(num_receivers, num_var))
        y = self.rng.normal(0, 1, size=(num_receivers, 1)) + \
            1j*self.rng.normal(0, 1, size=(num_receivers, 1))

        # Random test case:
        vUnwrap = self.rng.normal(0, 1, size=(2*num_var, 1))
        v = vUnwrap[:num_var, :] + 1j*vUnwrap[num_var:, :]
        vec = y - np.matmul(F, v)
        val1 = np.matmul(vec.T.conj(), vec)

        # Check complex quadratic form
        k, h, J = dimod.generators.wireless._quadratic_form(y, F)
        val2 = np.matmul(v.T.conj(), np.matmul(J, v)) + (np.matmul(h.T.conj(), v)).real + k
        self.assertLess(abs(val1 - val2), 1e-8)

        # Check unwrapped complex quadratic form:
        h, J = dimod.generators.wireless._real_quadratic_form(h, J)
        val3 = np.matmul(vUnwrap.T, np.matmul(J, vUnwrap)) + np.matmul(h.T, vUnwrap) + k
        self.assertLess(abs(val1 - val3), 1e-8)

        # Check zero energy for y generated from F:
        y = np.matmul(F, v)
        k, h, J = dimod.generators.wireless._quadratic_form(y, F)
        val2 = np.matmul(v.T.conj(), np.matmul(J, v)) + (np.matmul(h.T.conj(), v)).real + k
        self.assertLess(abs(val2), 1e-8)
        h, J = dimod.generators.wireless._real_quadratic_form(h, J)
        val3 = np.matmul(vUnwrap.T, np.matmul(J, vUnwrap)) + np.matmul(h.T, vUnwrap) + k
        self.assertLess(abs(val3), 1e-8)

    def test_real_quadratic_form(self):
        h_in, J_in = np.array([1, 1]), np.array([2])
        h, J = dimod.generators.wireless._real_quadratic_form(h_in, J_in)
        self.assertTrue(np.array_equal(h_in, h))
        self.assertTrue(np.array_equal(J_in, J))

        h_in, J_in = np.array([1+1j, 1-1j]), np.array([[0, 2], [0, 0]])
        h, J = dimod.generators.wireless._real_quadratic_form(h_in, J_in)
        self.assertTrue(len(h) == 2*len(h_in))
        self.assertTrue(J.shape == (4, 4))

        h, J = dimod.generators.wireless._real_quadratic_form(h_in, J_in, 'BPSK')
        self.assertTrue(np.array_equal(np.real(h_in), h))
        self.assertTrue(np.array_equal(J_in, J))

    def test_amplitude_modulated_quadratic_form(self):
        num_var = 3
        h = self.rng.random(size=(num_var, 1))
        J = self.rng.random(size=(num_var, num_var))
        mods = ['BPSK', 'QPSK', '16QAM', '64QAM']
        mod_pref = [1, 1, 2, 3]
        for offset in [0]:
            for modI, modulation in enumerate(mods):
                hO, JO = dimod.generators.wireless._amplitude_modulated_quadratic_form(h, 
                    J, modulation=modulation)
                self.assertEqual(hO.shape[0], num_var*mod_pref[modI])
                self.assertEqual(JO.shape[0], hO.shape[0])
                self.assertEqual(JO.shape[0], JO.shape[1])

                max_val = 2**mod_pref[modI]-1
                self.assertLess(abs(max_val*np.sum(h)-np.sum(hO)), 1e-8)
                self.assertLess(abs(max_val*max_val*np.sum(J)-np.sum(JO)), 1e-8)
                #self.assertEqual(h.shape[0], num_var*mod_pref[modI])
                #self.assertLess(abs(bqm.offset-np.sum(np.diag(J))), 1e-8)

    def test_yF_to_hJ(self):
        F = np.array([[0, 1], [1, 1]])

        y = np.ones(2)
        h_bpsk, J_bpsk, o_bpsk = dimod.generators.wireless._yF_to_hJ(y, F, 'BPSK')
        h_qpsk, J_qpsk, o_qpsk = dimod.generators.wireless._yF_to_hJ(y, F, 'QPSK')
        self.assertTrue(np.array_equal(h_bpsk, h_qpsk))
        self.assertTrue(np.array_equal(J_bpsk, J_qpsk))
        self.assertTrue(np.array_equal(o_bpsk, o_qpsk))

        y = np.array([1, -1+1j])
        h_bpsk, J_bpsk, o_bpsk = dimod.generators.wireless._yF_to_hJ(y, F, 'BPSK')
        h_qpsk, J_qpsk, o_qpsk = dimod.generators.wireless._yF_to_hJ(y, F, 'QPSK')
        h_16, J_16, o_16 = dimod.generators.wireless._yF_to_hJ(y, F, '16QAM')
        self.assertFalse(np.array_equal(h_bpsk, h_qpsk))
        self.assertFalse(np.array_equal(J_bpsk, J_qpsk))
        self.assertFalse(np.array_equal(h_qpsk, h_16))
        self.assertFalse(np.array_equal(J_qpsk, J_16))
        self.assertTrue(np.array_equal(h_bpsk, h_qpsk[:, :2]))
        self.assertTrue(np.array_equal(J_bpsk, J_qpsk[:2, :2]))
        self.assertTrue(np.array_equal(o_bpsk, o_qpsk))
        self.assertTrue(np.array_equal(h_qpsk, h_16[:1, :]))
        self.assertTrue(np.array_equal(J_qpsk, J_16[:4, :4]))
        self.assertTrue(np.array_equal(o_qpsk, o_16))

    def test_bits_to_symbols(self):
        symbols = dimod.generators.wireless._bits_to_symbols(self.symbols_bpsk, 
            modulation='BPSK')
        self.assertTrue(np.array_equal(self.symbols_bpsk, symbols))
        
        symbols = dimod.generators.wireless._bits_to_symbols(self.symbols_bpsk, 
            modulation='BPSK', num_transmitters=1)
        self.assertTrue(np.array_equal(self.symbols_bpsk, symbols))

        symbols = dimod.generators.wireless._bits_to_symbols(self.symbols_qam(1), 
            modulation='QPSK')
        self.assertEqual(len(symbols), 2)

        symbols = dimod.generators.wireless._bits_to_symbols(self.symbols_qam(1), 
            modulation='16QAM')
        self.assertEqual(len(symbols), 1)
        
        with self.assertRaises(ValueError):
            bits = dimod.generators.wireless._bits_to_symbols(self.symbols_qam(1), 
            modulation='QPSK', num_transmitters=3)


    def test_symbols_to_bits(self):
        # Standard symbol cases (2D input):
        bits = dimod.generators.wireless._symbols_to_bits(self.symbols_bpsk, 
            modulation='BPSK')
        self.assertEqual(bits.sum(), 0)
        self.assertTrue(bits.ndim, 2)

        bits = dimod.generators.wireless._symbols_to_bits(self.symbols_qam(1), 
            modulation='QPSK')
        self.assertEqual(bits[:len(bits//2)].sum(), 0)
        self.assertEqual(bits[len(bits//2):].sum(), 0)
        self.assertTrue(bits.ndim, 2)

        bits = dimod.generators.wireless._symbols_to_bits(self.symbols_qam(3), 
            modulation='16QAM')
        self.assertEqual(bits[:len(bits//2)].sum(), 0)
        self.assertEqual(bits[len(bits//2):].sum(), 0)

        bits = dimod.generators.wireless._symbols_to_bits(self.symbols_qam(5), 
            modulation='64QAM')
        self.assertEqual(bits[:len(bits//2)].sum(), 0)
        self.assertEqual(bits[len(bits//2):].sum(), 0)

        # Standard symbol cases (1D input):
        bits = dimod.generators.wireless._symbols_to_bits(
            self.symbols_qam(1).reshape(4,), 
            modulation='QPSK')
        self.assertTrue(bits.ndim, 1)
        self.assertEqual(bits[:len(bits//2)].sum(), 0)
        self.assertEqual(bits[len(bits//2):].sum(), 0)

        # Unsupported input
        with self.assertRaises(ValueError):
            bits = dimod.generators.wireless._symbols_to_bits(self.symbols_bpsk, 
            modulation='unsupported')
                   
    def test_BPSK_symbol_coding(self):
        #This is simply read in read out.
        num_bits = 5
        bits = self.rng.choice([-1, 1], size=num_bits)
        symbols = dimod.generators.wireless._bits_to_symbols(bits=bits, modulation='BPSK')
        self.assertTrue(np.all(bits == symbols))
        bits = dimod.generators.wireless._symbols_to_bits(symbols=bits, modulation='BPSK')
        self.assertTrue(np.all(bits == symbols))
            
    def test_constellation_properties(self):
        _cp = dimod.generators.wireless._constellation_properties
        self.assertEqual(_cp("QPSK")[0], 2)
        self.assertEqual(sum(_cp("16QAM")[1]), 4)
        self.assertEqual(_cp("64QAM")[2], 42.0) 
        with self.assertRaises(ValueError):
            bits_per_transmitter, amps, constellation_mean_power = _cp("dummy")

    def test_create_transmitted_symbols(self):
        _cts = dimod.generators.wireless._create_transmitted_symbols
        self.assertTrue(_cts(1, amps=[-1, 1], quadrature=False)[0][0] in [-1, 1])
        self.assertTrue(_cts(1, amps=[-1, 1])[0][0].real in [-1, 1])
        self.assertTrue(_cts(1, amps=[-1, 1])[0][0].imag in [-1, 1])
        self.assertEqual(len(_cts(5, amps=[-1, 1])), 5)
        self.assertTrue(np.isin(_cts(20, amps=[-1, -3, 1, 3]).real, [-1, -3, 1, 3]).all())
        self.assertTrue(np.isin(_cts(20, amps=[-1, -3, 1, 3]).imag, [-1, -3, 1, 3]).all())
        with self.assertRaises(ValueError):
            transmitted_symbols = _cts(1, amps=[-1.1, 1], quadrature=False)
        with self.assertRaises(ValueError):
            transmitted_symbols = _cts(1, amps=np.array([-1, 1.1]), quadrature=False)
        with self.assertRaises(ValueError):
            transmitted_symbols = _cts(1, amps=np.array([-1, 1+1j]))

    def test_complex_symbol_coding(self):
        num_symbols = 5
        mod_pref = [1, 2, 3]
        mods = ['QPSK', '16QAM', '64QAM']

        for modI, mod in enumerate(mods):
            num_bits = 2*num_symbols*mod_pref[modI]
            max_symb = 2**mod_pref[modI]-1

            #uniform encoding (max bits = max amplitude symbols):
            bits = np.ones(num_bits)
            symbols = max_symb*np.ones(num_symbols) + 1j*max_symb*np.ones(num_symbols)
            symbols_enc = dimod.generators.wireless._bits_to_symbols(bits=bits, modulation=mod)
            self.assertTrue(np.all(symbols_enc == symbols ))
            bits_enc = dimod.generators.wireless._symbols_to_bits(symbols=symbols, modulation=mod)
            self.assertTrue(np.all(bits_enc == bits))

            #random encoding:
            bits = self.rng.choice([-1, 1], size=num_bits)
            symbols_enc = dimod.generators.wireless._bits_to_symbols(bits=bits, modulation=mod)
            bits_enc = dimod.generators.wireless._symbols_to_bits(symbols=symbols_enc, modulation=mod)
            self.assertTrue(np.all(bits_enc == bits))

    def test_mimo(self):
        for num_transmitters, num_receivers in [(1, 1), (5, 1), (1, 3), (11, 7)]:
            F = self.rng.normal(0, 1, size=(num_receivers, num_transmitters)) + \
                    1j*self.rng.normal(0, 1, size=(num_receivers, num_transmitters))
            y = self.rng.normal(0, 1, size=(num_receivers, 1)) + \
                    1j*self.rng.normal(0, 1, size=(num_receivers, 1))
            bqm = dimod.generators.wireless.mimo(modulation='QPSK', y=y, F=F)

            mod_pref = [1, 1, 2, 3]
            mods = ['BPSK', 'QPSK', '16QAM', '64QAM']
            for modI, modulation in enumerate(mods):
                bqm = dimod.generators.wireless.mimo(modulation=modulation, 
                    num_transmitters=num_transmitters, num_receivers=num_receivers)
                if modulation == 'BPSK':
                    constellation = [-1, 1]
                    dtype = np.float64
                else:
                    max_val = 2**mod_pref[modI] - 1
                    dtype = np.complex128
                    # All 1 bit encoding (max symbol in constellation)
                    constellation = [real_part + 1j*imag_part
                                     for real_part in range(-max_val, max_val+1, 2)
                                     for imag_part in range(-max_val, max_val+1, 2)]
                    
                F_simple = np.ones(shape=(num_receivers, num_transmitters), dtype=dtype)
                transmitted_symbols_max = np.ones(shape=(num_transmitters, 1), 
                    dtype=dtype)*constellation[-1]
                transmitted_symbols_random = self.rng.choice(constellation, 
                    size=(num_transmitters, 1))
                transmitted_bits_random = dimod.generators.wireless._symbols_to_bits(
                    symbols=transmitted_symbols_random.flatten(), modulation=modulation)

                #Trivial channel (F_simple), machine numbers
                bqm = dimod.generators.wireless.mimo(modulation=modulation, 
                    F=F_simple, transmitted_symbols=transmitted_symbols_max, 
                    SNRb=float('Inf'))
                
                ef = self._effective_fields(bqm)
                self.assertLessEqual(np.max(ef), 0)
                self.assertLessEqual(abs(bqm.energy((np.ones(bqm.num_variables), 
                    np.arange(bqm.num_variables)))), 1e-10)
                
                #Random channel, potential precision
                bqm = dimod.generators.wireless.mimo(modulation=modulation, 
                    num_transmitters=num_transmitters, num_receivers=num_receivers, 
                    transmitted_symbols=transmitted_symbols_max, 
                    SNRb=float('Inf'))
                ef=self._effective_fields(bqm)
                self.assertLessEqual(np.max(ef), 0)
                self.assertLess(abs(bqm.energy((np.ones(bqm.num_variables), 
                    np.arange(bqm.num_variables)))), 1e-8)
                
                # Add noise, check that offset is positive (random, scales as num_var/SNRb)
                bqm = dimod.generators.wireless.mimo(modulation=modulation, 
                    num_transmitters=num_transmitters, num_receivers=num_receivers, 
                    transmitted_symbols=transmitted_symbols_max, SNRb=1)
                self.assertLess(0, abs(bqm.energy((np.ones(bqm.num_variables), 
                    np.arange(bqm.num_variables)))))
                
                # Random transmission, should match bit encoding. Bit-encoded energy should be minimal
                bqm = dimod.generators.wireless.mimo(modulation=modulation, 
                    num_transmitters=num_transmitters, num_receivers=num_receivers, 
                    transmitted_symbols=transmitted_symbols_random, SNRb=float('Inf'))
                self.assertLess(abs(bqm.energy((transmitted_bits_random, 
                    np.arange(bqm.num_variables)))), 1e-8)

    def create_channel(self):
        # Test some defaults
        c, cp = dimod.generators.wireless.create_channel()[0]
        self.assertEqual(cp, 2)
        self.assertEqual(c.shape, (1, 1))

        c, cp = dimod.generators.wireless.create_channel(5, 5, 
            F_distribution=("normal", "real"))
        self.assertTrue(np.isin(c, [-1, 1]).all())
        self.assertEqual(cp, 5)

        c, cp = dimod.generators.wireless.create_channel(5, 5, 
            F_distribution=("binary", "complex"))
        self.assertTrue(np.isin(c, [-1-1j, -1+1j, 1-1j, 1+1j]).all())
        self.assertEqual(cp, 10)

        n_trans = 40
        c, cp = dimod.generators.wireless.create_channel(30, n_trans, 
            F_distribution=("normal", "real"))
        self.assertLess(c.mean(), 0.2)  
        self.assertLess(c.std(), 1.3)    
        self.assertGreater(c.std(), 0.7)
        self.assertEqual(cp, n_trans)

        c, cp = dimod.generators.wireless.create_channel(30, n_trans, 
            F_distribution=("normal", "complex"))
        self.assertLess(c.mean().complex, 0.2)  
        self.assertLess(c.real.std(), 1.3)    
        self.assertGreater(c.real.std(), 0.7)
        self.assertEqual(cp, 2*n_trans)

        c, cp = dimod.generators.wireless.create_channel(5, 5, 
            F_distribution=("binary", "real"), 
            attenuation_matrix=np.array([[1, 2], [3, 4]]))
        self.assertLess(c.ptp(), 8)
        self.assertEqual(cp, 30)
        
    def test_create_signal(self):
        # Only required parameters
        got, sent, noise, _ = dimod.generators.wireless._create_signal(F=np.array([[1]]))
        self.assertEqual(got, sent)
        self.assertTrue(all(np.isreal(got)))
        self.assertIsNone(noise)

        got, sent, _, __ = dimod.generators.wireless._create_signal(F=np.array([[-1]]))
        self.assertEqual(got, -sent)

        got, sent, noise, _ = dimod.generators.wireless._create_signal(F=np.array([[1], [1]]))
        self.assertEqual(got.shape, (2, 1))
        self.assertEqual(sent.shape, (1, 1))
        self.assertIsNone(noise)

        got, sent, _, __ = dimod.generators.wireless._create_signal(F=np.array([[1, 1]]))
        self.assertEqual(got.shape, (1, 1))
        self.assertEqual(sent.shape, (2, 1))

        # Optional parameters
        got, sent, _, __ = dimod.generators.wireless._create_signal(F=np.array([[1]]), modulation="QPSK")
        self.assertTrue(all(np.iscomplex(got)))
        self.assertTrue(all(np.iscomplex(sent)))
        self.assertEqual(got.shape, (1, 1))
        self.assertEqual(got, sent)

        got, sent, _, __ = dimod.generators.wireless._create_signal(F=np.array([[1]]), 
            transmitted_symbols=np.array([[1]]))
        self.assertEqual(got, sent)
        self.assertEqual(got[0][0], 1)

        with self.assertRaises(ValueError): # Complex symbols for BPSK
            a, b, c, d = dimod.generators.wireless._create_signal(F=np.array([[1]]), 
            transmitted_symbols=np.array([[1+1j]]))

        with self.assertRaises(ValueError): # Non-complex symbols for non-BPSK
            a, b, c, d = dimod.generators.wireless._create_signal(F=np.array([[1]]), 
            transmitted_symbols=np.array([[1]]), modulation="QPSK")

        noise = 0.2+0.3j
        got, sent, _, __ = dimod.generators.wireless._create_signal(F=np.array([[1]]), 
            transmitted_symbols=np.array([[1]]), channel_noise=noise)
        self.assertEqual(got, sent)
        got, sent, _, __ = dimod.generators.wireless._create_signal(F=np.array([[1]]), 
            transmitted_symbols=np.array([[1]]), channel_noise=noise, SNRb=10 )
        self.assertEqual(got, sent + noise)
        got, sent, _, __ = dimod.generators.wireless._create_signal(F=np.array([[1]]), 
            transmitted_symbols=np.array([[1]]), SNRb=10 )
        self.assertNotEqual(got, sent)
   
    def test_coordinated_multipoint(self):
        bqm = dimod.generators.wireless.coordinated_multipoint(lattice=nx.complete_graph(1), 
            modulation='BPSK')
        lattice = self._make_honeycomb(1)
        bqm = dimod.generators.wireless.coordinated_multipoint(lattice=lattice)
        num_var = lattice.number_of_nodes()
        self.assertEqual(num_var,bqm.num_variables)
        self.assertEqual(21,bqm.num_interactions)
        # Transmitted symbols are 1 by default
        lattice = self._make_honeycomb(2)
        bqm = dimod.generators.wireless.coordinated_multipoint(lattice=lattice,
            modulation='BPSK', SNRb=float('Inf'))
        self.assertLess(abs(bqm.energy((np.ones(bqm.num_variables), bqm.variables))), 1e-10)

    def test_attenuation_matrix(self):
        #Check that attenuation matches the matrix
        lattice=nx.Graph()
        num_var = 10
        lattice.add_nodes_from(n for n in range(num_var))

        A,_,_ = dimod.generators.wireless._lattice_to_attenuation_matrix(lattice)
        self.assertFalse(np.any(A-np.identity(num_var)))

        for t_per_node in range(1,3):
            for r_per_node in range(1,3):
                A,_,_ = dimod.generators.wireless._lattice_to_attenuation_matrix(
                    lattice,
                    transmitters_per_node=t_per_node,
                    receivers_per_node=r_per_node,
                    neighbor_root_attenuation=self.rng.random())
                self.assertFalse(np.any(A - np.tile(np.identity(num_var), (r_per_node, t_per_node))))

        for ea in range(2):
            lattice.add_edge(ea, ea+1)
            neighbor_root_attenuation = self.rng.random()
            A,_,_ = dimod.generators.wireless._lattice_to_attenuation_matrix(
                lattice, neighbor_root_attenuation=2)
            self.assertFalse(np.any(A - A.transpose()))
            self.assertTrue(all(A[eap, eap + 1]==2 for eap in range(ea + 1)))

        ## Check num_transmitters and num_receivers override:
        nx.set_node_attributes(lattice, values=3, name="num_transmitters")
        nx.set_node_attributes(lattice, values=1, name="num_receivers")
        A,_,_ = dimod.generators.wireless._lattice_to_attenuation_matrix(
            lattice,
            transmitters_per_node=2,
            receivers_per_node=2)
        self.assertEqual(A.shape, (num_var, 3*num_var))
        nx.set_node_attributes(lattice, values=0, name="num_receivers")
        nx.set_node_attributes(lattice, values={0:2, 3:1}, name="num_receivers")
        nx.set_node_attributes(lattice, values=0, name="num_transmitters")
        nx.set_node_attributes(lattice, values={i:1 for i in [0,1,2,4]}, name="num_transmitters")

        # t/r2 -- t -- t   r  t #We can assume the ntr and ntt arguments.
        Acorrect = np.array([[1, 2, 0, 0], [1, 2, 0, 0], [0, 0, 0, 0]])
        A,_,_ = dimod.generators.wireless._lattice_to_attenuation_matrix(
            lattice, neighbor_root_attenuation=2)
        self.assertFalse(np.any(A - Acorrect))

    def test_noise_scale(self):
        # After applying use_offset, the expected energy is the sum of noise terms.
        # (num_transmitters/SNRb)*sum_{mu=1}^{num_receivers} nu_mu^2 , where <nu_mu^2>=1 under default channels
        # We can do a randomized test (for practicl purpose, I fix the seed to avoid rare outliers):
        for num_transmitters in [256]:
            for SNRb in [0.1]:  #[0.1,10]
                for mods in [('BPSK', 1, 1, 1),('64QAM', 2, 42, 6)]:    #,('QPSK',2,2,2),('16QAM',2,10,4)]:
                    mod, channel_power_per_transmitter, constellation_mean_power, bits_per_transmitter = mods
                    for num_receivers in [num_transmitters*4]: #[num_transmitters//4,num_transmitters]:
                        EoverN = (channel_power_per_transmitter * \
                            constellation_mean_power/bits_per_transmitter/SNRb) * \
                            num_transmitters * num_receivers
                        if mod=='BPSK':
                            EoverN *= 2 #Real part only
                        for seed in range(1):
                            bqm0 = dimod.generators.wireless.mimo(modulation=mod,
                                num_transmitters=num_transmitters, num_receivers=num_receivers,
                                seed=seed)                     
                            bqm = dimod.generators.wireless.mimo(modulation=mod,
                                num_transmitters=num_transmitters, num_receivers=num_receivers, 
                                SNRb=SNRb, seed=seed)

                            #E[n^2] constructed from offsets correctly:
                            scale_n = (bqm.offset - bqm0.offset)/EoverN
                            self.assertGreater(1.5, scale_n)
                            self.assertLess(0.5, scale_n)
                            #scale_n_alt = np.sum(abs(n)**2,axis=0)/EoverN)

                    for num_transmitter_block in [2]: #[1,2]:
                        lattice_size = num_transmitters//num_transmitter_block
                        for num_receiver_block in [1]:  #[1,2]:
                            # Similar applies for COMP, up to boundary conditions. Choose a symmetric lattice:
                            num_receiversT = lattice_size * num_receiver_block
                            num_transmittersT = lattice_size * num_transmitter_block
                            EoverN = (channel_power_per_transmitter * \
                                constellation_mean_power/bits_per_transmitter/SNRb) * \
                                num_transmittersT * num_receiversT
                        
                            if mod=='BPSK':
                                EoverN *= 2 #Real part only
                            lattice = nx.Graph()
                            lattice.add_edges_from((i, (i + 1)%lattice_size) for i in 
                                range(num_transmitters//num_transmitter_block))
                            for seed in range(1):
                                bqm = dimod.generators.wireless.coordinated_multipoint(lattice=lattice,
                                    modulation=mod, SNRb=SNRb)
                                bqm0 = dimod.generators.wireless.coordinated_multipoint(lattice=lattice,
                                    modulation=mod)
                                scale_n = (bqm.offset - bqm0.offset)/EoverN
                                self.assertGreater(1.5, scale_n)
                                #self.assertLess(0.5, scale_n)
                            


class TestBPSP(unittest.TestCase):
    def test_small_problem(self):
        # Problem from Fig.1 in Streif, Michael, et al. (2021)
        car_sequence = [1, 2, 1, 3, 3, 2]
        
        bqm = dimod.generators.bpsp.binary_paint_shop_problem(car_sequence)
        ss = dimod.ExactSolver().sample(bqm)
        car_colors, color_changes = dimod.generators.bpsp.sample_to_coloring(ss.first.sample, car_sequence)
        self.assertEqual(color_changes, 2)


class TestPowerR(unittest.TestCase):
    def test_singleton(self):
        bqm = dimod.generators.random.power_r(1, 1)

        # should have a single node
        self.assertEqual(len(bqm), 1)
        self.assertIn(0, bqm.variables)

    def test_empty(self):
        bqm = dimod.generators.random.power_r(1, 0)

        # should have no nodes
        self.assertEqual(len(bqm), 0)

    def test_seed(self):
        bqm0 = dimod.generators.random.power_r(3, 100, seed=506)
        bqm1 = dimod.generators.random.power_r(3, 100, seed=506)

        self.assertEqual(bqm0, bqm1)

        bqm2 = dimod.generators.random.power_r(3, 100, seed=123)

        self.assertNotEqual(bqm2, bqm1)

    def test_values(self):
        bqm = dimod.generators.random.power_r(5, 10)

        self.assertFalse(all(bqm.linear.values()))
        self.assertTrue(all(val != 0 for val in bqm.quadratic.values()))
        self.assertTrue(all(val <= 5 for val in bqm.quadratic.values()))
        self.assertTrue(all(val >= -5 for val in bqm.quadratic.values()))


class TestQuadraticAssignment(unittest.TestCase):

    def test_unequal_shape(self):
        distance_matrix = np.array([[1,2], [3,4]])
        flow_matrix = np.array([5])
        with self.assertRaises(ValueError):
            dimod.generators.quadratic_assignment(distance_matrix, flow_matrix)

    def test_not_square(self):
        distance_matrix = np.array([[1,2],[3,4],[5,6]])
        flow_matrix = np.array([[1,2],[3,4],[5,6]])
        with self.assertRaises(ValueError):
            dimod.generators.quadratic_assignment(distance_matrix, flow_matrix)

    def test_not_2d(self):
        distance_matrix = np.array([
                                    [[1, 2, 3, 4],
                                    [5, 6, 7, 8]],                                    
                                    [[13, 14, 15, 16],
                                    [17, 18, 19, 20]]
                                ])
        flow_matrix = np.array([
                                    [[1, 2, 3, 4],
                                    [5, 6, 7, 8]],                                    
                                    [[13, 14, 15, 16],
                                    [17, 18, 19, 20]]
                                ])
        with self.assertRaises(ValueError):
            dimod.generators.quadratic_assignment(distance_matrix, flow_matrix)

    def test_model(self):
        distance_matrix = np.array([[0,2,3],[4,0,5],[6,1,0]])
        flow_matrix = np.array([[0,1,7],[2,0,3],[8,5,0]])
        num_locations = distance_matrix.shape[0]
        cqm = dimod.generators.quadratic_assignment(distance_matrix, flow_matrix)
        self.assertEqual(len(cqm.variables), num_locations**2)
        self.assertEqual(len(cqm.constraints), 2*num_locations)

    def test_infeasible(self):
        distance_matrix = np.array([[0,2,3],[4,0,5],[6,1,0]])
        flow_matrix = np.array([[0,1,7],[2,0,3],[8,5,0]])
        num_locations = distance_matrix.shape[0]
        cqm = dimod.generators.quadratic_assignment(distance_matrix, flow_matrix)

        for i in range(num_locations):
            x = {f'x_{i}_{j}': 1 for j in range(num_locations)}
            lhs = cqm.constraints[f'discrete_constraint_{i}'].lhs.energy(x)
            self.assertNotEqual(lhs, 1)

        for j in range(num_locations):
            x = {f'x_{i}_{j}': 1 for i in range(num_locations)}
            lhs = cqm.constraints[f'facility_constraint_{j}'].lhs.energy(x)
            self.assertNotEqual(lhs, 0)

    def test_feasible(self):
        distance_matrix = np.array([[0,2,3],[4,0,5],[6,1,0]])
        flow_matrix = np.array([[0,1,7],[2,0,3],[8,5,0]])
        num_locations = distance_matrix.shape[0]
        cqm = dimod.generators.quadratic_assignment(distance_matrix, flow_matrix)

        for i in range(num_locations):
            x = {f'x_{i}_{j}': 1 if i==j else 0 for j in range(num_locations)}
            lhs = cqm.constraints[f'discrete_constraint_{i}'].lhs.energy(x)
            self.assertEqual(lhs, 1)

        for j in range(num_locations):
            x = {f'x_{i}_{j}': 1 if i==j else 0 for i in range(num_locations)}
            lhs = cqm.constraints[f'facility_constraint_{j}'].lhs.energy(x)
            self.assertEqual(lhs, 0)
