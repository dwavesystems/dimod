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
#
# =============================================================================
import itertools
import unittest

import dimod

try:
    import networkx as nx
except ImportError:
    _networkx = False
else:
    _networkx = True


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


@unittest.skipUnless(_networkx, "no networkx installed")
class TestFCL(unittest.TestCase):
    def test_singletile(self):
        G = nx.Graph()

        for u in range(4):
            for v in range(4, 8):
                G.add_edge(u, v)

        bqm = dimod.generators.frustrated_loop(G, 10)

        self.assertEqual(len(bqm), 8)
        self.assertEqual(len(bqm.quadratic), 16)
        for i in range(4):
            for j in range(4, 8):
                self.assertIn(i, bqm.adj)
                self.assertIn(j, bqm.adj[i])

    def test_seed(self):
        G = nx.Graph()

        for u in range(4):
            for v in range(4, 8):
                G.add_edge(u, v)

        bqm0 = dimod.generators.frustrated_loop(G, 10, seed=506)
        bqm1 = dimod.generators.frustrated_loop(G, 10, seed=506)

        self.assertEqual(bqm0, bqm1)

        bqm2 = dimod.generators.frustrated_loop(G, 10, seed=123)

        self.assertNotEqual(bqm2, bqm1)


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
