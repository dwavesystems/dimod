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


class TestRandomRandint(unittest.TestCase):
    def test_singleton(self):
        bqm = dimod.generators.random.randint(1, dimod.BINARY)

        # should have a single node
        self.assertEqual(len(bqm), 1)
        self.assertIn(0, bqm.variables)


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


class TestFCL(unittest.TestCase):
    @unittest.skipUnless(_networkx, "no networks installed")
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
