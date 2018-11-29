import unittest

import numpy as np

from dimod.bqm.views import LinearView, QuadraticView, AdjacencyView
from dimod.bqm.vectors import vector
from dimod.variables import MutableVariables


class TestLinearView(unittest.TestCase):
    class MockBQM(object):
        def __init__(self, variables=[], biases=[]):
            self._variables = MutableVariables(variables)
            self._ldata = vector(biases)
            self._iadj = {v: {} for v in variables}

        @property
        def ldata(self):
            return np.asarray(self._ldata)

    def test__getitem__contains__(self):
        linear = LinearView(self.MockBQM('abc', [-1, 0, 1]))

        self.assertIn('a', linear)
        self.assertIn('b', linear)
        self.assertIn('c', linear)
        self.assertEqual(linear, {'a': -1, 'b': 0, 'c': 1})
        self.assertEqual(linear['a'], -1)
        self.assertEqual(linear['b'], 0)
        self.assertEqual(linear['c'], 1)

        with self.assertRaises(KeyError):
            linear['d']

    def test__iter__(self):
        linear = LinearView(self.MockBQM('abc', [-1, 0, 1]))
        self.assertEqual(list(linear), ['a', 'b', 'c'])

    def test__len__(self):
        linear = LinearView(self.MockBQM('abc', [-1, 0, 1]))
        self.assertEqual(len(linear), 3)


class TestAdjacencyView(unittest.TestCase):
    class MockBQM(object):
        def __init__(self, J={}):
            self._variables = variables = MutableVariables(set().union(*J))

            if J:
                row, col, data = zip(*((u, v, b) for (u, v), b in J.items()))
            else:
                row = col = data = tuple()

            self._irow = vector(map(variables.index, row), dtype=np.int)
            self._icol = vector(map(variables.index, col), dtype=np.int)
            self._qdata = vector(data)

            self._iadj = iadj = {}

            for idx, (u, v) in enumerate(zip(row, col)):
                iadj.setdefault(u, {}).setdefault(v, idx)
                iadj.setdefault(v, {}).setdefault(u, idx)

    def test_instantiation(self):
        adj = AdjacencyView(self.MockBQM())
        self.assertEqual(len(adj), 0)
        self.assertEqual(adj, {})

    def test__getitem__contains__(self):
        J = {'ab': -1, 'bc': 1}
        adj = AdjacencyView(self.MockBQM(J))

        self.assertEqual(len(adj), 3)
        for (u, v), bias in J.items():
            self.assertIn(v, adj)
            self.assertIn(u, adj)
            self.assertIn(u, adj[v])
            self.assertIn(v, adj[u])
            self.assertEqual(adj[u][v], bias)
            self.assertEqual(adj[v][u], bias)


class TestQuadraticView(unittest.TestCase):
    class MockBQM(object):
        def __init__(self, J={}):
            self._variables = variables = MutableVariables(set().union(*J))

            if J:
                row, col, data = zip(*((u, v, b) for (u, v), b in J.items()))
            else:
                row = col = data = tuple()

            self._irow = vector(map(variables.index, row), dtype=np.int)
            self._icol = vector(map(variables.index, col), dtype=np.int)
            self._qdata = vector(data)

            self._iadj = iadj = {}

            for idx, (u, v) in enumerate(zip(row, col)):
                iadj.setdefault(u, {}).setdefault(v, idx)
                iadj.setdefault(v, {}).setdefault(u, idx)

    def test_instantiation(self):
        quadratic = QuadraticView(self.MockBQM())
        self.assertEqual(len(quadratic), 0)

    def test__getitem__contains__(self):
        quadratic = QuadraticView(self.MockBQM({'ab': -1, 'bc': 1}))
        self.assertEqual(len(quadratic), 2)
        self.assertEqual(quadratic[('a', 'b')], -1)
        self.assertEqual(quadratic[('b', 'a')], -1)
        self.assertEqual(quadratic[('c', 'b')], 1)
        self.assertEqual(quadratic[('b', 'c')], 1)
