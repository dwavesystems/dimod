import unittest

import numpy as np

from dimod.bqm.views import LinearView
from dimod.bqm.vectors import vector
from dimod.variables import MutableVariables


class TestLinearView(unittest.TestCase):
    def test_simple(self):
        linear = LinearView(MutableVariables([]), vector([], dtype=np.float))
        self.assertEqual(linear, {})
        linear['a'] = 1
        self.assertEqual(linear, {'a': 1})
        linear['a'] = -1
        self.assertEqual(linear, {'a': -1})
        linear['a'] += 2
        self.assertEqual(linear, {'a': 1})
        linear['b'] = 1
        del linear['a']
        self.assertEqual(linear, {'b': 1})

    def test__getitem__contains__(self):
        linear = LinearView(MutableVariables('abc'), vector([-1, 0, 1], dtype=np.float))

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
        linear = LinearView(MutableVariables('abc'), vector([-1, 0, 1], dtype=np.float))
        self.assertEqual(list(linear), ['a', 'b', 'c'])

    def test__len__(self):
        linear = LinearView(MutableVariables('abc'), vector([-1, 0, 1], dtype=np.float))
        self.assertEqual(len(linear), 3)

    def test__del__(self):
        linear = LinearView(MutableVariables([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
                            vector([-0.0, -0.13, -0.26, -0.39, -0.52, -0.65, -0.78, -0.91, -1.04, -1.17]))
        del linear[2]
        self.assertNotIn(2, linear)
