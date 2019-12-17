# Copyright 2019 D-Wave Systems Inc.
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
"""
Generic dimod/bqm tests.
"""
import itertools
import unittest

from collections import OrderedDict
from functools import wraps

import numpy as np

import dimod


def skipUnlessShapeable(f):
    """Ignore the wrapped test if the BQM is not shapeable."""
    @wraps(f)
    def _wrapper(test_case):
        if test_case.BQM.shapeable():
            return f(test_case)
        else:
            raise unittest.SkipTest("Unshapeable BQM")
    return _wrapper


class TestBQM:
    """Generic tests for BQM subclasses.

    These tests do not run on their own and are intended to be subclassed.
    Assumes that there is a class-attribute `.BQM`.

    Example:
        >>> class TestBQMSubClass(TestBQM, unittest.TestCase):
        ...     @classmethod
        ...     def setUpClass(cls):
        ...         cls.BQM = BQMSubClass  # the class to be tested

    """
    def assertConsistentBQM(self, bqm):
        # adjacency and linear are self-consistent
        for v in bqm.linear:
            self.assertIn(v, bqm.adj)
        for v in bqm.adj:
            self.assertIn(v, bqm.linear)

        # adjacency and quadratic are self-consistent
        for u, v in bqm.quadratic:
            self.assertIn(v, bqm.linear)
            self.assertIn(v, bqm.adj)
            self.assertIn(u, bqm.adj[v])

            self.assertIn(u, bqm.linear)
            self.assertIn(u, bqm.adj)
            self.assertIn(v, bqm.adj[u])

            self.assertEqual(bqm.adj[u][v], bqm.quadratic[(u, v)])
            self.assertEqual(bqm.adj[u][v], bqm.quadratic[(v, u)])
            self.assertEqual(bqm.adj[v][u], bqm.adj[u][v])

        for u, v in bqm.quadratic:
            self.assertEqual(bqm.get_quadratic(u, v), bqm.quadratic[(u, v)])
            self.assertEqual(bqm.get_quadratic(u, v), bqm.quadratic[(v, u)])
            self.assertEqual(bqm.get_quadratic(v, u), bqm.quadratic[(u, v)])
            self.assertEqual(bqm.get_quadratic(v, u), bqm.quadratic[(v, u)])

        for u in bqm.adj:
            for v in bqm.adj[u]:
                self.assertIn((u, v), bqm.quadratic)
                self.assertIn((v, u), bqm.quadratic)

        self.assertEqual(len(bqm.quadratic), bqm.num_interactions)
        self.assertEqual(len(bqm.linear), bqm.num_variables)
        self.assertEqual(len(bqm.quadratic), len(set(bqm.quadratic)))
        self.assertEqual(len(bqm.variables), len(bqm.linear))
        self.assertEqual((bqm.num_variables, bqm.num_interactions), bqm.shape)

    def assertConsistentEnergies(self, spin, binary):
        # brute-force check that all samples have matching energy
        assert spin.vartype is dimod.SPIN
        assert binary.vartype is dimod.BINARY

        variables = list(spin.variables)

        self.assertEqual(set(spin.variables), set(binary.variables))

        for spins in itertools.product([-1, +1], repeat=len(variables)):
            spin_sample = dict(zip(variables, spins))
            binary_sample = {v: (s + 1)//2 for v, s in spin_sample.items()}

            spin_energy = spin.offset
            spin_energy += sum(spin_sample[v] * bias
                               for v, bias in spin.linear.items())
            spin_energy += sum(spin_sample[v] * spin_sample[u] * bias
                               for (u, v), bias in spin.quadratic.items())

            binary_energy = binary.offset
            binary_energy += sum(binary_sample[v] * bias
                                 for v, bias in binary.linear.items())
            binary_energy += sum(binary_sample[v] * binary_sample[u] * bias
                                 for (u, v), bias in binary.quadratic.items())

            self.assertAlmostEqual(spin_energy, binary_energy)

    def test__len__(self):
        bqm = self.BQM(np.ones((107, 107)), dimod.BINARY)
        self.assertEqual(len(bqm), 107)

    @skipUnlessShapeable
    def test_add_variable_exception(self):
        bqm = self.BQM(dimod.SPIN)
        with self.assertRaises(TypeError):
            bqm.add_variable([])

    @skipUnlessShapeable
    def test_add_variable_labelled(self):
        bqm = self.BQM(dimod.SPIN)
        bqm.add_variable('a')
        bqm.add_variable(1)
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(list(bqm.iter_variables()), ['a', 1])
        bqm.add_variable()
        self.assertEqual(bqm.shape, (3, 0))
        self.assertEqual(list(bqm.iter_variables()), ['a', 1, 2])

    @skipUnlessShapeable
    def test_add_variable_int_labelled(self):
        bqm = self.BQM(dimod.SPIN)
        self.assertEqual(bqm.add_variable(1), 1)
        self.assertEqual(bqm.add_variable(), 0)  # 1 is already taken
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(bqm.add_variable(), 2)
        self.assertEqual(bqm.shape, (3, 0))

    @skipUnlessShapeable
    def test_add_variable_unlabelled(self):
        bqm = self.BQM(dimod.SPIN)
        bqm.add_variable()
        bqm.add_variable()
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(list(bqm.iter_variables()), [0, 1])

    def test_adj_setitem(self):
        bqm = self.BQM(({}, {'ab': -1}), 'SPIN')
        bqm.adj['a']['b'] = 5
        self.assertEqual(bqm.adj['a']['b'], 5)
        self.assertConsistentBQM(bqm)  # all the other cases

    def test_adj_neighborhoods(self):
        bqm = self.BQM(({}, {'ab': -1, 'ac': -1, 'bc': -1, 'cd': -1}), 'SPIN')

        self.assertEqual(len(bqm.adj['a']), 2)
        self.assertEqual(len(bqm.adj['b']), 2)
        self.assertEqual(len(bqm.adj['c']), 3)
        self.assertEqual(len(bqm.adj['d']), 1)

    def test_change_vartype_binary_to_binary_copy(self):
        bqm = self.BQM(({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, 1.4),
                       'BINARY')

        new = bqm.change_vartype(dimod.BINARY, inplace=False)
        self.assertEqual(bqm, new)
        self.assertIsNot(bqm, new)  # should be a copy

    def test_change_vartype_binary_to_spin_copy(self):
        bqm = self.BQM(({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.8}, 1.8),
                       'BINARY')

        # change vartype
        new = bqm.change_vartype(dimod.SPIN, inplace=False)

        self.assertConsistentEnergies(spin=new, binary=bqm)

    def test_change_vartype_spin_to_spin_copy(self):
        bqm = self.BQM(({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, 1.4),
                       'SPIN')

        new = bqm.change_vartype(dimod.SPIN, inplace=False)
        self.assertEqual(bqm, new)
        self.assertIsNot(bqm, new)  # should be a copy

    def test_change_vartype_spin_to_binary_copy(self):
        bqm = self.BQM(({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.8}, 1.8),
                       'SPIN')

        # change vartype
        new = bqm.change_vartype(dimod.BINARY, inplace=False)

        self.assertConsistentEnergies(spin=bqm, binary=new)

    def test_construction_args(self):
        # make sure all the orderes are ok

        with self.assertRaises(TypeError):
            self.BQM()

        bqm = self.BQM('SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (0, 0))

        bqm = self.BQM(vartype='SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (0, 0))

        bqm = self.BQM(5, 'SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (5, 0))

        bqm = self.BQM(5, vartype='SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (5, 0))

        bqm = self.BQM(vartype='SPIN', obj=5)
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (5, 0))

    def test_construction_symmetric(self):
        bqm = self.BQM(np.ones((5, 5)), 'BINARY')
        self.assertConsistentBQM(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)

    def test_copy(self):
        bqm = self.BQM(({'a': -1, 'b': 1}, {}), dimod.BINARY)
        new = bqm.copy()
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)

        # modify the original and make sure it doesn't propogate
        new.set_linear('a', 1)
        self.assertEqual(new.linear['a'], 1)

    def test_copy_subclass(self):
        # copy should respect subclassing
        class SubBQM(self.BQM):
            pass

        bqm = SubBQM(({'a': -1, 'b': 1}, {}), dimod.BINARY)
        new = bqm.copy()
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)

    def test_get_linear_disconnected_string_labels(self):
        bqm = self.BQM(({'a': -1, 'b': 1}, {}), dimod.BINARY)
        self.assertEqual(bqm.get_linear('a'), -1)
        self.assertEqual(bqm.get_linear('b'), 1)
        with self.assertRaises(ValueError):
            bqm.get_linear('c')

    def test_get_linear_disconnected(self):
        bqm = self.BQM(5, dimod.SPIN)

        for v in range(5):
            self.assertEqual(bqm.get_linear(v), 0)

        with self.assertRaises(ValueError):
            bqm.get_linear(-1)

        with self.assertRaises(ValueError):
            bqm.get_linear(5)

    def test_get_linear_dtype(self):
        bqm = self.BQM(5, dimod.SPIN)

        for v in range(5):
            self.assertIsInstance(bqm.get_linear(v), bqm.dtype.type)

    def test_get_quadratic_3x3array(self):
        bqm = self.BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]], dimod.SPIN)

        self.assertEqual(bqm.get_quadratic(0, 1), 1)
        self.assertEqual(bqm.get_quadratic(1, 0), 1)

        self.assertEqual(bqm.get_quadratic(0, 2), 2)
        self.assertEqual(bqm.get_quadratic(2, 0), 2)

        with self.assertRaises(ValueError):
            bqm.get_quadratic(2, 1)
        with self.assertRaises(ValueError):
            bqm.get_quadratic(1, 2)

        with self.assertRaises(ValueError):
            bqm.get_quadratic(0, 0)

    def test_get_linear_dtype(self):
        bqm = self.BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]], dimod.SPIN)

        self.assertIsInstance(bqm.get_quadratic(0, 1), bqm.dtype.type)
        self.assertIsInstance(bqm.get_quadratic(1, 0), bqm.dtype.type)

        self.assertIsInstance(bqm.get_quadratic(0, 2), bqm.dtype.type)
        self.assertIsInstance(bqm.get_quadratic(2, 0), bqm.dtype.type)

    def test_has_variable(self):
        h = OrderedDict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = self.BQM((h, J), dimod.SPIN)

        self.assertTrue(bqm.has_variable('a'))
        self.assertTrue(bqm.has_variable(1))
        self.assertTrue(bqm.has_variable(3))

        # no false positives
        self.assertFalse(bqm.has_variable(0))
        self.assertFalse(bqm.has_variable(2))

    def test_iter_quadratic_neighbours(self):
        bqm = self.BQM(({}, {'ab': -1, 'bc': 21, 'cd': 1}), dimod.SPIN)
        neighbours = set(bqm.iter_quadratic('b'))
        self.assertEqual(neighbours,
                         {('b', 'a', -1), ('b', 'c', 21)})

    def test_iter_quadratic_neighbours_bunch(self):
        bqm = self.BQM(({}, {'bc': 21, 'cd': 1}), dimod.SPIN)
        self.assertEqual(list(bqm.iter_quadratic(['b', 'c'])),
                         [('b', 'c', 21.0), ('c', 'd', 1.0)])

    def test_iter_variables(self):
        h = OrderedDict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = self.BQM((h, J), dimod.SPIN)

        self.assertEqual(list(bqm.iter_variables()), ['a', 1, 3])

    @skipUnlessShapeable
    def tests_linear_delitem(self):
        bqm = self.BQM([[0, 1, 2, 3, 4],
                        [0, 6, 7, 8, 9],
                        [0, 0, 10, 11, 12],
                        [0, 0, 0, 13, 14],
                        [0, 0, 0, 0, 15]], 'SPIN')
        del bqm.linear[2]
        self.assertEqual(set(bqm.iter_variables()), set([0, 1, 3, 4]))

        # all the values are correct
        self.assertEqual(bqm.linear[0], 0)
        self.assertEqual(bqm.linear[1], 6)
        self.assertEqual(bqm.linear[3], 13)
        self.assertEqual(bqm.linear[4], 15)
        self.assertEqual(bqm.quadratic[0, 1], 1)
        self.assertEqual(bqm.quadratic[0, 3], 3)
        self.assertEqual(bqm.quadratic[0, 4], 4)
        self.assertEqual(bqm.quadratic[1, 3], 8)
        self.assertEqual(bqm.quadratic[1, 4], 9)
        self.assertEqual(bqm.quadratic[3, 4], 14)

        self.assertConsistentBQM(bqm)

        with self.assertRaises(KeyError):
            del bqm.linear[2]

    def test_linear_setitem(self):
        bqm = self.BQM(({}, {'ab': -1}, 0), dimod.SPIN)
        bqm.linear['a'] = 5
        self.assertEqual(bqm.get_linear('a'), 5)
        self.assertConsistentBQM(bqm)

    @skipUnlessShapeable
    def test_quadratic_delitem(self):
        bqm = self.BQM([[0, 1, 2, 3, 4],
                        [0, 6, 7, 8, 9],
                        [0, 0, 10, 11, 12],
                        [0, 0, 0, 13, 14],
                        [0, 0, 0, 0, 15]], 'SPIN')
        del bqm.quadratic[0, 1]
        self.assertEqual(set(bqm.iter_neighbors(0)), set([2, 3, 4]))
        self.assertConsistentBQM(bqm)

        with self.assertRaises(KeyError):
            del bqm.quadratic[0, 1]

    def test_quadratic_setitem(self):
        bqm = self.BQM(({}, {'ab': -1}, 0), dimod.SPIN)
        bqm.quadratic[('a', 'b')] = 5
        self.assertEqual(bqm.get_quadratic('a', 'b'), 5)
        self.assertConsistentBQM(bqm)

    def test_offset(self):
        h = dict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = self.BQM((h, J), 'SPIN')
        self.assertEqual(bqm.offset, 0)
        self.assertIsInstance(bqm.offset, bqm.dtype.type)

        h = dict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = self.BQM((h, J, 1.5), 'SPIN')
        self.assertEqual(bqm.offset, 1.5)
        self.assertIsInstance(bqm.offset, bqm.dtype.type)

        bqm.offset = 6
        self.assertEqual(bqm.offset, 6)
        self.assertIsInstance(bqm.offset, bqm.dtype.type)

    @skipUnlessShapeable
    def test_remove_interaction(self):
        bqm = self.BQM(np.triu(np.ones((3, 3))), dimod.BINARY)
        bqm.remove_interaction(0, 1)
        with self.assertRaises(ValueError):
            bqm.remove_interaction(0, 1)
        self.assertEqual(bqm.shape, (3, 2))

        with self.assertRaises(ValueError):
            bqm.remove_interaction('a', 1)  # 'a' is not a variable

        with self.assertRaises(ValueError):
            bqm.remove_interaction(1, 1)

    @skipUnlessShapeable
    def test_remove_variable_labelled(self):
        bqm = self.BQM(dimod.SPIN)
        bqm.add_variable('a')
        bqm.add_variable(1)
        bqm.add_variable(0)
        self.assertEqual(bqm.remove_variable(), 0)
        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm.remove_variable(), 1)
        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm.remove_variable(), 'a')
        self.assertConsistentBQM(bqm)
        with self.assertRaises(ValueError):
            bqm.remove_variable()

    @skipUnlessShapeable
    def test_remove_variable_provided(self):
        bqm = self.BQM('SPIN')
        bqm.add_variable('a')
        bqm.add_variable('b')
        bqm.add_variable('c')

        bqm.remove_variable('b')
        self.assertConsistentBQM(bqm)

        # maintained order
        self.assertEqual(list(bqm.iter_variables()), ['a', 'c'])

        with self.assertRaises(ValueError):
            bqm.remove_variable('b')

    @skipUnlessShapeable
    def test_remove_variable_unlabelled(self):
        bqm = self.BQM(2, dimod.BINARY)
        self.assertEqual(bqm.remove_variable(), 1)
        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm.remove_variable(), 0)
        self.assertConsistentBQM(bqm)
        with self.assertRaises(ValueError):
            bqm.remove_variable()
        self.assertConsistentBQM(bqm)

    def test_set_linear(self):
        # does not change shape
        bqm = self.BQM(np.triu(np.ones((3, 3))), dimod.BINARY)

        self.assertEqual(bqm.get_linear(0), 1)
        bqm.set_linear(0, .5)
        self.assertEqual(bqm.get_linear(0), .5)

    def test_set_quadratic(self):
        # does not change shape
        bqm = self.BQM(np.triu(np.ones((3, 3))), dimod.BINARY)

        self.assertEqual(bqm.get_quadratic(0, 1), 1)
        bqm.set_quadratic(0, 1, .5)
        self.assertEqual(bqm.get_quadratic(0, 1), .5)
        self.assertEqual(bqm.get_quadratic(1, 0), .5)
        bqm.set_quadratic(0, 1, -.5)
        self.assertEqual(bqm.get_quadratic(0, 1), -.5)
        self.assertEqual(bqm.get_quadratic(1, 0), -.5)

    @skipUnlessShapeable
    def test_set_quadratic_exception(self):
        bqm = self.BQM(dimod.SPIN)
        with self.assertRaises(TypeError):
            bqm.set_quadratic([], 1, .5)
        with self.assertRaises(TypeError):
            bqm.set_quadratic(1, [], .5)

    def test_shape_3x3array(self):
        bqm = self.BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]], dimod.BINARY)

        self.assertEqual(bqm.shape, (3, 2))
        self.assertEqual(bqm.num_variables, 3)
        self.assertEqual(bqm.num_interactions, 2)

    def test_shape_disconnected(self):
        bqm = self.BQM(5, dimod.BINARY)

        self.assertEqual(bqm.shape, (5, 0))
        self.assertEqual(bqm.num_variables, 5)
        self.assertEqual(bqm.num_interactions, 0)

    def test_shape_empty(self):
        self.assertEqual(self.BQM(dimod.SPIN).shape, (0, 0))
        self.assertEqual(self.BQM(0, dimod.SPIN).shape, (0, 0))

        self.assertEqual(self.BQM(dimod.SPIN).num_variables, 0)
        self.assertEqual(self.BQM(0, dimod.SPIN).num_variables, 0)

        self.assertEqual(self.BQM(dimod.SPIN).num_interactions, 0)
        self.assertEqual(self.BQM(0, dimod.SPIN).num_interactions, 0)

    def test_to_coo_empty(self):
        bqm = self.BQM('SPIN')
        ldata, (irow, icol, qdata), off, labels = bqm.to_coo()

    def test_to_coo(self):
        bqm = self.BQM(({'a': -1}, {'ab': 2}), 'SPIN')
        ldata, (irow, icol, qdata), off, labels = bqm.to_coo()

        np.testing.assert_array_equal(ldata, [-1, 0])
        np.testing.assert_array_equal(irow, [0])
        np.testing.assert_array_equal(icol, [1])
        np.testing.assert_array_equal(qdata, [2])
        self.assertEqual(labels, ['a', 'b'])

    def test_vartype(self):
        bqm = self.BQM(vartype='SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)

        bqm = self.BQM(vartype=dimod.SPIN)
        self.assertEqual(bqm.vartype, dimod.SPIN)

        bqm = self.BQM(vartype=(-1, 1))
        self.assertEqual(bqm.vartype, dimod.SPIN)

        bqm = self.BQM(vartype='BINARY')
        self.assertEqual(bqm.vartype, dimod.BINARY)

        bqm = self.BQM(vartype=dimod.BINARY)
        self.assertEqual(bqm.vartype, dimod.BINARY)

        bqm = self.BQM(vartype=(0, 1))
        self.assertEqual(bqm.vartype, dimod.BINARY)

    def test_vartype_readonly(self):
        bqm = self.BQM(vartype='SPIN')
        with self.assertRaises(AttributeError):
            bqm.vartype = dimod.BINARY
