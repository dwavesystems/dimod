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

To run these tests for all bqms, you need to run them on the various adj
files AND this file, e.g. `python -m unittest tests/test_adj* tests/test_bqm.py`
"""
import contextlib
import itertools
import unittest

from collections import OrderedDict
from functools import wraps

import numpy as np

import dimod

__all__ = ['BQMTestCase']


class BQMTestCase(unittest.TestCase):
    """Utilities for writing test cases for the various BQM subclasses."""

    BQM_SUBCLASSES = set()

    @classmethod
    def register(cls, BQM):
        """Register a new subclass for testing."""
        cls.BQM_SUBCLASSES.add(BQM)

    @classmethod
    def multitest(cls, f):
        """Run the decorated method as a series of subTests, passing each BQM
        subclass in as an argument.
        """
        @wraps(f)
        def _wrapper(test_case):
            for BQM in cls.BQM_SUBCLASSES:
                with test_case.subTest(cls=BQM):
                    f(test_case, BQM)
        return _wrapper

    def assertConsistentBQM(self, bqm):
        """Check that the BQM's attributes are self-consistent."""

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
        """Brute force check that spin and binary bqms have idential energy
        for all possible samples.
        """
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


# pull into top-level namespace for convenience
multitest = BQMTestCase.multitest


class TestAddVariable(BQMTestCase):
    @multitest
    def test_bad_variable_type(self, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        bqm = BQM(dimod.SPIN)
        with self.assertRaises(TypeError):
            bqm.add_variable([])

    @multitest
    def test_index_labelled(self, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        bqm = BQM(dimod.SPIN)
        self.assertEqual(bqm.add_variable(1), 1)
        self.assertEqual(bqm.add_variable(), 0)  # 1 is already taken
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(bqm.add_variable(), 2)
        self.assertEqual(bqm.shape, (3, 0))

    @multitest
    def test_labelled(self, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        bqm = BQM(dimod.SPIN)
        bqm.add_variable('a')
        bqm.add_variable(1)
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(list(bqm.iter_variables()), ['a', 1])
        bqm.add_variable()
        self.assertEqual(bqm.shape, (3, 0))
        self.assertEqual(list(bqm.iter_variables()), ['a', 1, 2])

    @multitest
    def test_unlabelled(self, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        bqm = BQM(dimod.SPIN)
        bqm.add_variable()
        bqm.add_variable()
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(list(bqm.iter_variables()), [0, 1])


class TestChangeVartype(BQMTestCase):
    @multitest
    def test_change_vartype_binary_to_binary_copy(self, BQM):
        bqm = BQM({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, .4, 'BINARY')

        new = bqm.change_vartype(dimod.BINARY, inplace=False)
        self.assertEqual(bqm, new)
        self.assertIsNot(bqm, new)  # should be a copy

    @multitest
    def test_change_vartype_binary_to_spin_copy(self, BQM):
        bqm = BQM({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, .4, 'BINARY')

        # change vartype
        new = bqm.change_vartype(dimod.SPIN, inplace=False)

        self.assertConsistentEnergies(spin=new, binary=bqm)

    @multitest
    def test_change_vartype_spin_to_spin_copy(self, BQM):
        bqm = BQM({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, 1.4, 'SPIN')

        new = bqm.change_vartype(dimod.SPIN, inplace=False)
        self.assertEqual(bqm, new)
        self.assertIsNot(bqm, new)  # should be a copy

    @multitest
    def test_change_vartype_spin_to_binary_copy(self, BQM):
        bqm = BQM({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, 1.4, 'SPIN')

        # change vartype
        new = bqm.change_vartype(dimod.BINARY, inplace=False)

        self.assertConsistentEnergies(spin=bqm, binary=new)


class TestConstruction(BQMTestCase):
    @multitest
    def test_array_like(self, BQM):
        D = np.ones((5, 5)).tolist()
        bqm = BQM(D, 'BINARY')
        self.assertConsistentBQM(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)

        # with explicit kwarg
        bqm = BQM(D, vartype='BINARY')
        self.assertConsistentBQM(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)

    @multitest
    def test_array_like_1var(self, BQM):
        D = [[1]]
        bqm = BQM(D, 'BINARY')
        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm.shape, (1, 0))
        self.assertEqual(bqm.linear[0], 1)

    @multitest
    def test_array_like_spin(self, BQM):
        D = np.ones((5, 5)).tolist()
        bqm = BQM(D, 'SPIN')
        self.assertConsistentBQM(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 0)
        self.assertEqual(bqm.offset, 5)

    @multitest
    def test_array_linear(self, BQM):
        ldata = np.ones(5)
        qdata = np.ones((5, 5))
        bqm = BQM(ldata, qdata, 'BINARY')
        self.assertConsistentBQM(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 2)

    @multitest
    def test_array_linear_array_quadratic_spin(self, BQM):
        ldata = np.ones(5)
        qdata = np.ones((5, 5))
        bqm = BQM(ldata, qdata, 'SPIN')
        self.assertConsistentBQM(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)
        self.assertEqual(bqm.offset, 5)

    @multitest
    def test_array_linear_dict_quadratic_spin(self, BQM):
        ldata = np.ones(5)
        qdata = {(u, v): 1 for u in range(5) for v in range(5)}
        bqm = BQM(ldata, qdata, 'SPIN')
        self.assertConsistentBQM(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)
        self.assertEqual(bqm.offset, 5)

    def test_bqm_binary(self):
        linear = {'a': -1, 'b': 1, 0: 1.5}
        quadratic = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        offset = 0
        vartype = dimod.BINARY
        for source, target in itertools.product(self.BQM_SUBCLASSES, repeat=2):
            with self.subTest(source=source, target=target):
                bqm = source(linear, quadratic, offset, vartype)
                new = target(bqm)

                self.assertIsInstance(new, target)
                self.assertConsistentBQM(new)
                self.assertEqual(bqm.adj, new.adj)
                self.assertEqual(bqm.offset, new.offset)
                self.assertEqual(bqm.vartype, new.vartype)

    def test_bqm_spin(self):
        linear = {'a': -1, 'b': 1, 0: 1.5}
        quadratic = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        offset = 0
        vartype = dimod.SPIN
        for source, target in itertools.product(self.BQM_SUBCLASSES, repeat=2):
            with self.subTest(source=source, target=target):
                bqm = source(linear, quadratic, offset, vartype)
                new = target(bqm)

                self.assertIsInstance(new, target)
                self.assertConsistentBQM(new)
                self.assertEqual(bqm.adj, new.adj)
                self.assertEqual(bqm.offset, new.offset)
                self.assertEqual(bqm.vartype, new.vartype)

    def test_bqm_binary_to_spin(self):
        linear = {'a': -1, 'b': 1, 0: 1.5}
        quadratic = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        offset = 0
        vartype = dimod.BINARY
        for source, target in itertools.product(self.BQM_SUBCLASSES, repeat=2):
            with self.subTest(source=source, target=target):
                bqm = source(linear, quadratic, offset, vartype)
                new = target(bqm, vartype=dimod.SPIN)

                self.assertIsInstance(new, target)
                self.assertConsistentBQM(new)
                self.assertEqual(new.vartype, dimod.SPIN)

                # change back for equality check
                new.change_vartype(dimod.BINARY)
                self.assertEqual(bqm.adj, new.adj)
                self.assertEqual(bqm.offset, new.offset)
                self.assertEqual(bqm.vartype, new.vartype)

    def test_bqm_spin_to_binary(self):
        linear = {'a': -1, 'b': 1, 0: 1.5}
        quadratic = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        offset = 0
        vartype = dimod.SPIN
        for source, target in itertools.product(self.BQM_SUBCLASSES, repeat=2):
            with self.subTest(source=source, target=target):
                bqm = source(linear, quadratic, offset, vartype)
                new = target(bqm, vartype=dimod.BINARY)

                self.assertIsInstance(new, target)
                self.assertConsistentBQM(new)
                self.assertEqual(new.vartype, dimod.BINARY)

                # change back for equality check
                new.change_vartype(dimod.SPIN)
                self.assertEqual(bqm.adj, new.adj)
                self.assertEqual(bqm.offset, new.offset)
                self.assertEqual(bqm.vartype, new.vartype)

    @multitest
    def test_dense_zeros(self, BQM):
        # should ignore 0 off-diagonal
        D = np.zeros((5, 5))
        bqm = BQM(D, 'BINARY')
        self.assertEqual(bqm.shape, (5, 0))

    @multitest
    def test_integer(self, BQM):
        bqm = BQM(0, 'SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (0, 0))
        self.assertConsistentBQM(bqm)

        bqm = BQM(5, 'SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (5, 0))
        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm.linear, {v: 0 for v in range(5)})

    @multitest
    def test_legacy_bqm(self, BQM):
        lbqm = dimod.BinaryQuadraticModel.from_ising({'a': 2}, {'ab': -1}, 7)

        new = BQM(lbqm)

        self.assertEqual(lbqm.linear, new.linear)
        self.assertEqual(lbqm.adj, new.adj)
        self.assertEqual(lbqm.offset, new.offset)
        self.assertEqual(lbqm.vartype, new.vartype)

    @multitest
    def test_quadratic_only(self, BQM):
        M = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        bqm = BQM(M, 'BINARY')
        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm.linear, {0: 1, 1: 0, 2: 4, 4: 0, 5: 0})
        self.assertEqual(bqm.quadratic, {(0, 1): -1, (1, 2): 1.5, (4, 5): 7})

    @multitest
    def test_quadratic_only_spin(self, BQM):
        M = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        bqm = BQM(M, 'SPIN')
        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm.linear, {0: 0, 1: 0, 2: 0, 4: 0, 5: 0})
        self.assertEqual(bqm.quadratic, {(0, 1): -1, (1, 2): 1.5, (4, 5): 7})
        self.assertEqual(bqm.offset, 5)

    @multitest
    def test_no_args(self, BQM):
        with self.assertRaises(TypeError) as err:
            BQM()
        self.assertEqual(err.exception.args[0],
                         "A valid vartype or another bqm must be provided")

    @multitest
    def test_numpy_array(self, BQM):
        D = np.ones((5, 5))
        bqm = BQM(D, 'BINARY')
        self.assertConsistentBQM(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)

    @multitest
    def test_numpy_array_1var(self, BQM):
        D = np.ones((1, 1))
        bqm = BQM(D, 'BINARY')
        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm.shape, (1, 0))
        self.assertEqual(bqm.linear[0], 1)

    @multitest
    def test_vartype(self, BQM):
        bqm = BQM('SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)

        bqm = BQM(dimod.SPIN)
        self.assertEqual(bqm.vartype, dimod.SPIN)

        bqm = BQM((-1, 1))
        self.assertEqual(bqm.vartype, dimod.SPIN)

        bqm = BQM('BINARY')
        self.assertEqual(bqm.vartype, dimod.BINARY)

        bqm = BQM(dimod.BINARY)
        self.assertEqual(bqm.vartype, dimod.BINARY)

        bqm = BQM((0, 1))
        self.assertEqual(bqm.vartype, dimod.BINARY)

    @multitest
    def test_vartype_only(self, BQM):
        bqm = BQM('SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (0, 0))
        self.assertConsistentBQM(bqm)

        bqm = BQM(vartype='SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (0, 0))
        self.assertConsistentBQM(bqm)

        bqm = BQM('BINARY')
        self.assertEqual(bqm.vartype, dimod.BINARY)
        self.assertEqual(bqm.shape, (0, 0))
        self.assertConsistentBQM(bqm)

        bqm = BQM(vartype='BINARY')
        self.assertEqual(bqm.vartype, dimod.BINARY)
        self.assertEqual(bqm.shape, (0, 0))
        self.assertConsistentBQM(bqm)

    @multitest
    def test_vartype_readonly(self, BQM):
        bqm = BQM('SPIN')
        with self.assertRaises(AttributeError):
            bqm.vartype = dimod.BINARY


class TestCopy(BQMTestCase):
    @multitest
    def test_copy(self, BQM):
        bqm = BQM({'a': -1, 'b': 1}, {}, dimod.BINARY)
        new = bqm.copy()
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)

        # modify the original and make sure it doesn't propogate
        new.set_linear('a', 1)
        self.assertEqual(new.linear['a'], 1)

    @multitest
    def test_subclass(self, BQM):
        # copy should respect subclassing
        class SubBQM(BQM):
            pass

        bqm = SubBQM({'a': -1, 'b': 1}, {}, dimod.BINARY)
        new = bqm.copy()
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)


class TestEnergies(BQMTestCase):
    @multitest
    def test_2path(self, BQM):
        bqm = BQM([.1, -.2], [[0, -1], [0, 0]], 'SPIN')
        samples = [[-1, -1],
                   [-1, +1],
                   [+1, -1],
                   [+1, +1]]

        energies = bqm.energies(np.asarray(samples))

        np.testing.assert_array_almost_equal(energies, [-.9, .7, 1.3, -1.1])

    @multitest
    def test_5chain(self, BQM):
        arr = np.tril(np.triu(np.ones((5, 5)), 1), 1)
        bqm = BQM(arr, 'BINARY')
        samples = [[0, 0, 0, 0, 0]]

        energies = bqm.energies(np.asarray(samples))
        np.testing.assert_array_almost_equal(energies, [0])


class TestFromQUBO(BQMTestCase):
    @multitest
    def test_basic(self, BQM):
        Q = {(0, 0): -1, (0, 1): -1, (0, 2): -1, (1, 2): 1}
        bqm = BQM.from_qubo(Q)

        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm.linear, {0: -1, 1: 0, 2: 0})
        self.assertEqual(bqm.adj, {0: {1: -1, 2: -1},
                                   1: {0: -1, 2: 1},
                                   2: {0: -1, 1: 1}})
        self.assertEqual(bqm.offset, 0)

    @multitest
    def test_with_offset(self, BQM):
        Q = {(0, 0): -1, (0, 1): -1, (0, 2): -1, (1, 2): 1}
        bqm = BQM.from_qubo(Q, 1.6)

        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm.linear, {0: -1, 1: 0, 2: 0})
        self.assertEqual(bqm.adj, {0: {1: -1, 2: -1},
                                   1: {0: -1, 2: 1},
                                   2: {0: -1, 1: 1}})
        self.assertEqual(bqm.offset, 1.6)


class TestGetLinear(BQMTestCase):
    @multitest
    def test_disconnected_string_labels(self, BQM):
        bqm = BQM({'a': -1, 'b': 1}, {}, dimod.BINARY)
        self.assertEqual(bqm.get_linear('a'), -1)
        self.assertEqual(bqm.get_linear('b'), 1)
        with self.assertRaises(ValueError):
            bqm.get_linear('c')

    @multitest
    def test_disconnected(self, BQM):
        bqm = BQM(5, dimod.SPIN)

        for v in range(5):
            self.assertEqual(bqm.get_linear(v), 0)

        with self.assertRaises(ValueError):
            bqm.get_linear(-1)

        with self.assertRaises(ValueError):
            bqm.get_linear(5)

    @multitest
    def test_dtype(self, BQM):
        bqm = BQM(5, dimod.SPIN)

        # np.object_ does not play very nicely, even if it's accurate
        dtype = object if bqm.dtype.type is np.object_ else bqm.dtype.type

        for v in range(5):
            self.assertIsInstance(bqm.get_linear(v), dtype)


class TestGetQuadratic(BQMTestCase):
    @multitest
    def test_3x3array(self, BQM):
        bqm = BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]], dimod.SPIN)

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

    @multitest
    def test_default(self, BQM):
        bqm = BQM(5, 'SPIN')  # has no interactions
        with self.assertRaises(ValueError):
            bqm.get_quadratic(0, 1)
        self.assertEqual(bqm.get_quadratic(0, 1, default=5), 5)

    @multitest
    def test_dtype(self, BQM):
        bqm = BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]], dimod.SPIN)

        # np.object_ does not play very nicely, even if it's accurate
        dtype = object if bqm.dtype.type is np.object_ else bqm.dtype.type

        self.assertIsInstance(bqm.get_quadratic(0, 1), dtype)
        self.assertIsInstance(bqm.get_quadratic(1, 0), dtype)

        self.assertIsInstance(bqm.get_quadratic(0, 2), dtype)
        self.assertIsInstance(bqm.get_quadratic(2, 0), dtype)


class TestHasVariable(BQMTestCase):
    @multitest
    def test_typical(self, BQM):
        h = OrderedDict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = BQM(h, J, dimod.SPIN)

        self.assertTrue(bqm.has_variable('a'))
        self.assertTrue(bqm.has_variable(1))
        self.assertTrue(bqm.has_variable(3))

        # no false positives
        self.assertFalse(bqm.has_variable(0))
        self.assertFalse(bqm.has_variable(2))


class TestIteration(BQMTestCase):
    @multitest
    def test_iter_quadratic_neighbours(self, BQM):
        bqm = BQM({'ab': -1, 'bc': 21, 'cd': 1}, dimod.SPIN)
        neighbours = set(bqm.iter_quadratic('b'))
        self.assertEqual(neighbours,
                         {('b', 'a', -1), ('b', 'c', 21)})

    @multitest
    def test_iter_quadratic_neighbours_bunch(self, BQM):
        bqm = BQM({'bc': 21, 'cd': 1}, dimod.SPIN)
        self.assertEqual(list(bqm.iter_quadratic(['b', 'c'])),
                         [('b', 'c', 21.0), ('c', 'd', 1.0)])

    @multitest
    def test_iter_variables(self, BQM):
        h = OrderedDict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = BQM(h, J, dimod.SPIN)

        self.assertEqual(list(bqm.iter_variables()), ['a', 1, 3])


class TestLen(BQMTestCase):
    @multitest
    def test__len__(self, BQM):
        bqm = BQM(np.ones((107, 107)), dimod.BINARY)
        self.assertEqual(len(bqm), 107)


class TestOffset(BQMTestCase):
    @multitest
    def test_offset(self, BQM):
        h = dict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = BQM(h, J, 'SPIN')
        self.assertEqual(bqm.offset, 0)
        dtype = object if bqm.dtype.type is np.object_ else bqm.dtype.type
        self.assertIsInstance(bqm.offset, dtype)

        h = dict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = BQM(h, J, 1.5, 'SPIN')
        self.assertEqual(bqm.offset, 1.5)
        dtype = object if bqm.dtype.type is np.object_ else bqm.dtype.type
        self.assertIsInstance(bqm.offset, dtype)

        bqm.offset = 6
        self.assertEqual(bqm.offset, 6)
        dtype = object if bqm.dtype.type is np.object_ else bqm.dtype.type
        self.assertIsInstance(bqm.offset, dtype)


class TestRemoveInteraction(BQMTestCase):
    @multitest
    def test_basic(self, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest
        bqm = BQM(np.triu(np.ones((3, 3))), dimod.BINARY)
        bqm.remove_interaction(0, 1)
        with self.assertRaises(ValueError):
            bqm.remove_interaction(0, 1)
        self.assertEqual(bqm.shape, (3, 2))

        with self.assertRaises(ValueError):
            bqm.remove_interaction('a', 1)  # 'a' is not a variable

        with self.assertRaises(ValueError):
            bqm.remove_interaction(1, 1)


class TestRemoveVariable(BQMTestCase):
    @multitest
    def test_labelled(self, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest
        bqm = BQM(dimod.SPIN)
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

    @multitest
    def test_provided(self, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest
        bqm = BQM('SPIN')
        bqm.add_variable('a')
        bqm.add_variable('b')
        bqm.add_variable('c')

        bqm.remove_variable('b')
        self.assertConsistentBQM(bqm)

        # maintained order
        self.assertEqual(list(bqm.iter_variables()), ['a', 'c'])

        with self.assertRaises(ValueError):
            bqm.remove_variable('b')

    @multitest
    def test_unlabelled(self, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest
        bqm = BQM(2, dimod.BINARY)
        self.assertEqual(bqm.remove_variable(), 1)
        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm.remove_variable(), 0)
        self.assertConsistentBQM(bqm)
        with self.assertRaises(ValueError):
            bqm.remove_variable()
        self.assertConsistentBQM(bqm)


class TestSetLinear(BQMTestCase):
    @multitest
    def test_basic(self, BQM):
        # does not change shape
        bqm = BQM(np.triu(np.ones((3, 3))), dimod.BINARY)

        self.assertEqual(bqm.get_linear(0), 1)
        bqm.set_linear(0, .5)
        self.assertEqual(bqm.get_linear(0), .5)


class TestSetQuadratic(BQMTestCase):
    @multitest
    def test_basic(self, BQM):
        # does not change shape
        bqm = BQM(np.triu(np.ones((3, 3))), dimod.BINARY)

        self.assertEqual(bqm.get_quadratic(0, 1), 1)
        bqm.set_quadratic(0, 1, .5)
        self.assertEqual(bqm.get_quadratic(0, 1), .5)
        self.assertEqual(bqm.get_quadratic(1, 0), .5)
        bqm.set_quadratic(0, 1, -.5)
        self.assertEqual(bqm.get_quadratic(0, 1), -.5)
        self.assertEqual(bqm.get_quadratic(1, 0), -.5)

    @multitest
    def test_set_quadratic_exception(self, BQM):
        bqm = BQM(dimod.SPIN)
        if BQM.shapeable():
            with self.assertRaises(TypeError):
                bqm.set_quadratic([], 1, .5)
            with self.assertRaises(TypeError):
                bqm.set_quadratic(1, [], .5)
        else:
            with self.assertRaises(ValueError):
                bqm.set_quadratic([], 1, .5)
            with self.assertRaises(ValueError):
                bqm.set_quadratic(1, [], .5)


class TestShape(BQMTestCase):
    @multitest
    def test_3x3array(self, BQM):
        bqm = BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]], dimod.BINARY)

        self.assertEqual(bqm.shape, (3, 2))
        self.assertEqual(bqm.num_variables, 3)
        self.assertEqual(bqm.num_interactions, 2)

    @multitest
    def test_disconnected(self, BQM):
        bqm = BQM(5, dimod.BINARY)

        self.assertEqual(bqm.shape, (5, 0))
        self.assertEqual(bqm.num_variables, 5)
        self.assertEqual(bqm.num_interactions, 0)

    @multitest
    def test_empty(self, BQM):
        self.assertEqual(BQM(dimod.SPIN).shape, (0, 0))
        self.assertEqual(BQM(0, dimod.SPIN).shape, (0, 0))

        self.assertEqual(BQM(dimod.SPIN).num_variables, 0)
        self.assertEqual(BQM(0, dimod.SPIN).num_variables, 0)

        self.assertEqual(BQM(dimod.SPIN).num_interactions, 0)
        self.assertEqual(BQM(0, dimod.SPIN).num_interactions, 0)


class TestToNumpyVectors(BQMTestCase):
    @multitest
    def test_array_dense(self, BQM):
        bqm = BQM(np.arange(9).reshape((3, 3)), 'BINARY')

        ldata, (irow, icol, qdata), off = bqm.to_numpy_vectors()

        np.testing.assert_array_equal(ldata, [0, 4, 8])

        self.assertEqual(len(irow), len(icol))
        self.assertEqual(len(icol), len(qdata))
        self.assertEqual(len(qdata), len(bqm.quadratic))
        for u, v, bias in zip(irow, icol, qdata):
            self.assertAlmostEqual(bqm.adj[u][v], bias)

    @multitest
    def test_array_reversed_order(self, BQM):
        bqm = BQM(np.arange(9).reshape((3, 3)), 'BINARY')

        order = [2, 1, 0]
        ldata, (irow, icol, qdata), off \
            = bqm.to_numpy_vectors(variable_order=order)

        np.testing.assert_array_equal(ldata, [8, 4, 0])

        self.assertEqual(len(irow), len(icol))
        self.assertEqual(len(icol), len(qdata))
        self.assertEqual(len(qdata), len(bqm.quadratic))
        for ui, vi, bias in zip(irow, icol, qdata):
            u = order[ui]
            v = order[vi]
            self.assertAlmostEqual(bqm.adj[u][v], bias)

    @multitest
    def test_array_sparse(self, BQM):
        arr = np.arange(9).reshape((3, 3))
        arr[1, 2] = arr[2, 1] = 0
        bqm = BQM(arr, 'BINARY')
        self.assertEqual(bqm.shape, (3, 2))  # sparse

        ldata, (irow, icol, qdata), off = bqm.to_numpy_vectors()

        np.testing.assert_array_equal(ldata, [0, 4, 8])

        self.assertEqual(len(irow), len(icol))
        self.assertEqual(len(icol), len(qdata))
        self.assertEqual(len(qdata), len(bqm.quadratic))
        for u, v, bias in zip(irow, icol, qdata):
            self.assertAlmostEqual(bqm.adj[u][v], bias)

    @multitest
    def test_array_sparse_return_labels(self, BQM):
        arr = np.arange(9).reshape((3, 3))
        arr[1, 2] = arr[2, 1] = 0
        bqm = BQM(arr, 'BINARY')
        self.assertEqual(bqm.shape, (3, 2))  # sparse

        ldata, (irow, icol, qdata), off, labels \
            = bqm.to_numpy_vectors(return_labels=True)

        self.assertEqual(labels, list(range(3)))

        np.testing.assert_array_equal(ldata, [0, 4, 8])

        self.assertEqual(len(irow), len(icol))
        self.assertEqual(len(icol), len(qdata))
        self.assertEqual(len(qdata), len(bqm.quadratic))
        for u, v, bias in zip(irow, icol, qdata):
            self.assertAlmostEqual(bqm.adj[u][v], bias)

    @multitest
    def test_dict(self, BQM):
        bqm = BQM({'c': 1, 'a': -1}, {'ba': 1, 'bc': -2}, 0, dimod.SPIN)

        # these values are sortable, so returned order should be a,b,c
        order = 'abc'
        ldata, (irow, icol, qdata), off = bqm.to_numpy_vectors()

        np.testing.assert_array_equal(ldata, [-1, 0, 1])

        self.assertEqual(len(irow), len(icol))
        self.assertEqual(len(icol), len(qdata))
        self.assertEqual(len(qdata), len(bqm.quadratic))
        for ui, vi, bias in zip(irow, icol, qdata):
            u = order[ui]
            v = order[vi]
            self.assertAlmostEqual(bqm.adj[u][v], bias)

    @multitest
    def test_dict_return_labels(self, BQM):
        bqm = BQM({'c': 1, 'a': -1}, {'ba': 1, 'bc': -2}, 0, dimod.SPIN)

        # these values are sortable, so returned order should be a,b,c
        ldata, (irow, icol, qdata), off, order \
            = bqm.to_numpy_vectors(return_labels=True)

        self.assertEqual(order, list('abc'))

        np.testing.assert_array_equal(ldata, [-1, 0, 1])

        self.assertEqual(len(irow), len(icol))
        self.assertEqual(len(icol), len(qdata))
        self.assertEqual(len(qdata), len(bqm.quadratic))
        for ui, vi, bias in zip(irow, icol, qdata):
            u = order[ui]
            v = order[vi]
            self.assertAlmostEqual(bqm.adj[u][v], bias)

    @multitest
    def test_empty(self, BQM):
        bqm = BQM('SPIN')
        h, (i, j, values), off = bqm.to_numpy_vectors()

        np.testing.assert_array_equal(h, [])
        np.testing.assert_array_equal(i, [])
        np.testing.assert_array_equal(j, [])
        np.testing.assert_array_equal(values, [])
        self.assertEqual(off, 0)

    @multitest
    def test_unsorted_labels(self, BQM):
        bqm = BQM(OrderedDict([('b', -1), ('a', 1)]), {}, 'SPIN')

        ldata, (irow, icol, qdata), off, order \
            = bqm.to_numpy_vectors(return_labels=True, sort_labels=False)

        self.assertEqual(order, ['b', 'a'])

        np.testing.assert_array_equal(ldata, [-1, 1])
        np.testing.assert_array_equal(irow, [])
        np.testing.assert_array_equal(icol, [])
        np.testing.assert_array_equal(qdata, [])
        self.assertEqual(off, 0)

    @multitest
    def test_sort_indices(self, BQM):
        bqm = BQM.from_ising({}, {(0, 1): .5, (3, 2): -1, (0, 3): 1.5})

        h, (i, j, values), off = bqm.to_numpy_vectors(sort_indices=True)

        np.testing.assert_array_equal(h, [0, 0, 0, 0])
        np.testing.assert_array_equal(i, [0, 0, 2])
        np.testing.assert_array_equal(j, [1, 3, 3])
        np.testing.assert_array_equal(values, [.5, 1.5, -1])


class TestViews(BQMTestCase):
    @multitest
    def test_adj_setitem(self, BQM):
        bqm = BQM({'ab': -1}, 'SPIN')
        bqm.adj['a']['b'] = 5
        self.assertEqual(bqm.adj['a']['b'], 5)
        self.assertConsistentBQM(bqm)  # all the other cases

    @multitest
    def test_adj_neighborhoods(self, BQM):
        bqm = BQM({'ab': -1, 'ac': -1, 'bc': -1, 'cd': -1}, 'SPIN')

        self.assertEqual(len(bqm.adj['a']), 2)
        self.assertEqual(len(bqm.adj['b']), 2)
        self.assertEqual(len(bqm.adj['c']), 3)
        self.assertEqual(len(bqm.adj['d']), 1)

    @multitest
    def test_linear_delitem(self, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest
        bqm = BQM([[0, 1, 2, 3, 4],
                   [0, 6, 7, 8, 9],
                   [0, 0, 10, 11, 12],
                   [0, 0, 0, 13, 14],
                   [0, 0, 0, 0, 15]], 'BINARY')
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

    @multitest
    def test_linear_setitem(self, BQM):
        bqm = BQM({'ab': -1}, dimod.SPIN)
        bqm.linear['a'] = 5
        self.assertEqual(bqm.get_linear('a'), 5)
        self.assertConsistentBQM(bqm)

    @multitest
    def test_quadratic_delitem(self, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest
        bqm = BQM([[0, 1, 2, 3, 4],
                   [0, 6, 7, 8, 9],
                   [0, 0, 10, 11, 12],
                   [0, 0, 0, 13, 14],
                   [0, 0, 0, 0, 15]], 'SPIN')
        del bqm.quadratic[0, 1]
        self.assertEqual(set(bqm.iter_neighbors(0)), set([2, 3, 4]))
        self.assertConsistentBQM(bqm)

        with self.assertRaises(KeyError):
            del bqm.quadratic[0, 1]

    @multitest
    def test_quadratic_setitem(self, BQM):
        bqm = BQM({'ab': -1}, dimod.SPIN)
        bqm.quadratic[('a', 'b')] = 5
        self.assertEqual(bqm.get_quadratic('a', 'b'), 5)
        self.assertConsistentBQM(bqm)
