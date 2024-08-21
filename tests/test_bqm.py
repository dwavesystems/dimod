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

"""Generic dimod/bqm tests."""
import itertools
import json
import fractions
import numbers
import operator
import os.path as path
import shutil
import tempfile
import unittest
import unittest.mock

from collections import OrderedDict
from functools import wraps

import numpy as np

from parameterized import parameterized

import dimod

from dimod.binary import BinaryQuadraticModel, DictBQM, Float32BQM, Float64BQM
from dimod.binary import as_bqm
from dimod.binary import Spin, Binary
from dimod.testing import assert_consistent_bqm, assert_bqm_almost_equal


def cross_vartype_view(*args, **kwargs):
    bqm = BinaryQuadraticModel(*args, **kwargs)
    if bqm.vartype is dimod.SPIN:
        bqm.change_vartype(dimod.BINARY)
        return bqm.spin
    else:
        bqm.change_vartype(dimod.SPIN)
        return bqm.binary


def vartype_view(*args, **kwargs):
    bqm = BinaryQuadraticModel(*args, **kwargs)
    if bqm.vartype is dimod.SPIN:
        return bqm.spin
    else:
        return bqm.binary


BQMs = dict(BinaryQuadraticModel=BinaryQuadraticModel,
            DictBQM=DictBQM,
            Float32BQM=Float32BQM,
            Float64BQM=Float64BQM,
            VartypeView=vartype_view,
            CrossVartypeView=cross_vartype_view,
            )

BQM_CLSs = dict((k, v) for k, v in BQMs.items() if isinstance(v, type))


class TestAddOffset(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_typical(self, name, BQM):
        bqm = BQM({}, {'ab': -1}, 1.5, 'SPIN')
        with self.assertWarns(DeprecationWarning):
            bqm.add_offset(2)
        self.assertEqual(bqm.offset, 3.5)


class TestAddVariable(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_bad_variable_type(self, name, BQM):
        bqm = BQM(dimod.SPIN)
        with self.assertRaises(TypeError):
            bqm.add_variable([])

    @parameterized.expand(BQMs.items())
    def test_bias_new_variable(self, name, BQM):
        bqm = BQM(dimod.BINARY)
        bqm.add_variable(bias=5)

        self.assertEqual(bqm.linear, {0: 5})

        bqm.add_variable('a', -6)
        self.assertEqual(bqm.linear, {0: 5, 'a': -6})

    @parameterized.expand(BQMs.items())
    def test_bias_additive(self, name, BQM):
        bqm = BQM(dimod.BINARY)
        bqm.add_variable(bqm.add_variable(bias=3), 3)

        self.assertEqual(bqm.linear, {0: 6})

    @parameterized.expand(BQMs.items())
    def test_index_labelled(self, name, BQM):
        bqm = BQM(dimod.SPIN)
        self.assertEqual(bqm.add_variable(1), 1)
        self.assertEqual(bqm.add_variable(), 0)  # 1 is already taken
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(bqm.add_variable(), 2)
        self.assertEqual(bqm.shape, (3, 0))

    @parameterized.expand(BQMs.items())
    def test_labelled(self, name, BQM):
        bqm = BQM(dimod.SPIN)
        bqm.add_variable('a')
        bqm.add_variable(1)
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(list(bqm.variables), ['a', 1])
        bqm.add_variable()
        self.assertEqual(bqm.shape, (3, 0))
        self.assertEqual(list(bqm.variables), ['a', 1, 2])

    @parameterized.expand(BQMs.items())
    def test_unlabelled(self, name, BQM):
        bqm = BQM(dimod.SPIN)
        bqm.add_variable()
        bqm.add_variable()
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(list(bqm.variables), [0, 1])


class TestAddVariablesFrom(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_iterable(self, name, BQM):
        # add from 2-tuples
        bqm = BQM(dimod.SPIN)
        bqm.add_variables_from(iter([('a', .5), ('b', -.5)]))

        self.assertEqual(bqm.linear, {'a': .5, 'b': -.5})

    @parameterized.expand(BQMs.items())
    def test_mapping(self, name, BQM):
        bqm = BQM(dimod.SPIN)
        bqm.add_variables_from({'a': .5, 'b': -.5})

        self.assertEqual(bqm.linear, {'a': .5, 'b': -.5})

        # check that it's additive
        bqm.add_variables_from({'a': -1, 'b': 3, 'c': 4})

        self.assertEqual(bqm.linear, {'a': -.5, 'b': 2.5, 'c': 4})


class TestAddInteractionsFrom(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_iterable(self, name, BQM):
        bqm = BQM(dimod.SPIN)
        bqm.add_interactions_from({('a', 'b'): -.5})
        self.assertEqual(bqm.adj, {'a': {'b': -.5},
                                   'b': {'a': -.5}})

    @parameterized.expand(BQMs.items())
    def test_mapping(self, name, BQM):
        bqm = BQM(dimod.SPIN)
        bqm.add_interactions_from([('a', 'b', -.5)])
        self.assertEqual(bqm.adj, {'a': {'b': -.5},
                                   'b': {'a': -.5}})


class TestAdjacency(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_contains(self, name, BQM):
        bqm = BQM({0: 1.0}, {(0, 1): 2.0, (2, 1): 0.4}, 0.0, dimod.SPIN)

        self.assertIn(0, bqm.adj[1])
        self.assertEqual(2.0, bqm.adj[1][0])
        self.assertIn(1, bqm.adj[0])
        self.assertEqual(2.0, bqm.adj[0][1])

        self.assertIn(2, bqm.adj[1])
        self.assertAlmostEqual(.4, bqm.adj[1][2])
        self.assertIn(1, bqm.adj[2])
        self.assertAlmostEqual(.4, bqm.adj[2][1])

        self.assertNotIn(2, bqm.adj[0])
        with self.assertRaises(KeyError):
            bqm.adj[0][2]
        self.assertNotIn(0, bqm.adj[2])
        with self.assertRaises(KeyError):
            bqm.adj[2][0]


class TestAsBQM(unittest.TestCase):
    def test_basic(self):
        bqm = as_bqm({0: -1}, {(0, 1): 5}, 1.6, dimod.SPIN)

        assert_consistent_bqm(bqm)

    @parameterized.expand(BQMs.items())
    def test_bqm_input(self, name, BQM):
        bqm = BQM({'ab': -1}, dimod.BINARY)

        self.assertIs(as_bqm(bqm), bqm)
        self.assertEqual(as_bqm(bqm), bqm)
        self.assertIsNot(as_bqm(bqm, copy=True), bqm)
        self.assertEqual(as_bqm(bqm, copy=True), bqm)

    @parameterized.expand(BQMs.items())
    def test_bqm_input_change_vartype(self, name, BQM):
        bqm = BQM({'ab': -1}, dimod.BINARY)

        self.assertEqual(as_bqm(bqm, 'SPIN').vartype, dimod.SPIN)

        self.assertIs(as_bqm(bqm, 'BINARY'), bqm)
        self.assertIsNot(as_bqm(bqm, 'BINARY', copy=True), bqm)
        self.assertEqual(as_bqm(bqm, 'BINARY', copy=True), bqm)

    def test_cls(self):
        bqm = BinaryQuadraticModel({'ab': -1}, dimod.BINARY)
        with self.assertWarns(DeprecationWarning):
            as_bqm(bqm, cls=123)


class TestBinary(unittest.TestCase):
    def test_init_no_label(self):
        binary_bqm = Binary()
        self.assertIsInstance(binary_bqm.variables[0], str)

    def test_binary_array_int_init(self):
        binary_array = dimod.BinaryArray(3)
        self.assertIsInstance(binary_array, np.ndarray)
        for element in binary_array:
            self.assertIsInstance(element, BinaryQuadraticModel)

    def test_binary_array_label_init(self):
        labels = 'ijk'
        binary_array = dimod.BinaryArray(labels=labels)
        self.assertIsInstance(binary_array, np.ndarray)
        self.assertEqual(len(binary_array), len(labels))

    def test_multiple_labelled(self):
        x, y, z = dimod.Binaries('abc')

        self.assertEqual(x.variables[0], 'a')
        self.assertEqual(y.variables[0], 'b')
        self.assertEqual(z.variables[0], 'c')
        self.assertIs(x.vartype, dimod.BINARY)
        self.assertIs(y.vartype, dimod.BINARY)
        self.assertIs(z.vartype, dimod.BINARY)

    def test_multiple_unlabelled(self):
        x, y, z = dimod.Binaries(3)

        self.assertNotEqual(x.variables[0], y.variables[0])
        self.assertNotEqual(x.variables[0], z.variables[0])
        self.assertIs(x.vartype, dimod.BINARY)
        self.assertIs(y.vartype, dimod.BINARY)
        self.assertIs(z.vartype, dimod.BINARY)

    def test_no_label_collision(self):
        bqm_1 = Binary()
        bqm_2 = Binary()
        self.assertNotEqual(bqm_1.variables[0], bqm_2.variables[0])


class TestChangeVartype(unittest.TestCase):
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

            self.assertAlmostEqual(spin_energy, binary_energy, places=5)

    @parameterized.expand(BQMs.items())
    def test_change_vartype_binary_to_binary_copy(self, name, BQM):
        bqm = BQM({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, .4, 'BINARY')

        new = bqm.change_vartype(dimod.BINARY, inplace=False)
        self.assertEqual(bqm, new)
        self.assertIsNot(bqm, new)  # should be a copy

    @parameterized.expand(BQMs.items())
    def test_change_vartype_binary_to_spin_copy(self, name, BQM):
        bqm = BQM({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, .4, 'BINARY')

        # change vartype
        new = bqm.change_vartype(dimod.SPIN, inplace=False)

        self.assertConsistentEnergies(spin=new, binary=bqm)

    @parameterized.expand(BQMs.items())
    def test_change_vartype_spin_to_spin_copy(self, name, BQM):
        bqm = BQM({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, 1.4, 'SPIN')

        new = bqm.change_vartype(dimod.SPIN, inplace=False)
        self.assertEqual(bqm, new)
        self.assertIsNot(bqm, new)  # should be a copy

    @parameterized.expand(BQMs.items())
    def test_change_vartype_spin_to_binary_copy(self, name, BQM):
        bqm = BQM({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, 1.4, 'SPIN')

        # change vartype
        new = bqm.change_vartype(dimod.BINARY, inplace=False)

        self.assertConsistentEnergies(spin=bqm, binary=new)


class TestClear(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_clear(self, name, BQM):
        bqm = BQM(np.ones((5, 5)), 'BINARY')
        bqm.clear()
        self.assertEqual(bqm.num_variables, 0)
        self.assertEqual(bqm.num_interactions, 0)
        self.assertEqual(bqm.offset, 0)
        self.assertEqual(len(bqm.variables), 0)


class TestConstruction(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_array_like(self, name, BQM):
        D = np.ones((5, 5)).tolist()
        bqm = BQM(D, 'BINARY')
        assert_consistent_bqm(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)

        # with explicit kwarg
        bqm = BQM(D, vartype='BINARY')
        assert_consistent_bqm(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)

    @parameterized.expand(BQMs.items())
    def test_array_like_1var(self, name, BQM):
        D = [[1]]
        bqm = BQM(D, 'BINARY')
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.shape, (1, 0))
        self.assertEqual(bqm.linear[0], 1)

    @parameterized.expand(BQMs.items())
    def test_array_like_spin(self, name, BQM):
        D = np.ones((5, 5)).tolist()
        bqm = BQM(D, 'SPIN')
        assert_consistent_bqm(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 0)
        self.assertEqual(bqm.offset, 5)

    @parameterized.expand(BQMs.items())
    def test_array_linear(self, name, BQM):
        ldata = np.ones(5)
        qdata = np.ones((5, 5))
        bqm = BQM(ldata, qdata, 'BINARY')
        assert_consistent_bqm(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 2)

    @parameterized.expand(BQMs.items())
    def test_array_linear_array_quadratic_spin(self, name, BQM):
        ldata = np.ones(5)
        qdata = np.ones((5, 5))
        bqm = BQM(ldata, qdata, 'SPIN')
        assert_consistent_bqm(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)
        self.assertEqual(bqm.offset, 5)

    @parameterized.expand(BQMs.items())
    def test_array_linear_dict_quadratic_spin(self, name, BQM):
        ldata = np.ones(5)
        qdata = {(u, v): 1 for u in range(5) for v in range(5)}
        bqm = BQM(ldata, qdata, 'SPIN')
        assert_consistent_bqm(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)
        self.assertEqual(bqm.offset, 5)

    @parameterized.expand(BQMs.items())
    def test_array_types(self, name, BQM):
        # comes from a bug where this was returning an array
        h = [0, 1, 2]
        J = np.asarray([[0, 1, 2], [0, 0, 3], [0, 0, 0]])
        bqm = BQM(h, J, 'SPIN')
        for bias in bqm.quadratic.values():
            self.assertIsInstance(bias, numbers.Number)

    def test_bqm_binary(self):
        linear = {'a': -1, 'b': 1, 0: 1.5}
        quadratic = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        offset = 0
        vartype = dimod.BINARY
        for source, target in itertools.product(BQMs.values(), repeat=2):
            with self.subTest(source=source, target=target):
                bqm = source(linear, quadratic, offset, vartype)
                new = target(bqm)

                assert_consistent_bqm(new)
                self.assertEqual(bqm.adj, new.adj)
                self.assertEqual(bqm.offset, new.offset)
                self.assertEqual(bqm.vartype, new.vartype)

                if isinstance(target, type):
                    self.assertIsInstance(new, target)

    def test_bqm_spin(self):
        linear = {'a': -1, 'b': 1, 0: 1.5}
        quadratic = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        offset = 0
        vartype = dimod.SPIN
        for source, target in itertools.product(BQMs.values(), repeat=2):
            with self.subTest(source=source, target=target):
                bqm = source(linear, quadratic, offset, vartype)
                new = target(bqm)

                assert_consistent_bqm(new)
                self.assertEqual(bqm.adj, new.adj)
                self.assertEqual(bqm.offset, new.offset)
                self.assertEqual(bqm.vartype, new.vartype)

                if isinstance(target, type):
                    self.assertIsInstance(new, target)

    def test_bqm_binary_to_spin(self):
        linear = {'a': -1, 'b': 1, 0: 1.5}
        quadratic = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        offset = 0
        vartype = dimod.BINARY
        for source, target in itertools.product(BQMs.values(), repeat=2):
            with self.subTest(source=source, target=target):
                bqm = source(linear, quadratic, offset, vartype)
                new = target(bqm, vartype=dimod.SPIN)

                assert_consistent_bqm(new)
                self.assertEqual(new.vartype, dimod.SPIN)

                # change back for equality check
                new.change_vartype(dimod.BINARY)
                self.assertEqual(bqm.adj, new.adj)
                self.assertEqual(bqm.offset, new.offset)
                self.assertEqual(bqm.vartype, new.vartype)

                if isinstance(target, type):
                    self.assertIsInstance(new, target)

    def test_bqm_spin_to_binary(self):
        linear = {'a': -1, 'b': 1, 0: 1.5}
        quadratic = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        offset = 0
        vartype = dimod.SPIN
        for source, target in itertools.product(BQMs.values(), repeat=2):
            with self.subTest(source=source, target=target):
                bqm = source(linear, quadratic, offset, vartype)
                new = target(bqm, vartype=dimod.BINARY)

                assert_consistent_bqm(new)
                self.assertEqual(new.vartype, dimod.BINARY)

                # change back for equality check
                new.change_vartype(dimod.SPIN)
                self.assertEqual(bqm.adj, new.adj)
                self.assertEqual(bqm.offset, new.offset)
                self.assertEqual(bqm.vartype, new.vartype)

                if isinstance(target, type):
                    self.assertIsInstance(new, target)

    @parameterized.expand(BQMs.items())
    def test_dense_zeros(self, name, BQM):
        # should ignore 0 off-diagonal
        D = np.zeros((5, 5))
        bqm = BQM(D, 'BINARY')
        self.assertEqual(bqm.shape, (5, 0))

    def test_DictBQM(self):
        self.assertEqual(DictBQM('SPIN').dtype, object)
        self.assertEqual(DictBQM({'a': 1}, {}, 1.5, 'SPIN').dtype, object)

    @parameterized.expand(BQMs.items())
    def test_integer(self, name, BQM):
        bqm = BQM(0, 'SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (0, 0))
        assert_consistent_bqm(bqm)

        bqm = BQM(5, 'SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (5, 0))
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.linear, {v: 0 for v in range(5)})

    @parameterized.expand(BQMs.items())
    def test_iterator_2arg(self, name, BQM):
        Q = ((u, v, -1) for u in range(5) for v in range(u+1, 5))
        bqm = BQM(Q, dimod.BINARY)

        self.assertEqual(bqm.shape, (5, 10))

    @parameterized.expand(BQMs.items())
    def test_iterator_3arg(self, name, BQM):
        h = ((v, 1) for v in range(5))
        J = ((u, v, -1) for u in range(5) for v in range(u+1, 5))
        bqm = BQM(h, J, dimod.SPIN)

        self.assertEqual(bqm.shape, (5, 10))

    @parameterized.expand(BQMs.items())
    def test_linear_array_quadratic_array(self, name, BQM):
        h = [1, 2, 3, 4, 5]
        J = np.zeros((5, 5))
        bqm = BQM(h, J, 1.2, 'SPIN')

        self.assertEqual(bqm.linear, {v: v+1 for v in range(5)})
        self.assertEqual(bqm.quadratic, {})
        self.assertAlmostEqual(bqm.offset, 1.2)
        self.assertIs(bqm.vartype, dimod.SPIN)

    @parameterized.expand(BQMs.items())
    def test_linear_array_quadratic_dict(self, name, BQM):
        h = [1, 2, 3, 4, 5]
        J = {'ab': -1}
        bqm = BQM(h, J, 1.2, 'SPIN')

        htarget = {v: v+1 for v in range(5)}
        htarget.update(a=0, b=0)
        adj_target = {v: {} for v in range(5)}
        adj_target.update(a=dict(b=-1), b=dict(a=-1))
        self.assertEqual(bqm.linear, htarget)
        self.assertEqual(bqm.adj, adj_target)
        self.assertAlmostEqual(bqm.offset, 1.2)
        self.assertIs(bqm.vartype, dimod.SPIN)

    @parameterized.expand(BQMs.items())
    def test_quadratic_only(self, name, BQM):
        M = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        bqm = BQM(M, 'BINARY')
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.linear, {0: 1, 1: 0, 2: 4, 4: 0, 5: 0})
        self.assertEqual(bqm.quadratic, {(0, 1): -1, (1, 2): 1.5, (4, 5): 7})

    @parameterized.expand(BQMs.items())
    def test_quadratic_only_spin(self, name, BQM):
        M = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        bqm = BQM(M, 'SPIN')
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.linear, {0: 0, 1: 0, 2: 0, 4: 0, 5: 0})
        self.assertEqual(bqm.quadratic, {(0, 1): -1, (1, 2): 1.5, (4, 5): 7})
        self.assertAlmostEqual(bqm.offset, 5)

    @parameterized.expand(BQMs.items())
    def test_no_args(self, name, BQM):
        with self.assertRaises(TypeError) as err:
            BQM()
        self.assertEqual(err.exception.args[0],
                         "A valid vartype or another bqm must be provided")

    @parameterized.expand(BQMs.items())
    def test_numpy_array(self, name, BQM):
        D = np.ones((5, 5))
        bqm = BQM(D, 'BINARY')
        assert_consistent_bqm(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)

    @parameterized.expand(BQMs.items())
    def test_numpy_array_1var(self, name, BQM):
        D = np.ones((1, 1))
        bqm = BQM(D, 'BINARY')
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.shape, (1, 0))
        self.assertEqual(bqm.linear[0], 1)

    def test_offset_kwarg(self):
        # the various constructions but with a kwarg
        with self.subTest('vartype only'):
            bqm = dimod.BQM(vartype='SPIN', offset=7)
            self.assertEqual(bqm.shape, (0, 0))
            self.assertIs(bqm.vartype, dimod.SPIN)
            self.assertEqual(bqm.offset, 7)

        with self.subTest('bqm'):
            bqm = dimod.BQM('SPIN')
            with self.assertRaises(TypeError):
                dimod.BQM(bqm, offset=5)
            with self.assertRaises(TypeError):
                dimod.BQM(bqm, vartype='BINARY', offset=5)

        with self.subTest('integer'):
            bqm = dimod.BQM(5, offset=5, vartype='SPIN')
            self.assertEqual(bqm.num_variables, 5)
            self.assertEqual(bqm.linear, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
            self.assertTrue(bqm.is_linear())
            self.assertEqual(bqm.offset, 5)

        with self.subTest('linear/quadratic'):
            bqm = dimod.BQM({'a': 1}, {'ab': 2}, offset=6, vartype='SPIN')
            self.assertEqual(bqm.shape, (2, 1))
            self.assertEqual(bqm.offset, 6)
            self.assertIs(bqm.vartype, dimod.SPIN)

        with self.subTest('linear/quadratic/offset'):
            with self.assertRaises(TypeError):
                dimod.BQM({}, {}, 1.5, 'SPIN', offset=5)

    @parameterized.expand(BQMs.items())
    def test_vartype(self, name, BQM):
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

    @parameterized.expand(BQMs.items())
    def test_vartype_only(self, name, BQM):
        bqm = BQM('SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (0, 0))
        assert_consistent_bqm(bqm)

        bqm = BQM(vartype='SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (0, 0))
        assert_consistent_bqm(bqm)

        bqm = BQM('BINARY')
        self.assertEqual(bqm.vartype, dimod.BINARY)
        self.assertEqual(bqm.shape, (0, 0))
        assert_consistent_bqm(bqm)

        bqm = BQM(vartype='BINARY')
        self.assertEqual(bqm.vartype, dimod.BINARY)
        self.assertEqual(bqm.shape, (0, 0))
        assert_consistent_bqm(bqm)

    @parameterized.expand(BQMs.items())
    def test_vartype_readonly(self, name, BQM):
        bqm = BQM('SPIN')
        with self.assertRaises(AttributeError):
            bqm.vartype = dimod.BINARY


class TestContractVariables(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_binary(self, name, BQM):
        bqm = BQM({'a': 2, 'b': -8}, {('a', 'b'): -2, ('b', 'c'): 1}, 1.2,
                  dimod.BINARY)

        bqm.contract_variables('a', 'b')

        assert_consistent_bqm(bqm)

        target = BQM({'a': -8}, {'ac': 1}, 1.2, dimod.BINARY)

        self.assertEqual(bqm, target)

    @parameterized.expand(BQMs.items())
    def test_spin(self, name, BQM):
        bqm = BQM({'a': 2, 'b': -8}, {('a', 'b'): -2, ('b', 'c'): 1}, 1.2,
                  dimod.SPIN)

        bqm.contract_variables('a', 'b')

        assert_consistent_bqm(bqm)

        target = BQM({'a': -6}, {'ac': 1}, -.8, dimod.SPIN)

        assert_bqm_almost_equal(bqm, target, places=5)

    @parameterized.expand(BQMs.items())
    def test_no_interaction(self, name, BQM):
        bqm = BQM({'a': 2, 'b': -8}, {('a', 'b'): -2, ('b', 'c'): 1}, 1.2,
                  dimod.BINARY)

        bqm.contract_variables('a', 'c')

        assert_consistent_bqm(bqm)

        target = BQM({'a': 2, 'b': -8}, {('a', 'b'): -1}, 1.2, dimod.BINARY)

        self.assertEqual(bqm, target)

class TestCoo(unittest.TestCase):
    @parameterized.expand(BQM_CLSs.items())
    def test_to_coo_string_empty_BINARY(self, name, BQM):
        bqm = BQM.empty(dimod.BINARY)

        with self.assertWarns(DeprecationWarning):
            bqm_str = bqm.to_coo()

        self.assertIsInstance(bqm_str, str)

        self.assertEqual(bqm_str, '')

    @parameterized.expand(BQM_CLSs.items())
    def test_to_coo_string_empty_SPIN(self, name, BQM):
        bqm = BQM.empty(dimod.SPIN)

        with self.assertWarns(DeprecationWarning):
            bqm_str = bqm.to_coo()

        self.assertIsInstance(bqm_str, str)

        self.assertEqual(bqm_str, '')

    @parameterized.expand(BQM_CLSs.items())
    def test_to_coo_string_typical_SPIN(self, name, BQM):
        bqm = BQM.from_ising({0: 1.}, {(0, 1): 2, (2, 3): .4})
        with self.assertWarns(DeprecationWarning):
            s = bqm.to_coo()
        contents = "0 0 1.000000\n0 1 2.000000\n2 3 0.400000"
        self.assertEqual(s, contents)

    @parameterized.expand(BQM_CLSs.items())
    def test_to_coo_string_typical_BINARY(self, name, BQM):
        bqm = BQM.from_qubo({(0, 0): 1, (0, 1): 2, (2, 3): .4})
        with self.assertWarns(DeprecationWarning):
            s = bqm.to_coo()
        contents = "0 0 1.000000\n0 1 2.000000\n2 3 0.400000"
        self.assertEqual(s, contents)

    @parameterized.expand(BQM_CLSs.items())
    def test_from_coo_file(self, name, BQM):
        import os.path as path

        filepath = path.join(path.dirname(path.abspath(__file__)), 'data', 'coo_qubo.qubo')

        with open(filepath, 'r') as fp:
            with self.assertWarns(DeprecationWarning):
                bqm = BQM.from_coo(fp, dimod.BINARY)

        self.assertEqual(bqm, BQM.from_qubo({(0, 0): -1, (1, 1): -1, (2, 2): -1, (3, 3): -1}))

    def test_from_coo_string(self):
        contents = "0 0 1.000000\n0 1 2.000000\n2 3 0.400000"
        with self.assertWarns(DeprecationWarning):
            bqm = BinaryQuadraticModel.from_coo(contents, dimod.SPIN)
        self.assertEqual(bqm, BinaryQuadraticModel.from_ising({0: 1.}, {(0, 1): 2, (2, 3): .4}))

    @parameterized.expand(BQM_CLSs.items())
    def test_coo_functional_file_empty_BINARY(self, name, BQM):
        bqm = BQM.empty(dimod.BINARY)

        tmpdir = tempfile.mkdtemp()
        filename = path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            with self.assertWarns(DeprecationWarning):
                bqm.to_coo(file)

        with open(filename, 'r') as file:
            with self.assertWarns(DeprecationWarning):
                new_bqm = BQM.from_coo(file, dimod.BINARY)

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    @parameterized.expand(BQM_CLSs.items())
    def test_coo_functional_file_empty_SPIN(self, name, BQM):
        bqm = BQM.empty(dimod.SPIN)

        tmpdir = tempfile.mkdtemp()
        filename = path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            with self.assertWarns(DeprecationWarning):
                bqm.to_coo(file)

        with open(filename, 'r') as file:
            with self.assertWarns(DeprecationWarning):
                new_bqm = BQM.from_coo(file, dimod.SPIN)

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    def test_coo_functional_file_BINARY(self):
        bqm = BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.BINARY)

        tmpdir = tempfile.mkdtemp()
        filename = path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            with self.assertWarns(DeprecationWarning):
                bqm.to_coo(file)

        with open(filename, 'r') as file:
            with self.assertWarns(DeprecationWarning):
                new_bqm = BinaryQuadraticModel.from_coo(file, dimod.BINARY)

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    def test_coo_functional_file_SPIN(self):
        bqm = BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.SPIN)

        tmpdir = tempfile.mkdtemp()
        filename = path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            with self.assertWarns(DeprecationWarning):
                bqm.to_coo(file)

        with open(filename, 'r') as file:
            with self.assertWarns(DeprecationWarning):
                new_bqm = BinaryQuadraticModel.from_coo(file, dimod.SPIN)

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    @parameterized.expand(BQM_CLSs.items())
    def test_coo_functional_string_empty_BINARY(self, name, BQM):
        bqm = BQM.empty(dimod.BINARY)

        with self.assertWarns(DeprecationWarning):
            s = bqm.to_coo()
        with self.assertWarns(DeprecationWarning):
            new_bqm = BQM.from_coo(s, dimod.BINARY)

        self.assertEqual(bqm, new_bqm)

    @parameterized.expand(BQM_CLSs.items())
    def test_coo_functional_string_empty_SPIN(self, name, BQM):
        bqm = BQM.empty(dimod.SPIN)

        with self.assertWarns(DeprecationWarning):
            s = bqm.to_coo()
        with self.assertWarns(DeprecationWarning):
            new_bqm = BQM.from_coo(s, dimod.SPIN)

        self.assertEqual(bqm, new_bqm)

    def test_coo_functional_string_BINARY(self):
        bqm = BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.BINARY)

        with self.assertWarns(DeprecationWarning):
            s = bqm.to_coo()
        with self.assertWarns(DeprecationWarning):
            new_bqm = BinaryQuadraticModel.from_coo(s, dimod.BINARY)

        self.assertEqual(bqm, new_bqm)

    @parameterized.expand(BQM_CLSs.items())
    def test_coo_functional_two_digit_integers_string(self, name, BQM):
        bqm = BQM.from_ising({12: .5, 0: 1}, {(0, 12): .5})

        with self.assertWarns(DeprecationWarning):
            s = bqm.to_coo()
        with self.assertWarns(DeprecationWarning):
            new_bqm = BQM.from_coo(s, dimod.SPIN)

        self.assertEqual(bqm, new_bqm)

    def test_coo_functional_string_SPIN(self):
        bqm = BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.SPIN)

        with self.assertWarns(DeprecationWarning):
            s = bqm.to_coo()
        with self.assertWarns(DeprecationWarning):
            new_bqm = BinaryQuadraticModel.from_coo(s, dimod.SPIN)

        self.assertEqual(bqm, new_bqm)


class TestDegree(unittest.TestCase):
    @parameterized.expand(BQM_CLSs.items())
    def test_degrees(self, name, BQM):
        bqm = BQM({}, {'ab': 1, 'bc': 1, 'ac': 1, 'ad': 1}, 'SPIN')
        self.assertEqual(bqm.degrees(), {'a': 3, 'b': 2, 'c': 2, 'd': 1})

    @parameterized.expand(BQM_CLSs.items())
    def test_degrees_array(self, name, BQM):
        bqm = BQM('SPIN')
        bqm.add_linear_from((v, 0) for v in 'abcd')
        bqm.add_quadratic_from({'ab': 1, 'bc': 1, 'ac': 1, 'ad': 1})
        np.testing.assert_array_equal(bqm.degrees(array=True), [3, 2, 2, 1])


class TestDeprecation(unittest.TestCase):
    @parameterized.expand(BQM_CLSs.items())
    def test_shapeable(self, name, BQM):
        with self.assertWarns(DeprecationWarning):
            self.assertTrue(BQM.shapeable())

    @parameterized.expand(BQMs.items())
    def test_iter_neighbors(self, name, BQM):
        pass

    @parameterized.expand(BQMs.items())
    def test_has_variable(self, name, BQM):
        h = OrderedDict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = BQM(h, J, dimod.SPIN)

        with self.assertWarns(DeprecationWarning):
            self.assertTrue(bqm.has_variable('a'))
            self.assertTrue(bqm.has_variable(1))
            self.assertTrue(bqm.has_variable(3))

            # no false positives
            self.assertFalse(bqm.has_variable(0))
            self.assertFalse(bqm.has_variable(2))


class TestDictBQM(unittest.TestCase):
    def test_numeric_required_args(self):
        bqm = DictBQM('SPIN')

        class N(float):
            def __init__(self, a):  # required argument
                pass

        bqm.add_linear('a', N(1))
        bqm.add_quadratic('a', 'b', N(2))

        self.assertEqual(bqm.linear, {'a': 1, 'b': 0})
        self.assertEqual(bqm.quadratic, {('a', 'b'): 2})


class TestCopy(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_copy(self, name, BQM):
        bqm = BQM({'a': -1, 'b': 1}, {'ab': 2}, 6, dimod.BINARY)
        new = bqm.copy()
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)

        # modify the original and make sure it doesn't propogate
        new.set_linear('a', 1)
        self.assertEqual(new.linear['a'], 1)

    @parameterized.expand(BQMs.items())
    def test_standardlib_copy(self, name, BQM):
        from copy import copy

        bqm = BQM({'a': -1, 'b': 1}, {'ab': 2}, 6, dimod.BINARY)
        new = copy(bqm)
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)

    @parameterized.expand(BQMs.items())
    def test_standardlib_deepcopy(self, name, BQM):
        from copy import deepcopy

        bqm = BQM({'a': -1, 'b': 1}, {'ab': 2}, 6, dimod.BINARY)
        new = deepcopy(bqm)
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)

    @parameterized.expand(BQMs.items())
    def test_standardlib_deepcopy_multi(self, name, BQM):
        from copy import deepcopy

        bqm = BQM({'a': -1, 'b': 1}, {'ab': 2}, 6, dimod.BINARY)
        copied = deepcopy([bqm, [bqm]])

        new = copied[0]
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)

        self.assertIs(new, copied[1][0])

    @parameterized.expand(BQM_CLSs.items())
    def test_subclass(self, name, BQM):
        # copy should respect subclassing
        class SubBQM(BQM):
            pass

        bqm = BQM({'a': -1, 'b': 1}, {'ab': 2}, 6, dimod.BINARY)
        new = bqm.copy()
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)

    @parameterized.expand(BQM_CLSs.items())
    def test_bug(self, name, BQM):
        bqm = BQM({'a': 1}, {}, 'SPIN')
        bqm.get_linear('a')
        new = bqm.copy()
        new.scale(-1)
        self.assertEqual(new, BQM({'a': -1}, {}, 0, 'SPIN'))


class TestEmpty(unittest.TestCase):
    @parameterized.expand(BQM_CLSs.items())
    def test_binary(self, name, BQM):
        bqm = BQM.empty(dimod.BINARY)
        self.assertIsInstance(bqm, BQM)
        assert_consistent_bqm(bqm)
        self.assertIs(bqm.vartype, dimod.BINARY)
        self.assertEqual(bqm.shape, (0, 0))

    @parameterized.expand(BQM_CLSs.items())
    def test_spin(self, name, BQM):
        bqm = BQM.empty(dimod.SPIN)
        self.assertIsInstance(bqm, BQM)
        self.assertIs(bqm.vartype, dimod.SPIN)
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.shape, (0, 0))


class TestEnergies(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_2path(self, name, BQM):
        bqm = BQM([.1, -.2], [[0, -1], [0, 0]], 'SPIN')
        samples = [[-1, -1],
                   [-1, +1],
                   [+1, -1],
                   [+1, +1]]

        energies = bqm.energies(np.asarray(samples))

        np.testing.assert_array_almost_equal(energies, [-.9, .7, 1.3, -1.1])

    @parameterized.expand(BQMs.items())
    def test_5chain(self, name, BQM):
        arr = np.tril(np.triu(np.ones((5, 5)), 1), 1)
        bqm = BQM(arr, 'BINARY')
        samples = [[0, 0, 0, 0, 0]]

        energies = bqm.energies(np.asarray(samples))
        np.testing.assert_array_almost_equal(energies, [0])

    def test_bug922(self):
        # https://github.com/dwavesystems/dimod/issues/922
        bqm = BinaryQuadraticModel([1], [[0, 1], [0, 0]], 0, 'SPIN', dtype=object)
        bqm.energies([0, 1])

        bqm = BinaryQuadraticModel([1], {}, 0, 'SPIN', dtype=object)
        bqm.energies([1])

        bqm = BinaryQuadraticModel([.1], {}, 0, 'SPIN', dtype=object)
        bqm.energies([1])

        bqm = BinaryQuadraticModel([.1], [[0, 1], [0, 0]], 0, 'SPIN', dtype=object)
        bqm.energies([0, 1])

        bqm = BinaryQuadraticModel([1], [[.0, 1], [0, 0]], 0, 'SPIN', dtype=object)
        bqm.energies([0, 1])

    @parameterized.expand(BQMs.items())
    def test_dtype(self, name, BQM):
        arr = np.arange(9).reshape((3, 3))
        bqm = BQM(arr, dimod.BINARY)

        samples = [[0, 0, 1], [1, 1, 0]]

        energies = bqm.energies(samples, dtype=np.float32)

        self.assertEqual(energies.dtype, np.float32)

    @parameterized.expand(BQMs.items())
    def test_empty(self, name, BQM):
        empty = BQM('BINARY')

        self.assertEqual(empty.energy({}), 0)
        self.assertEqual(empty.energy([]), 0)

        np.testing.assert_array_equal(empty.energies([]), [])
        np.testing.assert_array_equal(empty.energies([[], []]), [0, 0])
        np.testing.assert_array_equal(empty.energies([{}, {}]), [0, 0])

    @parameterized.expand(BQMs.items())
    def test_energy(self, name, BQM):
        arr = np.triu(np.ones((5, 5)))
        bqm = BQM(arr, 'BINARY')
        samples = [[0, 0, 1, 0, 0]]

        energy = bqm.energy(np.asarray(samples))
        self.assertEqual(energy, 1)

    @parameterized.expand(BQMs.items())
    def test_label_mismatch(self, name, BQM):
        arr = np.arange(9).reshape((3, 3))
        bqm = BQM(arr, dimod.BINARY)

        samples = ([[0, 0, 1], [1, 1, 0]], 'abc')

        with self.assertRaises(ValueError):
            bqm.energies(samples)

    def test_sample_dtype(self):
        x, y = dimod.Binaries('xy')
        bqm = 3*x + y - 5*x*y + 5

        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            with self.subTest(dtype):
                arr = np.array([5, 2], dtype=dtype)
                self.assertEqual(bqm.energy((arr, 'xy')), -28)

        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            with self.subTest(dtype):
                arr = np.array([5, 2], dtype=dtype)
                self.assertEqual(bqm.energy((arr, 'xy')), -28)

        for dtype in [np.float32, np.float64]:
            with self.subTest(dtype):
                arr = np.array([5, 2.5], dtype=dtype)
                self.assertEqual(bqm.energy((arr, 'xy')), -40)

        for dtype in [complex]:
            with self.subTest(dtype):
                arr = np.array([5, 2], dtype=dtype)
                with self.assertRaises(ValueError):
                    bqm.energy((arr, 'xy'))

    @parameterized.expand(BQMs.items())
    def test_superset(self, name, BQM):
        bqm = BQM({'a': 1}, {'ab': 1}, 1.5, 'BINARY')

        self.assertEqual(bqm.energy({'a': 1, 'b': 1, 'c': 1}), 3.5)
        self.assertEqual(bqm.energy({'a': 1, 'b': 0, 'c': 1}), 2.5)

    @parameterized.expand(BQMs.items())
    def test_subset(self, name, BQM):
        arr = np.arange(9).reshape((3, 3))
        bqm = BQM(arr, dimod.BINARY)

        samples = [0, 0]

        with self.assertRaises(ValueError):
            bqm.energies(samples)

    @parameterized.expand(BQMs.items())
    def test_subset_empty(self, name, BQM):
        arr = np.arange(9).reshape((3, 3))
        bqm = BQM(arr, dimod.BINARY)

        with self.assertRaises(ValueError):
            bqm.energies([])

    @parameterized.expand(BQMs.items())
    def test_samples_like(self, name, BQM):
        bqm = BQM({'a': 1}, {'ab': 2}, 3, 'BINARY')

        samples = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
        labels = 'ab'

        energies = [3, 3, 4, 6]

        with self.subTest('tuple'):
            np.testing.assert_array_equal(bqm.energies((samples, labels)), energies)

        with self.subTest('dicts'):
            np.testing.assert_array_equal(
                bqm.energies([dict(zip(labels, row)) for row in samples]),
                energies)

        with self.subTest('sample set'):
            np.testing.assert_array_equal(
                bqm.energies(dimod.SampleSet.from_samples_bqm((samples, labels), bqm)),
                energies)


class TestMaximumDeltaEnergy(unittest.TestCase):
    @parameterized.expand(itertools.product(BQMs.values(), ('SPIN', 'BINARY')))
    def test_empty(self, BQM, vartype):
        bqm = BQM(vartype)
        self.assertEqual(bqm.maximum_energy_delta(), 0)

    @parameterized.expand(itertools.product(BQMs.values(), (('SPIN', 46), ('BINARY', 23))))
    def test_nonempty(self, BQM, vartype_and_expected_value):
        vartype, expected_value = vartype_and_expected_value
        bqm = BQM((1, 3, 7), {(1, 0): 2, (2, 0): 5, (2, 1): 11}, vartype)
        self.assertEqual(bqm.maximum_energy_delta(), expected_value)


class TestFileView(unittest.TestCase):
    @parameterized.expand(BQM_CLSs.items())
    def test_empty(self, name, BQM):
        bqm = BQM('SPIN')

        with tempfile.TemporaryFile() as tf:
            with bqm.to_file() as bqmf:
                shutil.copyfileobj(bqmf, tf)
            tf.seek(0)
            new = BQM.from_file(tf)

        self.assertEqual(bqm, new)

    @parameterized.expand(BQM_CLSs.items())
    def test_2path(self, name, BQM):
        bqm = BQM([.1, -.2], [[0, -1], [0, 0]], 'SPIN')

        with tempfile.TemporaryFile() as tf:
            with bqm.to_file() as bqmf:
                shutil.copyfileobj(bqmf, tf)
            tf.seek(0)
            new = BQM.from_file(tf)

        self.assertEqual(bqm, new)


class TestFixVariable(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_spin(self, name, BQM):
        bqm = BQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        bqm.fix_variable('a', +1)
        self.assertEqual(bqm, BQM({'b': -1}, {}, 1.5, dimod.SPIN))

        bqm = BQM({'a': .5}, {('a', 'b'): -1}, 1.5, dimod.SPIN)
        bqm.fix_variable('a', -1)
        self.assertEqual(bqm, BQM({'b': +1}, {}, 1, dimod.SPIN))

    @parameterized.expand(BQMs.items())
    def test_binary(self, name, BQM):
        bqm = BQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        bqm.fix_variable('a', 1)
        self.assertEqual(bqm, BQM({'b': -1}, {}, 1.5, dimod.BINARY))

        bqm = BQM({'a': .5}, {('a', 'b'): -1}, 1.5, dimod.BINARY)
        bqm.fix_variable('a', 0)
        self.assertEqual(bqm, BQM({'b': 0}, {}, 1.5, dimod.BINARY))

    @parameterized.expand(BQMs.items())
    def test_missing_variable(self, name, BQM):
        with self.assertRaises(ValueError):
            BQM('SPIN').fix_variable('a', -1)

    @parameterized.expand(BQMs.items())
    def test_bug(self, name, BQM):
        bqm = BQM({1: 4.0, 2: -4.0, 3: 0.0, 4: 1.0, 5: -1.0},
                  {(1, 0): -4.0, (3, 2): 4.0, (5, 4): -2.0}, 0.0, 'BINARY')
        fixed = {2: 0, 3: 0, 4: 0, 5: 0}

        bqm.fix_variables(fixed)  # should not raise an error


class TestFixVariables(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_typical(self, name, BQM):
        bqm = BQM({'a': -1, 'b': 1, 'c': 3}, {}, dimod.SPIN)

        bqm.fix_variables({'a': 1, 'b': -1})

        self.assertEqual(bqm.linear, {'c': 3})
        self.assertEqual(bqm.quadratic, {})
        self.assertEqual(bqm.offset, -2)


class TestFlipVariable(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_binary(self, name, BQM):
        bqm = BQM({'a': -1, 'b': 1}, {'ab': -1}, 0, dimod.BINARY)
        bqm.flip_variable('a')
        self.assertEqual(bqm, BQM({'a': 1}, {'ab': 1}, -1.0, dimod.BINARY))

    @parameterized.expand(BQMs.items())
    def test_spin(self, name, BQM):
        bqm = BQM({'a': -1, 'b': 1}, {'ab': -1}, 1.0, dimod.SPIN)
        bqm.flip_variable('a')
        self.assertEqual(bqm, BQM({'a': 1, 'b': 1}, {'ab': 1}, 1.0, dimod.SPIN))


class TestFromNumpyVectors(unittest.TestCase):
    @parameterized.expand(BQM_CLSs.items())
    def test_3var(self, _, BQM):
        h = np.array([-1, 1, 5])
        heads = np.array([0, 1])
        tails = np.array([1, 2])
        values = np.array([-1, +1])

        bqm = BQM.from_numpy_vectors(h, (heads, tails, values), 0.5, 'SPIN')

        self.assertIs(type(bqm), BQM)
        self.assertEqual(bqm.linear, {0: -1, 1: 1, 2: 5})
        self.assertEqual(bqm.adj, {0: {1: -1}, 1: {0: -1, 2: 1}, 2: {1: 1}})
        self.assertEqual(bqm.offset, 0.5)
        self.assertIs(bqm.vartype, dimod.SPIN)

    @parameterized.expand(BQM_CLSs.items())
    def test_3var_duplicate(self, _, BQM):
        h = np.array([-1, 1, 5])
        heads = np.array([0, 1, 0, 1])
        tails = np.array([1, 2, 1, 0])
        values = np.array([-1, +1, -2, -3])

        bqm = BQM.from_numpy_vectors(h, (heads, tails, values), 0.5, 'SPIN')

        self.assertIs(type(bqm), BQM)
        self.assertEqual(bqm.linear, {0: -1, 1: 1, 2: 5})
        self.assertEqual(bqm.adj, {0: {1: -6}, 1: {0: -6, 2: 1}, 2: {1: 1}})
        self.assertEqual(bqm.offset, 0.5)
        self.assertIs(bqm.vartype, dimod.SPIN)

    @parameterized.expand(BQM_CLSs.items())
    def test_3var_labels(self, _, BQM):
        h = np.array([-1, 1, 5])
        heads = np.array([0, 1])
        tails = np.array([1, 2])
        values = np.array([-1, +1])

        bqm = BQM.from_numpy_vectors(h, (heads, tails, values), 0.5, 'SPIN',
                                     variable_order=['a', 'b', 'c'])

        self.assertEqual(bqm,
                         BQM.from_ising({'a': -1, 'b': 1, 'c': 5},
                                        {('a', 'b'): -1, ('b', 'c'): 1},
                                        .5))
        self.assertEqual(list(bqm.variables), ['a', 'b', 'c'])

    @parameterized.expand(BQM_CLSs.items())
    def test_5var_labels(self, _, BQM):
        bqm = BQM.from_numpy_vectors(range(5), ([], [], []), .5, 'SPIN',
                                     variable_order='abcde')
        self.assertEqual(list(bqm.variables), list('abcde'))

    @parameterized.expand(BQM_CLSs.items())
    def test_dtypes(self, _, BQM):
        # we don't test uint64 because when combined with int it gets promoted
        # to float
        integral = [np.uint8, np.uint16, np.uint32,
                    np.int8, np.int16, np.int32, np.int64]
        numeric = [np.int8, np.int16, np.int32, np.int64,
                   np.float32, np.float64]

        h = [1, 2, 3]
        heads = [0, 1]
        tails = [1, 2]
        values = [4, 5]

        for types in itertools.product(numeric, integral, integral, numeric):
            with self.subTest(' '.join(map(str, types))):
                bqm = BQM.from_numpy_vectors(
                    np.asarray(h, dtype=types[0]),
                    (np.asarray(heads, dtype=types[1]),
                     np.asarray(tails, dtype=types[2]),
                     np.asarray(values, dtype=types[3])),
                    0.0, 'SPIN')

                self.assertEqual(bqm.linear, {0: 1, 1: 2, 2: 3})
                self.assertEqual(
                    bqm.adj, {0: {1: 4}, 1: {0: 4, 2: 5}, 2: {1: 5}})

    @parameterized.expand(BQM_CLSs.items())
    def test_empty(self, _, BQM):
        bqm = BQM.from_numpy_vectors([], ([], [], []), 1.5, 'SPIN')
        self.assertEqual(bqm.shape, (0, 0))
        self.assertEqual(bqm.offset, 1.5)

    @parameterized.expand(BQM_CLSs.items())
    def test_linear_in_quadratic(self, _, BQM):
        h = np.array([-1, 1, 5])
        heads = np.array([0, 1])
        tails = np.array([0, 2])
        values = np.array([-1, +1])
        spin = BQM.from_numpy_vectors(h, (heads, tails, values), 0.5, 'SPIN')
        binary = BQM.from_numpy_vectors(h, (heads, tails, values), 0.5, 'BINARY')

        self.assertEqual(spin.adj, binary.adj)

        self.assertEqual(spin.linear, {0: -1, 1: 1, 2: 5})
        self.assertEqual(binary.linear, {0: -2, 1: 1, 2: 5})
        self.assertEqual(spin.offset, -.5)
        self.assertEqual(binary.offset, .5)

    @parameterized.expand(BQM_CLSs.items())
    def test_noncontiguous(self, _, BQM):
        quadratic = np.asarray([[0, 1], [1, 2]])

        bqm = BQM.from_numpy_vectors(
            [], (quadratic[:, 0], quadratic[:, 1], [.5, .6]), 1.5, 'SPIN')

    @parameterized.expand(BQM_CLSs.items())
    def test_oversized_linear(self, _, BQM):
        bqm = BQM.from_numpy_vectors([0, 1, 2], ([], [], []), 1.5, 'SPIN')
        self.assertEqual(bqm.shape, (3, 0))
        self.assertEqual(bqm.linear, {0: 0, 1: 1, 2: 2})
        self.assertEqual(bqm.offset, 1.5)

    @parameterized.expand(BQM_CLSs.items())
    def test_undersized_linear(self, _, BQM):
        bqm = BQM.from_numpy_vectors([0, 1], ([3], [4], [1]), 1.5, 'SPIN')
        self.assertEqual(bqm.shape, (5, 1))
        self.assertEqual(bqm.linear, {0: 0, 1: 1, 2: 0, 3: 0, 4: 0})
        self.assertEqual(bqm.offset, 1.5)
        self.assertEqual(bqm.adj, {0: {}, 1: {}, 2: {}, 3: {4: 1}, 4: {3: 1}})


class TestFromQUBO(unittest.TestCase):
    @parameterized.expand(BQM_CLSs.items())
    def test_basic(self, name, BQM):
        Q = {(0, 0): -1, (0, 1): -1, (0, 2): -1, (1, 2): 1}
        bqm = BQM.from_qubo(Q)

        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.linear, {0: -1, 1: 0, 2: 0})
        self.assertEqual(bqm.adj, {0: {1: -1, 2: -1},
                                   1: {0: -1, 2: 1},
                                   2: {0: -1, 1: 1}})
        self.assertEqual(bqm.offset, 0)

    @parameterized.expand(BQM_CLSs.items())
    def test_with_offset(self, name, BQM):
        Q = {(0, 0): -1, (0, 1): -1, (0, 2): -1, (1, 2): 1}
        bqm = BQM.from_qubo(Q, 1.6)

        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.linear, {0: -1, 1: 0, 2: 0})
        self.assertEqual(bqm.adj, {0: {1: -1, 2: -1},
                                   1: {0: -1, 2: 1},
                                   2: {0: -1, 1: 1}})
        self.assertAlmostEqual(bqm.offset, 1.6)


class TestGetLinear(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_disconnected_string_labels(self, name, BQM):
        bqm = BQM({'a': -1, 'b': 1}, {}, dimod.BINARY)
        self.assertEqual(bqm.get_linear('a'), -1)
        self.assertEqual(bqm.get_linear('b'), 1)
        with self.assertRaises(ValueError):
            bqm.get_linear('c')

    @parameterized.expand(BQMs.items())
    def test_disconnected(self, name, BQM):
        bqm = BQM(5, dimod.SPIN)

        for v in range(5):
            self.assertEqual(bqm.get_linear(v), 0)

        with self.assertRaises(ValueError):
            bqm.get_linear(-1)

        with self.assertRaises(ValueError):
            bqm.get_linear(5)

    @parameterized.expand(BQMs.items())
    def test_dtype(self, name, BQM):
        bqm = BQM(5, dimod.SPIN)

        # np.object_ does not play very nicely, even if it's accurate
        dtype = object if bqm.dtype.type is np.object_ else bqm.dtype.type

        for v in range(5):
            self.assertIsInstance(bqm.get_linear(v), dtype)


class TestGetQuadratic(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_3x3array(self, name, BQM):
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

    @parameterized.expand(BQMs.items())
    def test_default(self, name, BQM):
        bqm = BQM(5, 'SPIN')  # has no interactions
        with self.assertRaises(ValueError):
            bqm.get_quadratic(0, 1)
        self.assertEqual(bqm.get_quadratic(0, 1, default=5), 5)

    @parameterized.expand(BQMs.items())
    def test_dtype(self, name, BQM):
        bqm = BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]], dimod.SPIN)

        # np.object_ does not play very nicely, even if it's accurate
        dtype = object if bqm.dtype.type is np.object_ else bqm.dtype.type

        self.assertIsInstance(bqm.get_quadratic(0, 1), dtype)
        self.assertIsInstance(bqm.get_quadratic(1, 0), dtype)

        self.assertIsInstance(bqm.get_quadratic(0, 2), dtype)
        self.assertIsInstance(bqm.get_quadratic(2, 0), dtype)


class TestIsAlmostEqual(unittest.TestCase):
    def test_number(self):
        bqm = BinaryQuadraticModel('SPIN')
        bqm.offset = 1.01
        self.assertTrue(bqm.is_almost_equal(1, places=1))
        self.assertFalse(bqm.is_almost_equal(1, places=2))
        self.assertTrue(bqm.is_almost_equal(1.01, places=2))

    def test_bqm(self):
        bqm = BinaryQuadraticModel({'a': 1.01}, {'ab': 1.01}, 1.01, 'SPIN')

        # different quadratic bias
        other = BinaryQuadraticModel({'a': 1.01}, {'ab': 1}, 1.01, 'SPIN')
        self.assertTrue(bqm.is_almost_equal(other, places=1))
        self.assertFalse(bqm.is_almost_equal(other, places=2))

        # different linear biases
        other = BinaryQuadraticModel({'a': 1.}, {'ab': 1.01}, 1.01, 'SPIN')
        self.assertTrue(bqm.is_almost_equal(other, places=1))
        self.assertFalse(bqm.is_almost_equal(other, places=2))

        # different offset
        other = BinaryQuadraticModel({'a': 1.01}, {'ab': 1.01}, 1, 'SPIN')
        self.assertTrue(bqm.is_almost_equal(other, places=1))
        self.assertFalse(bqm.is_almost_equal(other, places=2))

    def test_qm(self):
        bqm = BinaryQuadraticModel({'a': 1.01}, {'ab': 1.01}, 1.01, 'SPIN')
        qm = dimod.QuadraticModel.from_bqm(bqm)

        self.assertTrue(bqm.is_almost_equal(qm))


class TestEqual(unittest.TestCase):
    def test_qm(self):
        bqm = BinaryQuadraticModel({'a': 1.01}, {'ab': 1.01}, 1.01, 'SPIN')
        qm = dimod.QuadraticModel.from_bqm(bqm)

        self.assertTrue(bqm.is_equal(qm))


class TestIsLinear(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_no_variables(self, name, BQM):
        bqm = BQM('SPIN')
        bqm.offset = 1
        self.assertTrue(bqm.is_linear())

    @parameterized.expand(BQMs.items())
    def test_linear_only(self, name, BQM):
        bqm = BQM({'a': 1, 'b': 2}, {}, 1, 'SPIN')
        self.assertTrue(bqm.is_linear())

    @parameterized.expand(BQMs.items())
    def test_quadratic(self, name, BQM):
        bqm = BQM({'a': 1, 'b': 2}, {'ab': 1}, 1, 'SPIN')
        self.assertFalse(bqm.is_linear())

    @parameterized.expand(BQMs.items())
    def test_three_quadratic(self, name, BQM):
        bqm = BQM({}, {'ab': 1, 'cd': 1}, 0, 'SPIN')
        self.assertFalse(bqm.is_linear())


class TestIteration(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_iter_quadratic_neighbours(self, name, BQM):
        bqm = BQM({'ab': -1, 'bc': 21, 'cd': 1}, dimod.SPIN)
        with self.assertWarns(DeprecationWarning):
            neighbours = set(bqm.iter_quadratic('b'))
        self.assertEqual(neighbours,
                         {('b', 'a', -1), ('b', 'c', 21)})

    @parameterized.expand(BQMs.items())
    def test_iter_quadratic_neighbours_bunch(self, name, BQM):
        bqm = BQM({'bc': 21, 'cd': 1}, dimod.SPIN)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(list(bqm.iter_quadratic(['b', 'c'])),
                             [('b', 'c', 21.0), ('c', 'd', 1.0)])

    @parameterized.expand(BQMs.items())
    def test_iter_variables(self, name, BQM):
        h = OrderedDict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = BQM(h, J, dimod.SPIN)

        self.assertEqual(list(bqm.variables), ['a', 1, 3])


class TestLen(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test__len__(self, name, BQM):
        bqm = BQM(np.ones((107, 107)), dimod.BINARY)
        self.assertEqual(len(bqm), 107)


class TestNBytes(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_small(self, name, BQM):
        bqm = BQM({'a': 1}, {'ab': 1, 'bc': 1}, 1.5, dimod.BINARY)

        if bqm.dtype == object:
            with self.assertRaises(TypeError):
                bqm.nbytes()
            return

        itemsize = bqm.dtype.itemsize

        size = sum([itemsize,  # offset
                    bqm.num_variables*itemsize,  # linear
                    2*bqm.num_interactions*(2*itemsize),  # quadratic
                    ])

        self.assertEqual(bqm.nbytes(), size)
        self.assertEqual(bqm.nbytes(), bqm.nbytes(False))
        self.assertGreaterEqual(bqm.nbytes(True), bqm.nbytes(False))


class TestNetworkxGraph(unittest.TestCase):
    def setUp(self):
        try:
            import networkx as nx
        except ImportError:
            raise unittest.SkipTest("NetworkX is not installed")

    def test_empty(self):
        import networkx as nx
        G = nx.Graph()
        G.vartype = 'SPIN'
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.BinaryQuadraticModel.from_networkx_graph(G)
        self.assertEqual(len(bqm), 0)
        self.assertIs(bqm.vartype, dimod.SPIN)

    def test_no_biases(self):
        import networkx as nx
        G = nx.complete_graph(5)
        G.vartype = 'BINARY'
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.BinaryQuadraticModel.from_networkx_graph(G)

        self.assertIs(bqm.vartype, dimod.BINARY)
        self.assertEqual(set(bqm.variables), set(range(5)))
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.adj[u][v], 0)
            self.assertEqual(bqm.linear[v], 0)
        self.assertEqual(len(bqm.quadratic), len(G.edges))

    def test_functional(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': .5},
                                                    {'bc': 1, 'cd': -4},
                                                    offset=6)
        with self.assertWarns(DeprecationWarning):
            new = dimod.BinaryQuadraticModel.from_networkx_graph(bqm.to_networkx_graph())
        self.assertEqual(bqm, new)

    def test_to_networkx_graph(self):
        import networkx as nx
        graph = nx.barbell_graph(7, 6)

        # build a BQM
        model = dimod.BinaryQuadraticModel({v: -.1 for v in graph},
                                           {edge: -.4 for edge in graph.edges},
                                           1.3,
                                           vartype=dimod.SPIN)

        # get the graph
        with self.assertWarns(DeprecationWarning):
            BQM = model.to_networkx_graph()

        self.assertEqual(set(graph), set(BQM))
        for u, v in graph.edges:
            self.assertIn(u, BQM[v])

        for v, bias in model.linear.items():
            self.assertEqual(bias, BQM.nodes[v]['bias'])


class TestNumpyMatrix(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_to_numpy_matrix(self, name, BQM):
        # integer-indexed, binary bqm
        linear = {v: v * .01 for v in range(10)}
        quadratic = {(v, u): u * v * .01 for u, v in itertools.combinations(linear, 2)}
        quadratic[(0, 1)] = quadratic[(1, 0)]
        del quadratic[(1, 0)]
        offset = 1.2
        vartype = dimod.BINARY
        bqm = BQM(linear, quadratic, offset, vartype)

        with self.assertWarns(DeprecationWarning):
            M = bqm.to_numpy_matrix()

        self.assertTrue(np.array_equal(M, np.triu(M)))  # upper triangular

        for (row, col), bias in np.ndenumerate(M):
            if row == col:
                self.assertAlmostEqual(bias, linear[row])
            else:
                self.assertTrue((row, col) in quadratic or (col, row) in quadratic)
                self.assertFalse((row, col) in quadratic and (col, row) in quadratic)

                if row > col:
                    self.assertEqual(bias, 0)
                else:
                    if (row, col) in quadratic:
                        self.assertAlmostEqual(quadratic[(row, col)], bias)
                    else:
                        self.assertAlmostEqual(quadratic[(col, row)], bias)

        #

        # integer-indexed, not contiguous
        bqm = BQM({}, {(0, 3): -1}, 0.0, dimod.BINARY)

        with self.assertRaises(ValueError):
            with self.assertWarns(DeprecationWarning):
                M = bqm.to_numpy_matrix()

        #

        # string-labeled, variable_order provided
        linear = {'a': -1}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3}
        bqm = BQM(linear, quadratic, 0.0, dimod.BINARY)

        with self.assertRaises(ValueError):
            with self.assertWarns(DeprecationWarning):
                bqm.to_numpy_matrix(['a', 'c'])  # incomplete variable order

        with self.assertWarns(DeprecationWarning):
            M = bqm.to_numpy_matrix(['a', 'c', 'b'])

        target = [[-1., 1.2, 0.], [0., 0., 0.3], [0., 0., 0.]]
        np.testing.assert_array_almost_equal(M, target)

    @parameterized.expand(BQM_CLSs.items())
    def test_functional(self, name, BQM):
        bqm = BQM({'a': -1}, {'ac': 1.2, 'bc': .3}, dimod.BINARY)

        order = ['a', 'b', 'c']

        with self.assertWarns(DeprecationWarning):
            M = bqm.to_numpy_matrix(variable_order=order)

        with self.assertWarns(DeprecationWarning):
            new = BQM.from_numpy_matrix(M, variable_order=order)

        assert_consistent_bqm(new)
        self.assertEqual(bqm, new)

    @parameterized.expand(BQM_CLSs.items())
    def test_from_numpy_matrix(self, name, BQM):

        linear = {'a': -1}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3}
        bqm = BQM(linear, quadratic, 0.0, dimod.BINARY)

        variable_order = ['a', 'c', 'b']

        with self.assertWarns(DeprecationWarning):
            M = bqm.to_numpy_matrix(variable_order=variable_order)

        with self.assertWarns(DeprecationWarning):
            new_bqm = BQM.from_numpy_matrix(M, variable_order=variable_order)

        self.assertEqual(bqm, new_bqm)

        # zero-interactions get ignored unless provided in interactions
        linear = {'a': -1}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3, ('a', 'b'): 0}
        bqm = BQM(linear, quadratic, 0.0, dimod.BINARY)
        variable_order = ['a', 'c', 'b']
        with self.assertWarns(DeprecationWarning):
            M = bqm.to_numpy_matrix(variable_order=variable_order)

        with self.assertWarns(DeprecationWarning):
            new_bqm = BQM.from_numpy_matrix(M, variable_order=variable_order)

        self.assertNotIn(('a', 'b'), new_bqm.quadratic)
        self.assertNotIn(('b', 'a'), new_bqm.quadratic)

        with self.assertWarns(DeprecationWarning):
            new_bqm = BQM.from_numpy_matrix(M, variable_order=variable_order,
                                            interactions=quadratic)

        self.assertEqual(bqm, new_bqm)

        #

        M = np.asarray([[0, 1], [0, 0]])
        with self.assertWarns(DeprecationWarning):
            bqm = BQM.from_numpy_matrix(M)
        self.assertEqual(bqm, BQM({0: 0, 1: 0}, {(0, 1): 1}, 0, dimod.BINARY))


class TestNormalize(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_normalize(self, name, BQM):
        bqm = BQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.normalize(.5)
        self.assertAlmostEqual(bqm.linear, {0: -.5, 1: .5})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -.25})
        self.assertAlmostEqual(bqm.offset, .25)
        assert_consistent_bqm(bqm)

    @parameterized.expand(BQMs.items())
    def test_exclusions(self, name, BQM):
        bqm = BQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.normalize(.5, ignored_variables=[0])
        self.assertAlmostEqual(bqm.linear, {0: -2, 1: .5})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -.25})
        self.assertAlmostEqual(bqm.offset, .25)
        assert_consistent_bqm(bqm)

        bqm = BQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.normalize(.5, ignored_interactions=[(1, 0)])
        self.assertAlmostEqual(bqm.linear, {0: -.5, 1: .5})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -1})
        self.assertAlmostEqual(bqm.offset, .25)
        assert_consistent_bqm(bqm)

        bqm = BQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.normalize(.5, ignore_offset=True)
        self.assertAlmostEqual(bqm.linear, {0: -.5, 1: .5})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -.25})
        self.assertAlmostEqual(bqm.offset, 1.)
        assert_consistent_bqm(bqm)

        bqm = BQM({0: -2, 1: 2}, {(0, 1): -5}, 1., dimod.SPIN)
        bqm.normalize(0.5, ignored_interactions=[(0, 1)])
        self.assertAlmostEqual(bqm.linear, {0: -.5, 1: .5})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -5})
        self.assertAlmostEqual(bqm.offset, 0.25)
        assert_consistent_bqm(bqm)

    @parameterized.expand(BQMs.items())
    def test_return_value(self, name, BQM):
        bqm = BQM({0: 2}, {(0, 1): 2}, 'SPIN')

        self.assertEqual(bqm.normalize([-1, 1]), .5)


class TestObjectDtype(unittest.TestCase):
    def test_dtypes_array_like_ints(self):
        obj = [[0, 1], [1, 2]]

        bqm = DictBQM(obj, 'BINARY')

        for _, bias in bqm.quadratic.items():
            self.assertIsInstance(bias, int)

    def test_dtypes_ndarray_ints(self):
        obj = np.asarray([[0, 1], [1, 2]], dtype=np.int32)

        bqm = DictBQM(obj, 'BINARY')

        for _, bias in bqm.quadratic.items():
            self.assertIsInstance(bias, np.int32)

    def test_fractions(self):
        from fractions import Fraction

        bqm = DictBQM({'a': Fraction(1, 3)}, {'ab': Fraction(2, 7)},
                      Fraction(5), 'SPIN')

        self.assertIsInstance(bqm.offset, Fraction)
        self.assertIsInstance(bqm.get_linear('a'), Fraction)
        self.assertIsInstance(bqm.get_quadratic('a', 'b'), Fraction)

    def test_string(self):
        bqm = DictBQM({0: 'a'}, {(0, 1): 'b'}, 'c', 'BINARY')

        self.assertIsInstance(bqm.offset, str)
        self.assertEqual(bqm.offset, 'c')
        self.assertIsInstance(bqm.get_linear(0), str)
        self.assertEqual(bqm.get_linear(0), 'a')
        self.assertIsInstance(bqm.get_quadratic(0, 1), str)
        self.assertEqual(bqm.get_quadratic(0, 1), 'b')

        bqm.add_linear(0, 't')
        self.assertEqual(bqm.get_linear(0), 'at')

        bqm.add_quadratic(0, 1, 't')
        self.assertEqual(bqm.get_quadratic(0, 1), 'bt')


class TestOffset(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_offset(self, name, BQM):
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


class TestPickle(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_picklable(self, name, BQM):
        import pickle

        bqm = BQM({'a': -1, 'b': 1}, {'ab': 2}, 6, dimod.BINARY)
        new = pickle.loads(pickle.dumps(bqm))
        self.assertIs(type(bqm), type(new))
        self.assertEqual(bqm, new)


class TestReduce(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_reduce_linear(self, name, BQM):
        bqm = BQM('SPIN')
        bqm.add_linear_from((v, v) for v in range(5))

        with self.subTest('min'):
            self.assertEqual(bqm.reduce_linear(min), 0)

        with self.subTest('min'):
            self.assertEqual(bqm.reduce_linear(max), 4)

        with self.subTest('sum'):
            self.assertEqual(bqm.reduce_linear(operator.add), 10)

        with self.subTest('custom'):
            def mymin(a, b):
                return min(a, b)
            self.assertEqual(bqm.reduce_linear(min),
                             bqm.reduce_linear(mymin))

    @parameterized.expand(BQMs.items())
    def test_reduce_neighborhood(self, name, BQM):
        bqm = BQM('SPIN')
        bqm.add_quadratic_from({'ab': 1, 'bc': 2, 'ac': 4})

        with self.subTest('min'):
            self.assertEqual(bqm.reduce_neighborhood('b', min), 1)

        with self.subTest('min'):
            self.assertEqual(bqm.reduce_neighborhood('b', max), 2)

        with self.subTest('sum'):
            self.assertEqual(bqm.reduce_neighborhood('b', operator.add), 3)

        with self.subTest('custom'):
            def mymin(a, b):
                return min(a, b)
            self.assertEqual(bqm.reduce_neighborhood('b', min),
                             bqm.reduce_neighborhood('b', mymin))

    @parameterized.expand(BQMs.items())
    def test_reduce_quadratic(self, name, BQM):
        bqm = BQM('SPIN')
        bqm.add_quadratic_from({'ab': 1, 'bc': 2, 'ac': 4})

        with self.subTest('min'):
            self.assertEqual(bqm.reduce_quadratic(min), 1)

        with self.subTest('min'):
            self.assertEqual(bqm.reduce_quadratic(max), 4)

        with self.subTest('sum'):
            self.assertEqual(bqm.reduce_quadratic(operator.add), 7)

        with self.subTest('custom'):
            def mymin(a, b):
                return min(a, b)
            self.assertEqual(bqm.reduce_quadratic(min),
                             bqm.reduce_quadratic(mymin))


class TestRemoveInteraction(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_basic(self, name, BQM):
        bqm = BQM(np.triu(np.ones((3, 3))), dimod.BINARY)
        bqm.remove_interaction(0, 1)
        with self.assertRaises(ValueError):
            bqm.remove_interaction(0, 1)
        self.assertEqual(bqm.shape, (3, 2))

        with self.assertRaises(ValueError):
            bqm.remove_interaction('a', 1)  # 'a' is not a variable

        with self.assertRaises(ValueError):
            bqm.remove_interaction(1, 1)

    @parameterized.expand(BQMs.items())
    def test_energy(self, name, BQM):
        bqm = BQM({'a': 1, 'b': 2, 'c': 3}, {'ab': 4, 'bc': 5}, 6, 'BINARY')
        en = bqm.energy({'a': 1, 'b': 1, 'c': 1})
        bqm.remove_interaction('a', 'b')
        self.assertEqual(bqm.energy({'a': 1, 'b': 1, 'c': 1}), en - 4)


class TestRemoveInteractionsFrom(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_basic(self, name, BQM):
        bqm = BQM(np.triu(np.ones((3, 3))), dimod.BINARY)

        bqm.remove_interactions_from([(0, 2), (2, 1)])

        self.assertEqual(bqm.num_interactions, 1)


class TestRemoveVariable(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_energy(self, name, BQM):
        bqm = BQM({'a': 1, 'b': 2, 'c': 3}, {'ab': 4, 'bc': 5}, 6, 'BINARY')
        en = bqm.energy({'a': 1, 'b': 1, 'c': 1})
        bqm.remove_variable('a')
        self.assertEqual(bqm.energy({'b': 1, 'c': 1}), en - 5)

    @parameterized.expand(BQMs.items())
    def test_labelled(self, name, BQM):
        bqm = BQM(dimod.SPIN)
        bqm.add_variable('a')
        bqm.add_variable(1)
        bqm.add_variable(0)
        self.assertEqual(bqm.remove_variable(), 0)
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.remove_variable(), 1)
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.remove_variable(), 'a')
        assert_consistent_bqm(bqm)
        with self.assertRaises(ValueError):
            bqm.remove_variable()

    @parameterized.expand(BQMs.items())
    def test_multiple(self, name, BQM):
        bqm = BQM('SPIN')
        bqm.add_variable('a')
        bqm.add_variable('b')
        bqm.add_variable('c')

        bqm.remove_variables_from('ac')
        self.assertEqual(list(bqm.variables), list('b'))

    @parameterized.expand(BQMs.items())
    def test_provided(self, name, BQM):
        bqm = BQM('SPIN')
        bqm.add_variable('a')
        bqm.add_variable('b')
        bqm.add_variable('c')

        bqm.remove_variable('b')
        assert_consistent_bqm(bqm)

        # maintained order
        self.assertEqual(list(bqm.variables), ['a', 'c'])

        with self.assertRaises(ValueError):
            bqm.remove_variable('b')

    @parameterized.expand(BQMs.items())
    def test_unlabelled(self, name, BQM):
        bqm = BQM(2, dimod.BINARY)
        self.assertEqual(bqm.remove_variable(), 1)
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.remove_variable(), 0)
        assert_consistent_bqm(bqm)
        with self.assertRaises(ValueError):
            bqm.remove_variable()
        assert_consistent_bqm(bqm)


class TestRelabel(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_inplace(self, name, BQM):
        linear = {0: .5, 1: 1.3}
        quadratic = {(0, 1): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        bqm = BQM(linear, quadratic, offset, vartype)

        mapping = {0: 'a', 1: 'b'}
        new = bqm.relabel_variables(mapping)
        assert_consistent_bqm(new)
        self.assertIs(bqm, new)

        # check that new model is correct
        linear = {'a': .5, 'b': 1.3}
        quadratic = {('a', 'b'): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        test = BQM(linear, quadratic, offset, vartype)
        self.assertEqual(bqm, test)

    @parameterized.expand(BQMs.items())
    def test_integer(self, name, BQM):
        bqm = BQM(np.arange(25).reshape((5, 5)), 'SPIN')

        # relabel variables with alphabet letter
        bqm.relabel_variables({0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'})

        # create a copy
        bqm_copy = bqm.copy()

        # this relabel is inplace
        _, inverse = bqm.relabel_variables_as_integers()

        self.assertEqual(set(bqm.variables), set(range(5)))

        # relabel the variables as alphabet letters again
        bqm.relabel_variables(inverse, inplace=True)
        self.assertEqual(bqm, bqm_copy)

        # check the inplace False case
        new, mapping = bqm.relabel_variables_as_integers(inplace=False)
        self.assertEqual(set(new.variables), set(range(5)))

        new.relabel_variables(mapping)
        self.assertEqual(new, bqm)

    @parameterized.expand(BQMs.items())
    def test_not_inplace(self, name, BQM):
        linear = {0: .5, 1: 1.3}
        quadratic = {(0, 1): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        bqm = BQM(linear, quadratic, offset, vartype)

        mapping = {0: 'a', 1: 'b'}
        new = bqm.relabel_variables(mapping, inplace=False)
        assert_consistent_bqm(new)
        self.assertIsNot(bqm, new)

        # check that new model is the same as old model
        linear = {'a': .5, 'b': 1.3}
        quadratic = {('a', 'b'): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        test = BQM(linear, quadratic, offset, vartype)

        self.assertTrue(new.is_almost_equal(test))

    @parameterized.expand(BQMs.items())
    def test_overlap(self, name, BQM):
        linear = {v: .1 * v for v in range(-5, 4)}
        quadratic = {(u, v): .1 * u * v for u, v in itertools.combinations(linear, 2)}
        offset = 1.2
        vartype = dimod.SPIN
        bqm = BQM(linear, quadratic, offset, vartype)

        partial_overlap_mapping = {v: -v for v in linear}  # has variables mapped to other old labels

        # construct a test model by using copy
        test = bqm.relabel_variables(partial_overlap_mapping, inplace=False)

        # now apply in place
        bqm.relabel_variables(partial_overlap_mapping, inplace=True)

        # should have stayed the same
        assert_consistent_bqm(test)
        assert_consistent_bqm(bqm)
        self.assertTrue(test.is_almost_equal(bqm))

    @parameterized.expand(BQMs.items())
    def test_identity(self, name, BQM):
        linear = {v: .1 * v for v in range(-5, 4)}
        quadratic = {(u, v): .1 * u * v for u, v in itertools.combinations(linear, 2)}
        offset = 1.2
        vartype = dimod.SPIN
        bqm = BQM(linear, quadratic, offset, vartype)
        old = bqm.copy()

        identity_mapping = {v: v for v in linear}

        bqm.relabel_variables(identity_mapping, inplace=True)

        # should have stayed the same
        assert_consistent_bqm(old)
        assert_consistent_bqm(bqm)
        self.assertTrue(old.is_almost_equal(bqm))

    @parameterized.expand(BQMs.items())
    def test_partial_relabel_copy(self, name, BQM):
        linear = {v: .1 * v for v in range(-5, 5)}
        quadratic = {(u, v): .1 * u * v for u, v in itertools.combinations(linear, 2)}
        offset = 1.2
        vartype = dimod.SPIN
        bqm = BQM(linear, quadratic, offset, vartype)

        mapping = {0: 'a', 1: 'b'}  # partial mapping
        newmodel = bqm.relabel_variables(mapping, inplace=False)

        newlinear = linear.copy()
        newlinear['a'] = newlinear[0]
        newlinear['b'] = newlinear[1]
        del newlinear[0]
        del newlinear[1]

        self.assertEqual(set(newlinear), set(newmodel.linear))
        for v in newlinear:
            self.assertAlmostEqual(newlinear[v], newmodel.linear[v])

    @parameterized.expand(BQMs.items())
    def test_partial_relabel_inplace(self, name, BQM):
        linear = {v: .1 * v for v in range(-5, 5)}
        quadratic = {(u, v): .1 * u * v for u, v in itertools.combinations(linear, 2)}
        offset = 1.2
        vartype = dimod.SPIN
        bqm = BQM(linear, quadratic, offset, vartype)

        newlinear = linear.copy()
        newlinear['a'] = newlinear[0]
        newlinear['b'] = newlinear[1]
        del newlinear[0]
        del newlinear[1]

        mapping = {0: 'a', 1: 'b'}  # partial mapping
        bqm.relabel_variables(mapping, inplace=True)

        self.assertEqual(set(newlinear), set(bqm.linear))
        for v in newlinear:
            self.assertAlmostEqual(newlinear[v], bqm.linear[v])


class TestResize(unittest.TestCase):
    def test_do_nothing(self):
        bqm = dimod.BQM('BINARY')
        bqm.add_variables_from([('a', 0), ('b', 0), ('c', 0)])
        bqm.set_quadratic('b', 'c', 1)

        self.assertEqual(bqm.resize(3), 0)
        self.assertEqual(bqm.num_variables, 3)
        self.assertEqual(bqm.num_interactions, 1)

    def test_exception(self):
        bqm = dimod.BinaryQuadraticModel('BINARY')

        with self.assertRaises(ValueError):
            bqm.resize(-100)

    def test_grow(self):
        bqm = dimod.BinaryQuadraticModel('BINARY')

        self.assertEqual(bqm.resize(10), 10)
        self.assertEqual(bqm.num_variables, 10)
        self.assertEqual(bqm.num_interactions, 0)
        for v, bias in bqm.iter_linear():
            self.assertEqual(bias, 0)

    def test_shrink(self):
        bqm = dimod.BQM('BINARY')
        bqm.add_variables_from([('a', 0), ('b', 0), ('c', 0)])
        bqm.set_quadratic('b', 'c', 1)

        self.assertEqual(bqm.resize(2), -1)
        self.assertEqual(bqm.num_variables, 2)
        self.assertEqual(bqm.num_interactions, 0)


class TestScale(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_exclusions(self, name, BQM):
        bqm = BQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.scale(.5, ignored_variables=[0])
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm, BQM({0: -2, 1: 1}, {(0, 1): -.5}, .5, dimod.SPIN))

        bqm = BQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.scale(.5, ignored_interactions=[(1, 0)])
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm, BQM({0: -1, 1: 1}, {(0, 1): -1.}, .5, dimod.SPIN))

        bqm = BQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.scale(.5, ignore_offset=True)
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm, BQM({0: -1, 1: 1}, {(0, 1): -.5}, 1., dimod.SPIN))

    @parameterized.expand(BQMs.items())
    def test_typical(self, name, BQM):
        bqm = BQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.scale(.5)
        self.assertAlmostEqual(bqm.linear, {0: -1., 1: 1.})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -.5})
        self.assertAlmostEqual(bqm.offset, .5)
        assert_consistent_bqm(bqm)

        with self.assertRaises(TypeError):
            bqm.scale('a')


class TestSetLinear(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_basic(self, name, BQM):
        # does not change shape
        bqm = BQM(np.triu(np.ones((3, 3))), dimod.BINARY)

        self.assertEqual(bqm.get_linear(0), 1)
        bqm.set_linear(0, .5)
        self.assertEqual(bqm.get_linear(0), .5)

    @parameterized.expand(BQMs.items())
    def test_set_on_empty(self, name, BQM):
        bqm = BQM('BINARY')
        bqm.set_linear('a', 7)
        self.assertEqual(bqm.get_linear('a'), 7)


class TestSetQuadratic(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_basic(self, name, BQM):
        # does not change shape
        bqm = BQM(np.triu(np.ones((3, 3))), dimod.BINARY)

        self.assertEqual(bqm.get_quadratic(0, 1), 1)
        bqm.set_quadratic(0, 1, .5)
        self.assertEqual(bqm.get_quadratic(0, 1), .5)
        self.assertEqual(bqm.get_quadratic(1, 0), .5)
        bqm.set_quadratic(0, 1, -.5)
        self.assertEqual(bqm.get_quadratic(0, 1), -.5)
        self.assertEqual(bqm.get_quadratic(1, 0), -.5)

    @parameterized.expand(BQMs.items())
    def test_set_on_empty(self, name, BQM):
        bqm = BQM('BINARY')
        bqm.set_quadratic('a', 'b', 7)
        self.assertEqual(bqm.get_quadratic('a', 'b'), 7)

    @parameterized.expand(BQMs.items())
    def test_set_quadratic_exception(self, name, BQM):
        bqm = BQM(dimod.SPIN)
        with self.assertRaises(TypeError):
            bqm.set_quadratic([], 1, .5)
        with self.assertRaises(TypeError):
            bqm.set_quadratic(1, [], .5)


class TestShape(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_3x3array(self, name, BQM):
        bqm = BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]], dimod.BINARY)

        self.assertEqual(bqm.shape, (3, 2))
        self.assertEqual(bqm.num_variables, 3)
        self.assertEqual(bqm.num_interactions, 2)

    @parameterized.expand(BQMs.items())
    def test_disconnected(self, name, BQM):
        bqm = BQM(5, dimod.BINARY)

        self.assertEqual(bqm.shape, (5, 0))
        self.assertEqual(bqm.num_variables, 5)
        self.assertEqual(bqm.num_interactions, 0)

    @parameterized.expand(BQMs.items())
    def test_empty(self, name, BQM):
        self.assertEqual(BQM(dimod.SPIN).shape, (0, 0))
        self.assertEqual(BQM(0, dimod.SPIN).shape, (0, 0))

        self.assertEqual(BQM(dimod.SPIN).num_variables, 0)
        self.assertEqual(BQM(0, dimod.SPIN).num_variables, 0)

        self.assertEqual(BQM(dimod.SPIN).num_interactions, 0)
        self.assertEqual(BQM(0, dimod.SPIN).num_interactions, 0)


class TestSpin(unittest.TestCase):
    def test_init_no_label(self):
        spin_bqm = Spin()
        self.assertIsInstance(spin_bqm.variables[0], str)

    def test_spin_array_int_init(self):
        spin_array = dimod.SpinArray(3)
        self.assertIsInstance(spin_array, np.ndarray)
        for element in spin_array:
            self.assertIsInstance(element, BinaryQuadraticModel)

    def test_spin_array_label_init(self):
        labels = 'ijk'
        spin_array = dimod.SpinArray(labels=labels)
        self.assertIsInstance(spin_array, np.ndarray)
        self.assertEqual(len(spin_array), len(labels))

    def test_multiple_labelled(self):
        r, s, t = dimod.Spins('abc')

        self.assertEqual(r.variables[0], 'a')
        self.assertEqual(s.variables[0], 'b')
        self.assertEqual(t.variables[0], 'c')
        self.assertIs(s.vartype, dimod.SPIN)
        self.assertIs(r.vartype, dimod.SPIN)
        self.assertIs(t.vartype, dimod.SPIN)

    def test_multiple_unlabelled(self):
        r, s, t = dimod.Spins(3)

        self.assertNotEqual(s.variables[0], r.variables[0])
        self.assertNotEqual(s.variables[0], t.variables[0])
        self.assertIs(s.vartype, dimod.SPIN)
        self.assertIs(r.vartype, dimod.SPIN)
        self.assertIs(t.vartype, dimod.SPIN)

    def test_no_label_collision(self):
        bqm_1 = Spin()
        bqm_2 = Spin()
        self.assertNotEqual(bqm_1.variables[0], bqm_2.variables[0])

    def test_serializable_label(self):
        import json
        bqm = Spin()
        json.dumps(bqm.variables.to_serializable())


class TestSymbolic(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_add_number(self, name, BQM):
        bqm = BQM('SPIN')
        new = bqm + 1
        self.assertIsNot(bqm, new)
        self.assertEqual(new.offset, 1)
        self.assertEqual(bqm.num_variables, 0)

    @parameterized.expand(BQMs.items())
    def test_iadd_number(self, name, BQM):
        bqm = BQM('SPIN')
        old = bqm
        bqm += 1
        self.assertIs(bqm, old)
        self.assertEqual(bqm.offset, 1)
        self.assertEqual(bqm.num_variables, 0)

    @parameterized.expand(BQMs.items())
    def test_radd_number(self, name, BQM):
        bqm = BQM('SPIN')
        new = 1 + bqm
        self.assertIsNot(bqm, new)
        self.assertEqual(new.offset, 1)
        self.assertEqual(bqm.num_variables, 0)

    @parameterized.expand(BQMs.items())
    def test_div_number(self, name, BQM):
        bqm = BQM({'u': 2}, {'uv': 4}, 6, 'BINARY')
        ref = bqm
        bqm /= 2
        self.assertIs(bqm, ref)
        self.assertEqual(bqm, BQM({'u': 1}, {'uv': 2}, 3, 'BINARY'))

    @parameterized.expand(BQMs.items())
    def test_exceptions_symbolic_mode(self, name, BQM):
        bqm = BQM('SPIN')
        with self.assertRaises(TypeError):
            bqm + 'a'
        with self.assertRaises(TypeError):
            'a' + bqm
        with self.assertRaises(TypeError):
            bqm += 'a'

        with self.assertRaises(TypeError):
            bqm * 'a'
        with self.assertRaises(TypeError):
            bqm *= 'a'

    def test_expressions_binary(self):
        u = Binary('u')
        v = Binary('v')

        BQM = BinaryQuadraticModel

        self.assertEqual(u*v, BQM({}, {'uv': 1}, 0, 'BINARY'))
        self.assertEqual(u*u, BQM({'u': 1}, {}, 0, 'BINARY'))
        self.assertEqual(u*(v-1), BQM({'u': -1}, {'uv': 1}, 0, 'BINARY'))
        self.assertEqual(-u, BQM({'u': -1}, {}, 0, 'BINARY'))
        self.assertEqual(-u*v, BQM({}, {'uv': -1}, 0, 'BINARY'))
        self.assertEqual(1-u, BQM({'u': -1}, {}, 1, 'BINARY'))
        self.assertEqual(u - v, BQM({'u': 1, 'v': -1}, {}, 0, 'BINARY'))
        self.assertEqual((u - 1)*(v - 1), BQM({'u': -1, 'v': -1}, {'uv': 1}, 1, 'BINARY'))
        self.assertEqual((4*u + 2*u*v + 6) / 2, BQM({'u': 2, 'v': 0}, {'uv': 1}, 3, 'BINARY'))
        self.assertEqual((4*u + 2*u*v + 8) / 2.5, BQM({'u': 1.6, 'v': 0}, {'uv': .8}, 3.2, 'BINARY'))
        self.assertEqual((u - v)**2, (u - v)*(u - v))

    def test_expressions_spin(self):
        u = Spin('u')
        v = Spin('v')

        BQM = BinaryQuadraticModel

        self.assertEqual(u*v, BQM({}, {'uv': 1}, 0, 'SPIN'))
        self.assertEqual(u*u, BQM({'u': 0}, {}, 1, 'SPIN'))
        self.assertEqual(u*(v-1), BQM({'u': -1}, {'uv': 1}, 0, 'SPIN'))
        self.assertEqual(-u, BQM({'u': -1}, {}, 0, 'SPIN'))
        self.assertEqual(-u*v, BQM({}, {'uv': -1}, 0, 'SPIN'))
        self.assertEqual(1-u, BQM({'u': -1}, {}, 1, 'SPIN'))
        self.assertEqual(u - v, BQM({'u': 1, 'v': -1}, {}, 0, 'SPIN'))


class TestToFromSerializable(unittest.TestCase):
    def test_from_serializable_empty_v3(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        s = {'bias_type': 'float32',
             'index_type': 'uint16',
             'info': {},
             'linear_biases': [],
             'num_interactions': 0,
             'num_variables': 0,
             'offset': 0.0,
             'quadratic_biases': [],
             'quadratic_head': [],
             'quadratic_tail': [],
             'type': 'BinaryQuadraticModel',
             'use_bytes': False,
             'variable_labels': [],
             'variable_type': 'SPIN',
             'version': {'bqm_schema': '3.0.0'}}

        self.assertEqual(bqm, dimod.BinaryQuadraticModel.from_serializable(s))

    def test_from_serializable_v3(self):
        linear = {'a': -1, 4: 1, ('a', "complex key"): 3}
        quadratic = {('a', 'c'): 1, ('b', 'c'): 3.0, ('a', 3): -1}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 3, dimod.SPIN)

        s = {'bias_type': 'float32',
             'index_type': 'uint16',
             'info': {},
             'linear_biases': [-1.0, 1.0, 3.0, 0.0, 0.0, 0.0],
             'num_interactions': 3,
             'num_variables': 6,
             'offset': 3.0,
             'quadratic_biases': [1.0, 3.0, -1.0],
             'quadratic_head': [0, 3, 0],
             'quadratic_tail': [3, 4, 5],
             'type': 'BinaryQuadraticModel',
             'use_bytes': False,
             'variable_labels': ['a', 4, ('a', 'complex key'), 'c', 'b', 3],
             'variable_type': 'SPIN',
             'version': {'bqm_schema': '3.0.0'}}

        self.assertEqual(dimod.BinaryQuadraticModel.from_serializable(s), bqm)

    def test_from_serializable_bytes_empty_v3(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        s = {'bias_type': 'float32',
             'index_type': 'uint16',
             'info': {},
             'linear_biases': b'',
             'num_interactions': 0,
             'num_variables': 0,
             'offset': 0.0,
             'quadratic_biases': b'',
             'quadratic_head': b'',
             'quadratic_tail': b'',
             'type': 'BinaryQuadraticModel',
             'use_bytes': True,
             'variable_labels': [],
             'variable_type': 'SPIN',
             'version': {'bqm_schema': '3.0.0'}}

        self.assertEqual(bqm, dimod.BinaryQuadraticModel.from_serializable(s))

    def test_from_serializable_bytes_v3(self):
        linear = {'a': -1, 4: 1, ('a', "complex key"): 3}
        quadratic = {('a', 'c'): 1, ('b', 'c'): 3.0, ('a', 3): -1}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 3, dimod.SPIN)

        s = {'bias_type': 'float32',
             'index_type': 'uint16',
             'info': {},
             'linear_biases': b'\x00\x00\x80\xbf\x00\x00\x80?\x00\x00@@\x00\x00\x00\x00'
                              b'\x00\x00\x00\x00\x00\x00\x00\x00',
             'num_interactions': 3,
             'num_variables': 6,
             'offset': 3.0,
             'quadratic_biases': b'\x00\x00\x80?\x00\x00@@\x00\x00\x80\xbf',
             'quadratic_head': b'\x00\x00\x03\x00\x00\x00',
             'quadratic_tail': b'\x03\x00\x04\x00\x05\x00',
             'type': 'BinaryQuadraticModel',
             'use_bytes': True,
             'variable_labels': ['a', 4, ('a', 'complex key'), 'c', 'b', 3],
             'variable_type': 'SPIN',
             'version': {'bqm_schema': '3.0.0'}}

        self.assertEqual(dimod.BinaryQuadraticModel.from_serializable(s), bqm)

    def test_functional_empty(self):
        # round trip
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        new = dimod.BinaryQuadraticModel.from_serializable(bqm.to_serializable())

        self.assertEqual(bqm, new)

    def test_functional(self):
        linear = {'a': -1, 4: 1, ('a', "complex key"): 3}
        quadratic = {('a', 'c'): 1.5, ('b', 'c'): 3., ('a', 3): -1}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 3, dimod.SPIN)

        new = dimod.BinaryQuadraticModel.from_serializable(bqm.to_serializable())

        self.assertEqual(bqm, new)

    def test_functional_bytes_empty(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        new = dimod.BinaryQuadraticModel.from_serializable(bqm.to_serializable(use_bytes=True))

        self.assertEqual(bqm, new)

    def test_functional_bytes(self):
        linear = {'a': -1, 4: 1, ('a', "complex key"): 3}
        quadratic = {('a', 'c'): 1.5, ('b', 'c'): 3., ('a', 3): -1}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 3, dimod.SPIN)

        new = dimod.BinaryQuadraticModel.from_serializable(bqm.to_serializable(use_bytes=True))

        self.assertEqual(bqm, new)

    def test_variable_labels(self):
        h = {0: 0, 1: 1, np.int64(2): 2, np.float64(3): 3,
             fractions.Fraction(4, 1): 4, fractions.Fraction(5, 2): 5,
             '6': 6}
        J = {}

        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

        json.dumps(bqm.to_serializable())


class TestToIsing(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_spin(self, name, BQM):
        linear = {0: 7.1, 1: 103}
        quadratic = {frozenset((0, 1)): .97}
        offset = 0.3
        vartype = dimod.SPIN

        model = BQM(linear, quadratic, offset, vartype)

        h, J, off = model.to_ising()

        self.assertAlmostEqual(off, offset)
        self.assertEqual(set(linear), set(h))
        for v in linear:
            self.assertAlmostEqual(h[v], linear[v], 5)
        self.assertEqual(set(map(frozenset, J)), set(quadratic))
        for u, v in J:
            self.assertAlmostEqual(J[u, v], quadratic[frozenset([u, v])], 5)

    @parameterized.expand(BQMs.items())
    def test_to_ising_binary_to_ising(self, name, BQM):
        linear = {0: 7.1, 1: 103}
        quadratic = {(0, 1): .97}
        offset = 0.3
        vartype = dimod.BINARY

        model = BQM(linear, quadratic, offset, vartype)

        h, J, off = model.to_ising()

        for spins in itertools.product((-1, 1), repeat=len(model)):
            spin_sample = dict(zip(range(len(spins)), spins))
            bin_sample = {v: (s + 1) // 2 for v, s in spin_sample.items()}

            # calculate the qubo's energy
            energy = off
            for (u, v), bias in J.items():
                energy += spin_sample[u] * spin_sample[v] * bias
            for v, bias in h.items():
                energy += spin_sample[v] * bias

            # and the energy of the model
            self.assertAlmostEqual(energy, model.energy(bin_sample), 5)


class TestToPolyString(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_empty(self, name, BQM):
        self.assertEqual(BQM('BINARY').to_polystring(), "0")
        self.assertEqual(BQM('SPIN').to_polystring(), "0")

    @parameterized.expand(BQMs.items())
    def test_linear_with_offset(self, name, BQM):
        self.assertEqual(BQM({'a': 2}, {}, 7, 'BINARY').to_polystring(), "7 + 2*a")

    @parameterized.expand(BQMs.items())
    def test_linear_without_offset(self, name, BQM):
        self.assertEqual(BQM({'a': 2}, {}, 0, 'BINARY').to_polystring(), "2*a")

    @parameterized.expand(BQMs.items())
    def test_offset_only(self, name, BQM):
        bqm = BQM('BINARY')

        for offset in [-0.0, 0, 0.0]:
            with self.subTest(offset):
                bqm.offset = offset
                self.assertEqual(bqm.to_polystring(), '0')

        for offset in [-7, -7.]:
            with self.subTest(offset):
                bqm.offset = offset
                self.assertEqual(bqm.to_polystring(), '-7')

        for offset in [-7.5]:
            with self.subTest(offset):
                bqm.offset = offset
                self.assertEqual(bqm.to_polystring(), '-7.5')

        for offset in [7, 7.0]:
            with self.subTest(offset):
                bqm.offset = offset
                self.assertEqual(bqm.to_polystring(), '7')

        for offset in [7.5]:
            with self.subTest(offset):
                bqm.offset = offset
                self.assertEqual(bqm.to_polystring(), '7.5')

    @parameterized.expand(BQMs.items())
    def test_quadratic(self, name, BQM):
        bqm = BQM('BINARY')
        bqm.set_quadratic('a', 'b', 3)
        bqm.set_linear('c', 0)

        self.assertIn(bqm.to_polystring(), ['0*c + 3*a*b', '0*c + 3*b*a'])

        # bqm.offset = 6
        bqm.set_quadratic('c', 'a', 7)

        self.assertIn(bqm.to_polystring(), ['3*a*b + 7*a*c', '3*b*a + 7*c*a'])


class TestVartype(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_method_and_property(self, _, BQM):
        bqm = BQM({'a': 1}, {'ab': 1}, 1.5, dimod.BINARY)
        self.assertEqual(bqm.vartype, bqm.vartype())
        self.assertIs(bqm.vartype, bqm.vartype())
        self.assertEqual(str(bqm.vartype), str(bqm.vartype()))
        self.assertEqual(repr(bqm.vartype), repr(bqm.vartype()))

        self.assertEqual(bqm.vartype, bqm.vartype('a'))
        self.assertIs(bqm.vartype, bqm.vartype('a'))
        self.assertEqual(str(bqm.vartype), str(bqm.vartype('a')))
        self.assertEqual(repr(bqm.vartype), repr(bqm.vartype('a')))

    # # unfortunately, because the vartype objects are singletonss,
    # # we don't get this behaviour which would be desired
    # @parameterized.expand(BQMs.items())
    # def test_vartype_reference(self, _, BQM):
    #     bqm = BQM({'a': 1}, {'ab': 1}, 1.5, dimod.BINARY)
    #     vartype = bqm.vartype
    #     self.assertEqual(vartype(), dimod.BINARY)
    #     bqm.change_vartype("SPIN", inplace=True)
    #     self.assertEqual(vartype(), dimod.SPIN)


class TestVartypeViews(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_add_offset_binary(self, name, BQM):
        bqm = BQM({'a': -1}, {'ab': 2}, 1.5, dimod.SPIN)

        with self.assertWarns(DeprecationWarning):
            bqm.binary.add_offset(2)
        self.assertEqual(bqm.offset, 3.5)

    @parameterized.expand(BQMs.items())
    def test_add_offset_spin(self, name, BQM):
        bqm = BQM({'a': -1}, {'ab': 2}, 1.5, dimod.BINARY)

        with self.assertWarns(DeprecationWarning):
            bqm.spin.add_offset(2)
        self.assertEqual(bqm.offset, 3.5)

    @parameterized.expand(BQMs.items())
    def test_binary_binary(self, name, BQM):
        bqm = BQM(dimod.BINARY)
        self.assertIs(bqm.binary, bqm)
        self.assertIs(bqm.binary.binary, bqm)  # and so on

    @parameterized.expand(BQMs.items())
    def test_spin_spin(self, name, BQM):
        bqm = BQM(dimod.SPIN)
        self.assertIs(bqm.spin, bqm)
        self.assertIs(bqm.spin.spin, bqm)  # and so on

    @parameterized.expand(BQMs.items())
    def test_simple_binary(self, name, BQM):
        bqm = BQM({'a': 1, 'b': -3, 'c': 2}, {'ab': -5, 'bc': 6}, 16, 'SPIN')

        assert_consistent_bqm(bqm.binary)
        self.assertIs(bqm.binary.vartype, dimod.BINARY)
        binary = bqm.change_vartype(dimod.BINARY, inplace=False)
        self.assertEqual(binary, bqm.binary)
        self.assertNotEqual(binary, bqm)
        self.assertIs(bqm.binary.spin, bqm)
        self.assertIs(bqm.binary.binary, bqm.binary)  # and so on

    @parameterized.expand(BQMs.items())
    def test_simple_spin(self, name, BQM):
        bqm = BQM({'a': 1, 'b': -3, 'c': 2}, {'ab': -5, 'bc': 6}, 16, 'BINARY')

        assert_consistent_bqm(bqm.spin)
        self.assertIs(bqm.spin.vartype, dimod.SPIN)
        spin = bqm.change_vartype(dimod.SPIN, inplace=False)
        self.assertEqual(spin, bqm.spin)
        self.assertNotEqual(spin, bqm)
        self.assertIs(bqm.spin.binary, bqm)
        self.assertIs(bqm.spin.spin, bqm.spin)  # and so on

    @parameterized.expand(BQM_CLSs.items())
    def test_copy_binary(self, name, BQM):
        bqm = BQM({'a': 1, 'b': -3, 'c': 2}, {'ab': -5, 'bc': 6}, 16, 'SPIN')
        new = bqm.binary.copy()
        self.assertIsNot(new, bqm.binary)
        self.assertIsInstance(new, BQM)

    @parameterized.expand(BQM_CLSs.items())
    def test_copy_spin(self, name, BQM):
        bqm = BQM({'a': 1, 'b': -3, 'c': 2}, {'ab': -5, 'bc': 6}, 16, 'BINARY')
        new = bqm.spin.copy()
        self.assertIsNot(new, bqm.spin)
        self.assertIsInstance(new, BQM)

    @parameterized.expand(BQMs.items())
    def test_offset_binary(self, name, BQM):
        bqm = BQM({'a': 1}, {'ab': 2}, 3, dimod.SPIN)

        bqm.binary.offset -= 2
        self.assertEqual(bqm.offset, 1)

    @parameterized.expand(BQMs.items())
    def test_offset_spin(self, name, BQM):
        bqm = BQM({'a': 1}, {'ab': 2}, 3, dimod.BINARY)

        bqm.spin.offset -= 2
        self.assertEqual(bqm.offset, 1)

    @parameterized.expand(BQMs.items())
    def test_set_linear_binary(self, name, BQM):
        bqm = BQM({0: 1, 1: -3, 2: 2}, {(0, 1): -5, (1, 2): 6}, 16, dimod.SPIN)

        view = bqm.binary
        copy = bqm.change_vartype(dimod.BINARY, inplace=False)

        view.set_linear(0, .5)
        copy.set_linear(0, .5)

        self.assertEqual(view.get_linear(0), .5)
        self.assertEqual(copy.get_linear(0), .5)

        self.assertEqual(view.spin, copy.spin)
        self.assertEqual(view.binary, copy.binary)

    @parameterized.expand(BQMs.items())
    def test_set_linear_spin(self, name, BQM):
        bqm = BQM({0: 1, 1: -3, 2: 2}, {(0, 1): -5, (1, 2): 6}, 16, dimod.BINARY)

        view = bqm.spin
        copy = bqm.change_vartype(dimod.SPIN, inplace=False)

        view.set_linear(0, .5)
        copy.set_linear(0, .5)

        self.assertEqual(view.get_linear(0), .5)
        self.assertEqual(copy.get_linear(0), .5)

        self.assertEqual(view.spin, copy.spin)
        self.assertEqual(view.binary, copy.binary)

    @parameterized.expand(BQMs.items())
    def test_set_offset_binary(self, name, BQM):
        bqm = BQM({0: 1, 1: -3, 2: 2}, {(0, 1): -5, (1, 2): 6}, 16, dimod.SPIN)

        view = bqm.binary
        copy = bqm.change_vartype(dimod.BINARY, inplace=False)

        view.offset = .5
        copy.offset = .5

        self.assertEqual(view.offset, .5)
        self.assertEqual(copy.offset, .5)

        self.assertEqual(view.spin, copy.spin)
        self.assertEqual(view.binary, copy.binary)

    @parameterized.expand(BQMs.items())
    def test_set_offset_spin(self, name, BQM):
        bqm = BQM({0: 1, 1: -3, 2: 2}, {(0, 1): -5, (1, 2): 6}, 16, dimod.BINARY)

        view = bqm.spin
        copy = bqm.change_vartype(dimod.SPIN, inplace=False)

        view.offset = .5
        copy.offset = .5

        self.assertEqual(view.offset, .5)
        self.assertEqual(copy.offset, .5)

        self.assertEqual(view.spin, copy.spin)
        self.assertEqual(view.binary, copy.binary)

    @parameterized.expand(BQMs.items())
    def test_set_quadratic_binary(self, name, BQM):
        bqm = BQM({0: 1, 1: -3, 2: 2}, {(0, 1): -5, (1, 2): 6}, 16, dimod.SPIN)

        view = bqm.binary
        copy = bqm.change_vartype(dimod.BINARY, inplace=False)

        view.set_quadratic(0, 1, -1)
        copy.set_quadratic(0, 1, -1)

        self.assertEqual(view.get_quadratic(0, 1), -1)
        self.assertEqual(copy.get_quadratic(0, 1), -1)

        self.assertEqual(view.spin, copy.spin)
        self.assertEqual(view.binary, copy.binary)

    @parameterized.expand(BQMs.items())
    def test_set_quadratic_spin(self, name, BQM):
        bqm = BQM({0: 1, 1: -3, 2: 2}, {(0, 1): -5, (1, 2): 6}, 16, dimod.BINARY)

        view = bqm.spin
        copy = bqm.change_vartype(dimod.SPIN, inplace=False)

        view.set_quadratic(0, 1, -1)
        copy.set_quadratic(0, 1, -1)

        self.assertEqual(view.get_quadratic(0, 1), -1)
        self.assertEqual(copy.get_quadratic(0, 1), -1)

        self.assertEqual(view.spin, copy.spin)
        self.assertEqual(view.binary, copy.binary)

    @parameterized.expand(BQM_CLSs.items())
    def test_change_vartype_binary(self, name, BQM):
        bqm = BQM({'ab': -1, 'ac': -1, 'bc': -1, 'cd': -1}, 'BINARY')
        bqm.offset = 1

        spin = bqm.spin

        spin.change_vartype('SPIN')  # should do nothing
        self.assertIs(spin.vartype, dimod.SPIN)
        self.assertIs(bqm.spin, spin)

        new = spin.change_vartype('SPIN', inplace=False)
        self.assertIs(new.vartype, dimod.SPIN)
        self.assertIsNot(new, spin)
        self.assertIsInstance(new, BQM)

        new = spin.change_vartype('BINARY', inplace=False)
        self.assertIs(new.vartype, dimod.BINARY)
        self.assertIsNot(new, spin)
        self.assertIsInstance(new, BQM)

        spin.change_vartype('BINARY')
        self.assertIs(spin.vartype, dimod.BINARY)
        self.assertIsNot(bqm.spin, spin)

    @parameterized.expand(BQM_CLSs.items())
    def test_change_vartype_spin(self, name, BQM):
        bqm = BQM({'ab': -1, 'ac': -1, 'bc': -1, 'cd': -1}, 'SPIN')
        bqm.offset = 1

        binary = bqm.binary

        binary.change_vartype('BINARY')  # should do nothing
        self.assertIs(binary.vartype, dimod.BINARY)
        self.assertIs(bqm.binary, binary)

        new = binary.change_vartype('BINARY', inplace=False)
        self.assertIs(new.vartype, dimod.BINARY)
        self.assertIsNot(new, binary)
        self.assertIsInstance(new, BQM)

        new = binary.change_vartype('SPIN', inplace=False)
        self.assertIs(new.vartype, dimod.SPIN)
        self.assertIsNot(new, binary)
        self.assertIsInstance(new, BQM)

        binary.change_vartype('SPIN')
        self.assertIs(binary.vartype, dimod.SPIN)
        self.assertIsNot(bqm.binary, binary)

    @parameterized.expand(BQMs.items())
    def test_consistency_binary_to_spin(self, name, BQM):
        bqm = BQM({'a': 1, 'b': -2}, {'ab': 3, 'bc': 4}, 1.5, 'BINARY')

        spin = bqm.change_vartype('SPIN', inplace=False)
        view = bqm.spin

        self.assertEqual(spin, view)
        self.assertEqual(bqm, spin.binary)

    @parameterized.expand(BQMs.items())
    def test_consistency_spin_to_binary(self, name, BQM):
        bqm = BQM({'a': 1, 'b': -2}, {'ab': 3, 'bc': 4}, 1.5, 'SPIN')

        binary = bqm.change_vartype('BINARY', inplace=False)
        view = bqm.binary

        self.assertEqual(binary, view)
        self.assertEqual(bqm, binary.spin)

    @parameterized.expand(BQMs.items())
    def test_consistent_energies_binary(self, name, BQM):
        bqm = BQM({'a': -7, 'b': -32.2}, {'ab': -5, 'bc': 1.5}, 20.6, 'BINARY')

        bin_sampleset = dimod.ExactSolver().sample(bqm)
        spin_sampleset = dimod.ExactSolver().sample(bqm.spin)

        self.assertEqual(bin_sampleset.change_vartype('SPIN', inplace=False),
                         spin_sampleset)
        self.assertEqual(spin_sampleset.change_vartype('BINARY', inplace=False),
                         bin_sampleset)

    @parameterized.expand(BQMs.items())
    def test_consistent_energies_spin(self, name, BQM):
        bqm = BQM({'a': -7, 'b': -32.2}, {'ab': -5, 'bc': 1.5}, 20.6, 'SPIN')

        bin_sampleset = dimod.ExactSolver().sample(bqm.binary)
        spin_sampleset = dimod.ExactSolver().sample(bqm)

        self.assertEqual(bin_sampleset.change_vartype('SPIN', inplace=False),
                         spin_sampleset)
        self.assertEqual(spin_sampleset.change_vartype('BINARY', inplace=False),
                         bin_sampleset)

    @parameterized.expand([(cls.__name__, cls, inplace)
                           for (cls, inplace)
                           in itertools.product(BQMs.values(), [False, True])])
    def test_relabel_variables_binary(self, name, BQM, inplace):
        # to get a BinaryView, construct in SPIN, and ask for binary
        linear = {0: 1, 1: -3, 2: 2}
        quadratic = {(0, 1): -5, (1, 2): 6}
        offset = 16
        vartype = dimod.SPIN
        view = BQM(linear, quadratic, offset, vartype).binary

        # relabel view
        mapping = {0: 'a', 1: 'b', 2: 'c'}
        new = view.relabel_variables(mapping, inplace=inplace)
        assert_consistent_bqm(new)
        if inplace:
            self.assertIs(view, new)
        else:
            self.assertIsNot(view, new)

        # check that new model is correct
        linear = {'a': 1, 'b': -3, 'c': 2}
        quadratic = {'ab': -5, 'bc': 6}
        offset = 16
        vartype = dimod.SPIN
        test = BQM(linear, quadratic, offset, vartype).binary
        self.assertEqual(new, test)

    @parameterized.expand([(cls.__name__, cls, inplace)
                           for (cls, inplace)
                           in itertools.product(BQMs.values(), [False, True])])
    def test_relabel_variables_spin(self, name, BQM, inplace):
        # to get a SpinView, construct in BINARY, and ask for spin
        linear = {0: 1, 1: -3, 2: 2}
        quadratic = {(0, 1): -5, (1, 2): 6}
        offset = 16
        vartype = dimod.BINARY
        view = BQM(linear, quadratic, offset, vartype).spin

        # relabel view
        mapping = {0: 'a', 1: 'b', 2: 'c'}
        new = view.relabel_variables(mapping, inplace=inplace)
        assert_consistent_bqm(new)
        if inplace:
            self.assertIs(view, new)
        else:
            self.assertIsNot(view, new)

        # check that new model is correct
        linear = {'a': 1, 'b': -3, 'c': 2}
        quadratic = {'ab': -5, 'bc': 6}
        offset = 16
        vartype = dimod.BINARY
        test = BQM(linear, quadratic, offset, vartype).spin
        self.assertEqual(new, test)


class TestToNumpyVectors(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_array_dense(self, name, BQM):
        bqm = BQM(np.arange(9).reshape((3, 3)), 'BINARY')

        ldata, (irow, icol, qdata), off = bqm.to_numpy_vectors()

        np.testing.assert_array_equal(ldata, [0, 4, 8])

        self.assertTrue(np.issubdtype(irow.dtype, np.integer))
        self.assertTrue(np.issubdtype(icol.dtype, np.integer))
        self.assertEqual(len(irow), len(icol))
        self.assertEqual(len(icol), len(qdata))
        self.assertEqual(len(qdata), len(bqm.quadratic))
        for u, v, bias in zip(irow, icol, qdata):
            self.assertAlmostEqual(bqm.adj[u][v], bias)

    @parameterized.expand(BQMs.items())
    def test_array_reversed_order(self, name, BQM):
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

    @parameterized.expand(BQMs.items())
    def test_array_sparse(self, name, BQM):
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

    @parameterized.expand(BQMs.items())
    def test_array_sparse_return_labels(self, name, BQM):
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

    @parameterized.expand(BQMs.items())
    def test_dict(self, name, BQM):
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

    @parameterized.expand(BQMs.items())
    def test_dict_return_labels(self, name, BQM):
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

    @parameterized.expand(BQMs.items())
    def test_empty(self, name, BQM):
        bqm = BQM('SPIN')
        h, (i, j, values), off = bqm.to_numpy_vectors()

        np.testing.assert_array_equal(h, [])
        np.testing.assert_array_equal(i, [])
        np.testing.assert_array_equal(j, [])
        np.testing.assert_array_equal(values, [])
        self.assertEqual(off, 0)

    @parameterized.expand(BQMs.items())
    def test_namedtuple(self, name, BQM):
        bqm = BQM({'c': 1, 'a': -1}, {'ba': 1, 'bc': -2}, 0, dimod.SPIN)

        nt = bqm.to_numpy_vectors()
        ntl = bqm.to_numpy_vectors(return_labels=True)

        self.assertEqual(ntl.labels, list('abc'))

        np.testing.assert_array_equal(nt.linear_biases, [-1, 0, 1])
        np.testing.assert_array_equal(ntl.linear_biases, [-1, 0, 1])

    @parameterized.expand(BQMs.items())
    def test_unsorted_labels(self, name, BQM):
        bqm = BQM(OrderedDict([('b', -1), ('a', 1)]), {}, 'SPIN')

        ldata, (irow, icol, qdata), off, order \
            = bqm.to_numpy_vectors(return_labels=True, sort_labels=False)

        self.assertEqual(order, ['b', 'a'])

        np.testing.assert_array_equal(ldata, [-1, 1])
        np.testing.assert_array_equal(irow, [])
        np.testing.assert_array_equal(icol, [])
        np.testing.assert_array_equal(qdata, [])
        self.assertEqual(off, 0)

    @parameterized.expand(BQM_CLSs.items())
    def test_sort_indices(self, name, BQM):
        bqm = BQM.from_ising({}, {(0, 1): .5, (3, 2): -1, (0, 3): 1.5})

        h, (i, j, values), off = bqm.to_numpy_vectors(sort_indices=True)

        np.testing.assert_array_equal(h, [0, 0, 0, 0])
        np.testing.assert_array_equal(i, [0, 0, 2])
        np.testing.assert_array_equal(j, [1, 3, 3])
        np.testing.assert_array_equal(values, [.5, 1.5, -1])


class TestToQUBO(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_binary(self, name, BQM):
        linear = {0: 0, 1: 0}
        quadratic = {(0, 1): 1}
        offset = 0.0
        vartype = dimod.BINARY

        model = BQM(linear, quadratic, offset, vartype)

        Q, off = model.to_qubo()

        self.assertEqual(off, offset)
        self.assertEqual(len(Q), 3)
        self.assertEqual(Q[0, 0], 0)
        self.assertEqual(Q[1, 1], 0)
        if (0, 1) in Q:
            self.assertEqual(Q[0, 1], 1)
        elif (1, 0) in Q:
            self.assertEqual(Q[1, 0], 1)
        else:
            self.assertTrue(False)

    @parameterized.expand(BQMs.items())
    def test_spin(self, name, BQM):
        linear = {0: .5, 1: 1.3}
        quadratic = {(0, 1): -.435}
        offset = 1.2
        vartype = dimod.SPIN

        model = BQM(linear, quadratic, offset, vartype)

        Q, off = model.to_qubo()

        for spins in itertools.product((-1, 1), repeat=len(model)):
            spin_sample = dict(zip(range(len(spins)), spins))
            bin_sample = {v: (s + 1) // 2 for v, s in spin_sample.items()}

            # calculate the qubo's energy
            energy = off
            for (u, v), bias in Q.items():
                energy += bin_sample[u] * bin_sample[v] * bias

            # and the energy of the model
            self.assertAlmostEqual(energy, model.energy(spin_sample), 5)


class TestUpdate(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_cross_vartype(self, name, BQM):
        binary = BQM({'a': .3}, {('a', 'b'): -1}, 0, dimod.BINARY)
        spin = BQM({'c': -1}, {('b', 'c'): 1}, 1.2, dimod.SPIN)

        binary.update(spin)

        target = BQM({'a': .3, 'b': -2, 'c': -4}, {'ab': -1, 'cb': 4},
                     3.2, dimod.BINARY)

        self.assertEqual(binary, target)

    @parameterized.expand(BQMs.items())
    def test_simple(self, name, BQM):
        bqm0 = BQM({'a': -1}, {'ab': 1}, 1.5, 'SPIN')
        bqm1 = BQM({'c': 3, 'a': -2}, {'ab': 5, 'cb': 1}, 1.5, 'SPIN')

        bqm0.update(bqm1)

        target = BQM({'a': -3, 'c': 3}, {'ba': 6, 'cb': 1}, 3, 'SPIN')

        self.assertEqual(bqm0, target)


class TestViews(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_adj_setitem(self, name, BQM):
        bqm = BQM({'ab': -1}, 'SPIN')
        bqm.adj['a']['b'] = 5
        self.assertEqual(bqm.adj['a']['b'], 5)
        assert_consistent_bqm(bqm)  # all the other cases

    @parameterized.expand(BQMs.items())
    def test_adj_neighborhoods(self, name, BQM):
        bqm = BQM({'ab': -1, 'ac': -1, 'bc': -1, 'cd': -1}, 'SPIN')

        self.assertEqual(len(bqm.adj['a']), 2)
        self.assertEqual(len(bqm.adj['b']), 2)
        self.assertEqual(len(bqm.adj['c']), 3)
        self.assertEqual(len(bqm.adj['d']), 1)

    @parameterized.expand(BQMs.items())
    def test_linear_delitem(self, name, BQM):
        bqm = BQM([[0, 1, 2, 3, 4],
                   [0, 6, 7, 8, 9],
                   [0, 0, 10, 11, 12],
                   [0, 0, 0, 13, 14],
                   [0, 0, 0, 0, 15]], 'BINARY')
        del bqm.linear[2]
        self.assertEqual(set(bqm.variables), set([0, 1, 3, 4]))

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

        assert_consistent_bqm(bqm)

        with self.assertRaises(KeyError):
            del bqm.linear[2]

    @parameterized.expand(BQMs.items())
    def test_linear_setitem(self, name, BQM):
        bqm = BQM({'ab': -1}, dimod.SPIN)
        bqm.linear['a'] = 5
        self.assertEqual(bqm.get_linear('a'), 5)
        assert_consistent_bqm(bqm)

    @parameterized.expand(BQM_CLSs.items())
    def test_linear_sum(self, name, BQM):
        bqm = BQM.from_ising({'a': -1, 'b': 2}, {'ab': 1, 'bc': 1})
        self.assertEqual(bqm.linear.sum(), 1)
        self.assertEqual(bqm.linear.sum(start=5), 6)

    @parameterized.expand(BQM_CLSs.items())
    def test_linear_update(self, name, BQM):
        bqm = BQM('SPIN')
        bqm.linear.update({'a': -1.0, 'b': 1.0, 'c': 1.0})
        self.assertEqual(bqm.linear, {'a': -1.0, 'b': 1.0, 'c': 1.0})

    @parameterized.expand(BQM_CLSs.items())
    def test_neighborhood_max(self, name, BQM):
        bqm = BQM.from_ising({}, {'ab': 1, 'ac': 2, 'bc': 3})
        self.assertEqual(bqm.adj['a'].max(), 2)
        self.assertEqual(bqm.adj['b'].max(), 3)
        self.assertEqual(bqm.adj['c'].max(), 3)

    @parameterized.expand(BQM_CLSs.items())
    def test_neighborhood_max_empty(self, name, BQM):
        bqm = BQM.from_ising({'a': 1}, {})

        with self.assertRaises(ValueError):
            bqm.adj['a'].max()

        self.assertEqual(bqm.adj['a'].max(default=5), 5)

    @parameterized.expand(BQM_CLSs.items())
    def test_neighborhood_min(self, name, BQM):
        bqm = BQM.from_ising({}, {'ab': -1, 'ac': -2, 'bc': -3})
        self.assertEqual(bqm.adj['a'].min(), -2)
        self.assertEqual(bqm.adj['b'].min(), -3)
        self.assertEqual(bqm.adj['c'].min(), -3)

    @parameterized.expand(BQM_CLSs.items())
    def test_neighborhood_min_empty(self, name, BQM):
        bqm = BQM.from_ising({'a': 1}, {})

        with self.assertRaises(ValueError):
            bqm.adj['a'].min()

        self.assertEqual(bqm.adj['a'].min(default=5), 5)

    @parameterized.expand(BQM_CLSs.items())
    def test_neighborhood_sum(self, name, BQM):
        bqm = BQM.from_ising({}, {'ab': -1, 'ac': -2, 'bc': -3})
        self.assertEqual(bqm.adj['a'].sum(), -3)
        self.assertEqual(bqm.adj['b'].sum(), -4)
        self.assertEqual(bqm.adj['c'].sum(), -5)

    @parameterized.expand(BQM_CLSs.items())
    def test_neighborhood_sum_empty(self, name, BQM):
        bqm = BQM.from_ising({'a': 1}, {})
        self.assertEqual(bqm.adj['a'].sum(), 0)
        self.assertEqual(bqm.adj['a'].sum(start=5), 5)

    @parameterized.expand(BQMs.items())
    def test_quadratic_delitem(self, name, BQM):
        bqm = BQM([[0, 1, 2, 3, 4],
                   [0, 6, 7, 8, 9],
                   [0, 0, 10, 11, 12],
                   [0, 0, 0, 13, 14],
                   [0, 0, 0, 0, 15]], 'SPIN')
        del bqm.quadratic[0, 1]
        self.assertEqual(set(dict(bqm.iter_neighborhood(0))), set([2, 3, 4]))
        assert_consistent_bqm(bqm)

        with self.assertRaises(KeyError):
            del bqm.quadratic[0, 1]

    @parameterized.expand(BQMs.items())
    def test_quadratic_items(self, name, BQM):
        bqm = BQM({}, {'ab': 1, 'bc': 2, 'cd': 3}, 'SPIN')
        self.assertEqual(dict(bqm.quadratic.items()), bqm.quadratic)

    @parameterized.expand(BQMs.items())
    def test_quadratic_setitem(self, name, BQM):
        bqm = BQM({'ab': -1}, dimod.SPIN)
        bqm.quadratic[('a', 'b')] = 5
        self.assertEqual(bqm.get_quadratic('a', 'b'), 5)
        assert_consistent_bqm(bqm)

    @parameterized.expand(BQM_CLSs.items())
    def test_quadratic_sum(self, name, BQM):
        bqm = BQM.from_ising({'a': -1, 'b': 2}, {'ab': -1, 'bc': 6})
        self.assertEqual(bqm.quadratic.sum(), 5)
        self.assertEqual(bqm.quadratic.sum(start=5), 10)

    @parameterized.expand(BQM_CLSs.items())
    def test_quadratic_sum_cybqm(self, name, BQM):
        # make sure it doesn't use python's sum
        bqm = BQM.from_ising({'a': -1, 'b': 2}, {'ab': -1, 'bc': 6})

        def _sum(*args, **kwargs):
            raise Exception('boom')

        with unittest.mock.patch('builtins.sum', _sum):
            bqm.quadratic.sum()

    @parameterized.expand(BQMs.items())
    def test_lin_minmax(self, name, BQM):
        num_vars = 10
        D = np.arange(num_vars*num_vars).reshape((num_vars, num_vars))
        bqm = BQM(D, 'SPIN')

        lmin = min(bqm.linear.values())
        self.assertEqual(lmin, bqm.linear.min())

        lmax = max(bqm.linear.values())
        self.assertEqual(lmax, bqm.linear.max())

    @parameterized.expand(BQMs.items())
    def test_quad_minmax(self, name, BQM):
        num_vars = 10
        D = np.arange(num_vars*num_vars).reshape((num_vars, num_vars))
        bqm = BQM(D, 'SPIN')

        qmin = min(bqm.quadratic.values())
        self.assertEqual(qmin, bqm.quadratic.min())

        qmax = max(bqm.quadratic.values())
        self.assertEqual(qmax, bqm.quadratic.max())

    @parameterized.expand(BQMs.items())
    def test_lin_minmax_empty(self, name, BQM):
        bqm = BQM('SPIN')

        # Test when default is not set
        with self.assertRaises(ValueError):
            bqm.linear.min()

        with self.assertRaises(ValueError):
            bqm.linear.max()

        # Test when default is set
        self.assertEqual(bqm.linear.min(default=1), 1)
        self.assertEqual(bqm.linear.max(default=2), 2)

    @parameterized.expand(BQMs.items())
    def test_quad_minmax_empty(self, name, BQM):
        bqm = BQM(500, 'SPIN')

        # Test when default is not set
        with self.assertRaises(ValueError):
            bqm.quadratic.min()

        with self.assertRaises(ValueError):
            bqm.quadratic.max()

        # Test when default is set
        self.assertEqual(bqm.quadratic.min(default=1), 1)
        self.assertEqual(bqm.quadratic.max(default=2), 2)


class TestConstraint(unittest.TestCase):
    @parameterized.expand(BQMs.items())
    def test_simple_constraint(self, name, BQM):
        bqm = BQM('BINARY')
        num_variables = 2
        num_cases = 3
        x = {}
        for i in range(num_variables):
            x[i] = bqm.add_variable('x_{i}'.format(i=i))

        bqm.add_linear_equality_constraint(
            [(x[i], 1.0) for i in range(num_variables)],
            lagrange_multiplier=1.0, constant=-1.0)

        for i in x:
            for case in range(num_cases):
                self.assertEqual(bqm.get_linear(x[i]), -1)
            for j in x:
                if j > i:
                    for case in range(num_cases):
                        self.assertEqual(bqm.get_quadratic(x[i], x[j]), 2.0)

    @parameterized.expand(BQMs.items())
    def test_inequality_constraint(self, name, BQM):
        bqm = BQM('BINARY')
        num_variables = 3
        x = {}
        for i in range(num_variables):
            x[i] = bqm.add_variable('x_{i}'.format(i=i))
        slacks = [('slack_inequality0_0', 1), ('slack_inequality0_1', 2),
                  ('slack_inequality0_2', 1)]
        terms = iter([(x[i], 2.0) for i in range(num_variables)])
        slack_terms = bqm.add_linear_inequality_constraint(
            terms, lagrange_multiplier=1.0, constant=-4.0, label='inequality0')
        self.assertTrue(slacks == slack_terms)
        for i in x:
            self.assertEqual(bqm.get_linear(x[i]), -12)
            for j in x:
                if j > i:
                    self.assertEqual(bqm.get_quadratic(x[i], x[j]), 8.0)

    @parameterized.expand(BQMs.items())
    def test_inequality_constraint_cross_zero(self, name, BQM):
        bqm = BQM('BINARY')
        num_variables = 5
        x = {}
        for i in range(num_variables):
            x[i] = bqm.add_variable('x_{i}'.format(i=i))
        slacks = [('slack_inequality0_0', 1), ('slack_inequality0_1', 2),
                  ('slack_inequality0_2', 3), ('slack_inequality0_3', 4.0)]
        slack_terms = bqm.add_linear_inequality_constraint(
            [(x[i], 2.0) for i in range(num_variables)],
            lagrange_multiplier=1.0, constant=4.0, lb=8, ub=20, cross_zero=True,
            label='inequality0')
        self.assertTrue(slacks == slack_terms)
        for i in x:
            self.assertEqual(bqm.get_linear(x[i]), -36)
            for j in x:
                if j > i:
                    self.assertEqual(bqm.get_quadratic(x[i], x[j]), 8.0)

    @parameterized.expand(BQMs.items())
    def test_inequality_equality(self, name, BQM):
        bqm1 = BQM('BINARY')
        slacks = bqm1.add_linear_inequality_constraint(
            [('a', 1), ('b', 1), ('c', 1)],
            constant=-1,
            lb=0,
            ub=0,
            lagrange_multiplier=1.0,
            label='a'
        )
        self.assertTrue(len(slacks) == 0)

        bqm2 = BQM('BINARY')
        slacks = bqm2.add_linear_inequality_constraint(
            [('a', 1), ('b', 1), ('c', 1)],
            constant=0,
            lb=1,
            ub=1,
            lagrange_multiplier=1.0,
            label='a'
        )
        self.assertTrue(len(slacks) == 0)

        bqm_equal = BQM('BINARY')
        bqm_equal.add_linear_equality_constraint(
            [('a', 1), ('b', 1), ('c', 1)],
            constant=-1,
            lagrange_multiplier=1.0)

        self.assertTrue(len(slacks) == 0)
        self.assertEqual(bqm_equal, bqm1)
        self.assertEqual(bqm_equal, bqm2)

    @parameterized.expand(BQMs.items())
    def test_inequality_constraint_unbalanced(self, name, BQM):
        bqm = BQM('BINARY')
        num_variables = 3
        x = {}
        for i in range(num_variables):
            x[i] = bqm.add_variable('x_{i}'.format(i=i))
        terms = iter([(x[i], 2.0) for i in range(num_variables)])
        unbalanced_terms = bqm.add_linear_inequality_constraint(
            terms, lagrange_multiplier=[1.0, 1.0], label='inequality0', constant=0.0, ub=5,
            penalization_method="unbalanced")
        self.assertTrue(len(unbalanced_terms) == 0)
        for i in x:
            self.assertEqual(bqm.get_linear(x[i]), -14.0)
            for j in x:
                if j > i:
                    self.assertEqual(bqm.get_quadratic(x[i], x[j]), 8.0)

    @parameterized.expand(BQMs.items())
    def test_simple_constraint_iterator(self, name, BQM):
        bqm = BQM('BINARY')
        num_variables = 2
        num_cases = 3
        x = {}
        for i in range(num_variables):
            x[i] = bqm.add_variable('x_{i}'.format(i=i))

        bqm.add_linear_equality_constraint(
            ((x[i], 1.0) for i in range(num_variables)),
            lagrange_multiplier=1.0, constant=-1.0)

        for i in x:
            for case in range(num_cases):
                self.assertEqual(bqm.get_linear(x[i]), -1)
            for j in x:
                if j > i:
                    for case in range(num_cases):
                        self.assertEqual(bqm.get_quadratic(x[i], x[j]), 2.0)

    @parameterized.expand(BQMs.items())
    def test_more_constraint(self, name, BQM):
        bqm = BQM('BINARY')
        x = bqm.add_variable('x')
        y = bqm.add_variable('y')
        w = bqm.add_variable('w')

        expression = [(x, 1.0), (y, 2.0), (w, 1.0)]
        constant = -2.0
        bqm.add_linear_equality_constraint(
            expression,
            lagrange_multiplier=1.0, constant=constant)

        for cx, cy, cw in itertools.product(range(2), repeat=3):
            s = constant
            state = {'x': cx, 'y': cy, 'w': cw}
            for v, bias in expression:
                if state[v]:
                    s += bias
            self.assertAlmostEqual(bqm.energy(state), s ** 2)

    @parameterized.expand(BQMs.items())
    def test_random_constraint(self, name, BQM):
        num_variables = 4
        bqm_0 = dimod.generators.gnp_random_bqm(n=num_variables, p=0.5,
                                                vartype=dimod.BINARY)
        bqm = bqm_0.copy()
        x = list(bqm.variables)

        expression = [(x[i], np.random.randint(0, 10)) for i in x]
        constant = np.random.randint(1, 10) * num_variables
        lagrange_multiplier = np.random.randint(1, 10)
        bqm.add_linear_equality_constraint(
            expression,
            lagrange_multiplier=lagrange_multiplier, constant=constant)

        for binary_values in itertools.product(range(2), repeat=num_variables):
            state = {x[i]: binary_values[i] for i in x}
            energy = bqm.energy(state)
            s = constant
            for v, bias in expression:
                if state[v]:
                    s += bias

            self.assertAlmostEqual(energy,
                                   lagrange_multiplier * s ** 2 +
                                   bqm_0.energy(state))

    @parameterized.expand(BQMs.items())
    def test_spin(self, name, BQM):
        terms = [('r', -2), ('s', 1), ('t', 4)]

        bqm = BQM('SPIN')
        bqm.add_linear_equality_constraint(terms, 1, 0)

        for spins in itertools.product((-1, 1), repeat=3):
            sample = dict(zip('rst', spins))

            self.assertAlmostEqual(bqm.energy(sample),
                                   sum(sample[v]*b for v, b in terms)**2)


class TestAddBQM(unittest.TestCase):
    @parameterized.expand(itertools.product(BQMs.values(), repeat=2))
    def test_add_empty_bqm(self, BQM0, BQM1):
        for vtype0, vtype1 in itertools.product(*[("BINARY", "SPIN")]*2):
            empty = BQM0(vtype0)
            self.assertEqual(empty, empty + BQM1(vtype1))
            self.assertEqual(empty.change_vartype(vtype1),
                             BQM1(vtype1) + empty)

            empty_offset = BQM0(vtype0)
            empty_offset.offset = 3
            self.assertEqual(empty_offset, empty_offset + BQM1(vtype1))
            self.assertEqual(empty_offset.change_vartype(vtype1),
                             BQM1(vtype1) + empty_offset)

            nonempty = BQM0([[1]], vtype0)
            nonempty.offset = 3
            self.assertEqual(nonempty, nonempty + BQM1(vtype1))
            self.assertEqual(nonempty.change_vartype(vtype1),
                             BQM1(vtype1) + nonempty)
