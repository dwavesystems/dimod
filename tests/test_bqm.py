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
import numbers
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

from dimod.binary import BinaryQuadraticModel as newBQM

from dimod.testing import assert_consistent_bqm

BQM_CYTHON_SUBCLASSES = [dimod.AdjVectorBQM]

BQM_SHAPEABLE_SUBCLASSES = [dimod.AdjVectorBQM]

BQM_SUBCLASSES = [dimod.AdjDictBQM,
                  dimod.AdjVectorBQM,
                  dimod.BinaryQuadraticModel,
                  newBQM,
                  ]


class TestAddOffset(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_typical(self, name, BQM):
        bqm = BQM({}, {'ab': -1}, 1.5, 'SPIN')
        bqm.add_offset(2)
        self.assertEqual(bqm.offset, 3.5)


class TestAddVariable(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_bad_variable_type(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        bqm = BQM(dimod.SPIN)
        with self.assertRaises(TypeError):
            bqm.add_variable([])

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_bias_new_variable(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        bqm = BQM(dimod.BINARY)
        bqm.add_variable(bias=5)

        self.assertEqual(bqm.linear, {0: 5})

        bqm.add_variable('a', -6)
        self.assertEqual(bqm.linear, {0: 5, 'a': -6})

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_bias_additive(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        bqm = BQM(dimod.BINARY)
        bqm.add_variable(bqm.add_variable(bias=3), 3)

        self.assertEqual(bqm.linear, {0: 6})

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_index_labelled(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        bqm = BQM(dimod.SPIN)
        self.assertEqual(bqm.add_variable(1), 1)
        self.assertEqual(bqm.add_variable(), 0)  # 1 is already taken
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(bqm.add_variable(), 2)
        self.assertEqual(bqm.shape, (3, 0))

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_labelled(self, name, BQM):
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_unlabelled(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        bqm = BQM(dimod.SPIN)
        bqm.add_variable()
        bqm.add_variable()
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(list(bqm.iter_variables()), [0, 1])


class TestAddVariablesFrom(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_iterable(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        # add from 2-tuples
        bqm = BQM(dimod.SPIN)
        bqm.add_variables_from(iter([('a', .5), ('b', -.5)]))

        self.assertEqual(bqm.linear, {'a': .5, 'b': -.5})

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_mapping(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        bqm = BQM(dimod.SPIN)
        bqm.add_variables_from({'a': .5, 'b': -.5})

        self.assertEqual(bqm.linear, {'a': .5, 'b': -.5})

        # check that it's additive
        bqm.add_variables_from({'a': -1, 'b': 3, 'c': 4})

        self.assertEqual(bqm.linear, {'a': -.5, 'b': 2.5, 'c': 4})


class TestAddInteractionsFrom(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_iterable(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        bqm = BQM(dimod.SPIN)
        bqm.add_interactions_from({('a', 'b'): -.5})
        self.assertEqual(bqm.adj, {'a': {'b': -.5},
                                   'b': {'a': -.5}})

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_mapping(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest

        bqm = BQM(dimod.SPIN)
        bqm.add_interactions_from([('a', 'b', -.5)])
        self.assertEqual(bqm.adj, {'a': {'b': -.5},
                                   'b': {'a': -.5}})


class TestAdjacency(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_contains(self, name, BQM):
        bqm = BQM({0: 1.0}, {(0, 1): 2.0, (2, 1): 0.4}, 0.0, dimod.SPIN)

        self.assertIn(0, bqm.adj[1])
        self.assertEqual(2.0, bqm.adj[1][0])
        self.assertIn(1, bqm.adj[0])
        self.assertEqual(2.0, bqm.adj[0][1])

        self.assertIn(2, bqm.adj[1])
        self.assertEqual(.4, bqm.adj[1][2])
        self.assertIn(1, bqm.adj[2])
        self.assertEqual(.4, bqm.adj[2][1])

        self.assertNotIn(2, bqm.adj[0])
        with self.assertRaises(KeyError):
            bqm.adj[0][2]
        self.assertNotIn(0, bqm.adj[2])
        with self.assertRaises(KeyError):
            bqm.adj[2][0]


# class TestAsBQM(unittest.TestCase):
#     def test_basic(self):
#         bqm = dimod.as_bqm({0: -1}, {(0, 1): 5}, 1.6, dimod.SPIN)

#         assert_consistent_bqm(bqm)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_bqm_input(self, name, BQM):
#         bqm = BQM({'ab': -1}, dimod.BINARY)

#         self.assertIs(dimod.as_bqm(bqm), bqm)
#         self.assertEqual(dimod.as_bqm(bqm), bqm)
#         self.assertIsNot(dimod.as_bqm(bqm, copy=True), bqm)
#         self.assertEqual(dimod.as_bqm(bqm, copy=True), bqm)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_bqm_input_change_vartype(self, name, BQM):
#         bqm = BQM({'ab': -1}, dimod.BINARY)

#         self.assertEqual(dimod.as_bqm(bqm, 'SPIN').vartype, dimod.SPIN)

#         self.assertIs(dimod.as_bqm(bqm, 'BINARY'), bqm)
#         self.assertIsNot(dimod.as_bqm(bqm, 'BINARY', copy=True), bqm)
#         self.assertEqual(dimod.as_bqm(bqm, 'BINARY', copy=True), bqm)

#     def test_type_target(self):

#         for source, target in itertools.product(BQM_SUBCLASSES, repeat=2):
#             bqm = source({'a': -1}, {}, dimod.BINARY)
#             new = dimod.as_bqm(bqm, cls=target)

#             self.assertIsInstance(new, target)
#             self.assertEqual(bqm, new)

#             if issubclass(source, target):
#                 self.assertIs(bqm, new)
#             else:
#                 self.assertIsNot(bqm, new)

#     def test_type_target_copy(self):

#         for source, target in itertools.product(BQM_SUBCLASSES, repeat=2):
#             bqm = source({'a': -1}, {}, dimod.BINARY)
#             new = dimod.as_bqm(bqm, cls=target, copy=True)

#             self.assertIsInstance(new, target)
#             self.assertEqual(bqm, new)
#             self.assertIsNot(bqm, new)

#     def test_type_filter_empty(self):
#         subclasses = list(BQM_SUBCLASSES)

#         if len(subclasses) < 1:
#             return

#         BQM = subclasses[0]

#         with self.assertRaises(ValueError):
#             dimod.as_bqm(BQM('SPIN'), cls=[])

#     def test_type_filter_exclusive(self):
#         subclasses = list(BQM_SUBCLASSES)

#         if len(subclasses) < 3:
#             return

#         BQM0, BQM1, BQM2 = subclasses[0], subclasses[1], subclasses[2]

#         bqm = BQM0({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, .4, 'SPIN')

#         new = dimod.as_bqm(bqm, cls=[BQM1, BQM2])

#         self.assertIsInstance(new, (BQM1, BQM2))
#         self.assertEqual(new, bqm)

#     def test_type_filter_inclusive(self):
#         subclasses = list(BQM_SUBCLASSES)

#         if len(subclasses) < 2:
#             return

#         BQM0, BQM1 = subclasses[0], subclasses[1]

#         bqm = BQM0({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, .4, 'SPIN')

#         new = dimod.as_bqm(bqm, cls=[BQM1, BQM0])

#         self.assertIsInstance(new, BQM0)
#         self.assertIs(new, bqm)  # pass through


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

            self.assertAlmostEqual(spin_energy, binary_energy)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_change_vartype_binary_to_binary_copy(self, name, BQM):
        bqm = BQM({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, .4, 'BINARY')

        new = bqm.change_vartype(dimod.BINARY, inplace=False)
        self.assertEqual(bqm, new)
        self.assertIsNot(bqm, new)  # should be a copy

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_change_vartype_binary_to_spin_copy(self, name, BQM):
        bqm = BQM({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, .4, 'BINARY')

        # change vartype
        new = bqm.change_vartype(dimod.SPIN, inplace=False)

        self.assertConsistentEnergies(spin=new, binary=bqm)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_change_vartype_spin_to_spin_copy(self, name, BQM):
        bqm = BQM({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, 1.4, 'SPIN')

        new = bqm.change_vartype(dimod.SPIN, inplace=False)
        self.assertEqual(bqm, new)
        self.assertIsNot(bqm, new)  # should be a copy

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_change_vartype_spin_to_binary_copy(self, name, BQM):
        bqm = BQM({0: 1, 1: -1, 2: .5}, {(0, 1): .5, (1, 2): 1.5}, 1.4, 'SPIN')

        # change vartype
        new = bqm.change_vartype(dimod.BINARY, inplace=False)

        self.assertConsistentEnergies(spin=bqm, binary=new)


class TestConstruction(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_array_like_1var(self, name, BQM):
        D = [[1]]
        bqm = BQM(D, 'BINARY')
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.shape, (1, 0))
        self.assertEqual(bqm.linear[0], 1)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_array_like_spin(self, name, BQM):
        D = np.ones((5, 5)).tolist()
        bqm = BQM(D, 'SPIN')
        assert_consistent_bqm(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 0)
        self.assertEqual(bqm.offset, 5)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_array_linear(self, name, BQM):
        ldata = np.ones(5)
        qdata = np.ones((5, 5))
        bqm = BQM(ldata, qdata, 'BINARY')
        assert_consistent_bqm(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 2)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_array_types(self, name, BQM):
        # comes from a bug where this was returning an array
        h = [0, 1, 2]
        J = np.asarray([[0, 1, 2], [0, 0, 3], [0, 0, 0]])
        bqm = BQM(h, J, 'SPIN')
        for bias in bqm.quadratic.values():
            self.assertIsInstance(bias, numbers.Number)

    # def test_bqm_binary(self):
    #     linear = {'a': -1, 'b': 1, 0: 1.5}
    #     quadratic = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
    #     offset = 0
    #     vartype = dimod.BINARY
    #     for source, target in itertools.product(BQM_SUBCLASSES, repeat=2):
    #         with self.subTest(source=source, target=target):
    #             bqm = source(linear, quadratic, offset, vartype)
    #             new = target(bqm)

    #             self.assertIsInstance(new, target)
    #             assert_consistent_bqm(new)
    #             self.assertEqual(bqm.adj, new.adj)
    #             self.assertEqual(bqm.offset, new.offset)
    #             self.assertEqual(bqm.vartype, new.vartype)

    # def test_bqm_spin(self):
    #     linear = {'a': -1, 'b': 1, 0: 1.5}
    #     quadratic = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
    #     offset = 0
    #     vartype = dimod.SPIN
    #     for source, target in itertools.product(BQM_SUBCLASSES, repeat=2):
    #         with self.subTest(source=source, target=target):
    #             bqm = source(linear, quadratic, offset, vartype)
    #             new = target(bqm)

    #             self.assertIsInstance(new, target)
    #             assert_consistent_bqm(new)
    #             self.assertEqual(bqm.adj, new.adj)
    #             self.assertEqual(bqm.offset, new.offset)
    #             self.assertEqual(bqm.vartype, new.vartype)

    # def test_bqm_binary_to_spin(self):
    #     linear = {'a': -1, 'b': 1, 0: 1.5}
    #     quadratic = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
    #     offset = 0
    #     vartype = dimod.BINARY
    #     for source, target in itertools.product(BQM_SUBCLASSES, repeat=2):
    #         with self.subTest(source=source, target=target):
    #             bqm = source(linear, quadratic, offset, vartype)
    #             new = target(bqm, vartype=dimod.SPIN)

    #             self.assertIsInstance(new, target)
    #             assert_consistent_bqm(new)
    #             self.assertEqual(new.vartype, dimod.SPIN)

    #             # change back for equality check
    #             new.change_vartype(dimod.BINARY)
    #             self.assertEqual(bqm.adj, new.adj)
    #             self.assertEqual(bqm.offset, new.offset)
    #             self.assertEqual(bqm.vartype, new.vartype)

    # def test_bqm_spin_to_binary(self):
    #     linear = {'a': -1, 'b': 1, 0: 1.5}
    #     quadratic = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
    #     offset = 0
    #     vartype = dimod.SPIN
    #     for source, target in itertools.product(BQM_SUBCLASSES, repeat=2):
    #         with self.subTest(source=source, target=target):
    #             bqm = source(linear, quadratic, offset, vartype)
    #             new = target(bqm, vartype=dimod.BINARY)

    #             self.assertIsInstance(new, target)
    #             assert_consistent_bqm(new)
    #             self.assertEqual(new.vartype, dimod.BINARY)

    #             # change back for equality check
    #             new.change_vartype(dimod.SPIN)
    #             self.assertEqual(bqm.adj, new.adj)
    #             self.assertEqual(bqm.offset, new.offset)
    #             self.assertEqual(bqm.vartype, new.vartype)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_dense_zeros(self, name, BQM):
        # should ignore 0 off-diagonal
        D = np.zeros((5, 5))
        bqm = BQM(D, 'BINARY')
        self.assertEqual(bqm.shape, (5, 0))

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_iterator_2arg(self, name, BQM):
        Q = ((u, v, -1) for u in range(5) for v in range(u+1, 5))
        bqm = BQM(Q, dimod.BINARY)

        self.assertEqual(bqm.shape, (5, 10))

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_iterator_3arg(self, name, BQM):
        h = ((v, 1) for v in range(5))
        J = ((u, v, -1) for u in range(5) for v in range(u+1, 5))
        bqm = BQM(h, J, dimod.SPIN)

        self.assertEqual(bqm.shape, (5, 10))

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_legacy_bqm(self, name, BQM):
    #     lbqm = dimod.BinaryQuadraticModel.from_ising({'a': 2}, {'ab': -1}, 7)

    #     new = BQM(lbqm)

    #     self.assertEqual(lbqm.linear, new.linear)
    #     self.assertEqual(lbqm.adj, new.adj)
    #     self.assertEqual(lbqm.offset, new.offset)
    #     self.assertEqual(lbqm.vartype, new.vartype)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_linear_array_quadratic_array(self, name, BQM):
        h = [1, 2, 3, 4, 5]
        J = np.zeros((5, 5))
        bqm = BQM(h, J, 1.2, 'SPIN')

        self.assertEqual(bqm.linear, {v: v+1 for v in range(5)})
        self.assertEqual(bqm.quadratic, {})
        self.assertEqual(bqm.offset, 1.2)
        self.assertIs(bqm.vartype, dimod.SPIN)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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
        self.assertEqual(bqm.offset, 1.2)
        self.assertIs(bqm.vartype, dimod.SPIN)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_quadratic_only(self, name, BQM):
        M = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        bqm = BQM(M, 'BINARY')
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.linear, {0: 1, 1: 0, 2: 4, 4: 0, 5: 0})
        self.assertEqual(bqm.quadratic, {(0, 1): -1, (1, 2): 1.5, (4, 5): 7})

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_quadratic_only_spin(self, name, BQM):
        M = {(0, 1): -1, (0, 0): 1, (1, 2): 1.5, (2, 2): 4, (4, 5): 7}
        bqm = BQM(M, 'SPIN')
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.linear, {0: 0, 1: 0, 2: 0, 4: 0, 5: 0})
        self.assertEqual(bqm.quadratic, {(0, 1): -1, (1, 2): 1.5, (4, 5): 7})
        self.assertEqual(bqm.offset, 5)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_no_args(self, name, BQM):
        with self.assertRaises(TypeError) as err:
            BQM()
        self.assertEqual(err.exception.args[0],
                         "A valid vartype or another bqm must be provided")

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_numpy_array(self, name, BQM):
        D = np.ones((5, 5))
        bqm = BQM(D, 'BINARY')
        assert_consistent_bqm(bqm)
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_numpy_array_1var(self, name, BQM):
        D = np.ones((1, 1))
        bqm = BQM(D, 'BINARY')
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.shape, (1, 0))
        self.assertEqual(bqm.linear[0], 1)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_vartype_readonly(self, name, BQM):
        bqm = BQM('SPIN')
        with self.assertRaises(AttributeError):
            bqm.vartype = dimod.BINARY


class TestContractVariables(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_binary(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM({'a': 2, 'b': -8}, {('a', 'b'): -2, ('b', 'c'): 1}, 1.2,
                  dimod.BINARY)

        bqm.contract_variables('a', 'b')

        assert_consistent_bqm(bqm)

        target = BQM({'a': -8}, {'ac': 1}, 1.2, dimod.BINARY)

        self.assertEqual(bqm, target)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_spin(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM({'a': 2, 'b': -8}, {('a', 'b'): -2, ('b', 'c'): 1}, 1.2,
                  dimod.SPIN)

        bqm.contract_variables('a', 'b')

        assert_consistent_bqm(bqm)

        target = BQM({'a': -6}, {'ac': 1}, -.8, dimod.SPIN)

        self.assertEqual(bqm, target)


class TestCoo(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_to_coo_string_empty_BINARY(self, name, BQM):
        bqm = BQM.empty(dimod.BINARY)

        bqm_str = bqm.to_coo()

        self.assertIsInstance(bqm_str, str)

        self.assertEqual(bqm_str, '')

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_to_coo_string_empty_SPIN(self, name, BQM):
        bqm = BQM.empty(dimod.SPIN)

        bqm_str = bqm.to_coo()

        self.assertIsInstance(bqm_str, str)

        self.assertEqual(bqm_str, '')

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_to_coo_string_typical_SPIN(self, name, BQM):
        bqm = BQM.from_ising({0: 1.}, {(0, 1): 2, (2, 3): .4})
        s = bqm.to_coo()
        contents = "0 0 1.000000\n0 1 2.000000\n2 3 0.400000"
        self.assertEqual(s, contents)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_to_coo_string_typical_BINARY(self, name, BQM):
        bqm = BQM.from_qubo({(0, 0): 1, (0, 1): 2, (2, 3): .4})
        s = bqm.to_coo()
        contents = "0 0 1.000000\n0 1 2.000000\n2 3 0.400000"
        self.assertEqual(s, contents)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_from_coo_file(self, name, BQM):
        if not BQM.shapeable():
            return

        import os.path as path

        filepath = path.join(path.dirname(path.abspath(__file__)), 'data', 'coo_qubo.qubo')

        with open(filepath, 'r') as fp:
            bqm = BQM.from_coo(fp, dimod.BINARY)

        self.assertEqual(bqm, BQM.from_qubo({(0, 0): -1, (1, 1): -1, (2, 2): -1, (3, 3): -1}))

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_from_coo_string(self, name, BQM):
        if not BQM.shapeable():
            return
        contents = "0 0 1.000000\n0 1 2.000000\n2 3 0.400000"
        bqm = BQM.from_coo(contents, dimod.SPIN)
        self.assertEqual(bqm, BQM.from_ising({0: 1.}, {(0, 1): 2, (2, 3): .4}))

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_coo_functional_file_empty_BINARY(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM.empty(dimod.BINARY)

        tmpdir = tempfile.mkdtemp()
        filename = path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            bqm.to_coo(file)

        with open(filename, 'r') as file:
            new_bqm = BQM.from_coo(file, dimod.BINARY)

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_coo_functional_file_empty_SPIN(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM.empty(dimod.SPIN)

        tmpdir = tempfile.mkdtemp()
        filename = path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            bqm.to_coo(file)

        with open(filename, 'r') as file:
            new_bqm = BQM.from_coo(file, dimod.SPIN)

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_coo_functional_file_BINARY(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.BINARY)

        tmpdir = tempfile.mkdtemp()
        filename = path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            bqm.to_coo(file)

        with open(filename, 'r') as file:
            new_bqm = BQM.from_coo(file, dimod.BINARY)

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_coo_functional_file_SPIN(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.SPIN)

        tmpdir = tempfile.mkdtemp()
        filename = path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            bqm.to_coo(file)

        with open(filename, 'r') as file:
            new_bqm = BQM.from_coo(file, dimod.SPIN)

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_coo_functional_string_empty_BINARY(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM.empty(dimod.BINARY)

        s = bqm.to_coo()
        new_bqm = BQM.from_coo(s, dimod.BINARY)

        self.assertEqual(bqm, new_bqm)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_coo_functional_string_empty_SPIN(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM.empty(dimod.SPIN)

        s = bqm.to_coo()
        new_bqm = BQM.from_coo(s, dimod.SPIN)

        self.assertEqual(bqm, new_bqm)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_coo_functional_string_BINARY(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.BINARY)

        s = bqm.to_coo()
        new_bqm = BQM.from_coo(s, dimod.BINARY)

        self.assertEqual(bqm, new_bqm)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_coo_functional_two_digit_integers_string(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM.from_ising({12: .5, 0: 1}, {(0, 12): .5})

        s = bqm.to_coo()
        new_bqm = BQM.from_coo(s, dimod.SPIN)

        self.assertEqual(bqm, new_bqm)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_coo_functional_string_SPIN(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.SPIN)

        s = bqm.to_coo()
        new_bqm = BQM.from_coo(s, dimod.SPIN)

        self.assertEqual(bqm, new_bqm)


class TestCopy(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_copy(self, name, BQM):
        bqm = BQM({'a': -1, 'b': 1}, {'ab': 2}, 6, dimod.BINARY)
        new = bqm.copy()
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)

        # modify the original and make sure it doesn't propogate
        new.set_linear('a', 1)
        self.assertEqual(new.linear['a'], 1)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_standardlib_copy(self, name, BQM):
        from copy import copy

        bqm = BQM({'a': -1, 'b': 1}, {'ab': 2}, 6, dimod.BINARY)
        new = copy(bqm)
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_standardlib_deepcopy(self, name, BQM):
        from copy import deepcopy

        bqm = BQM({'a': -1, 'b': 1}, {'ab': 2}, 6, dimod.BINARY)
        new = deepcopy(bqm)
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_standardlib_deepcopy_multi(self, name, BQM):
        from copy import deepcopy

        bqm = BQM({'a': -1, 'b': 1}, {'ab': 2}, 6, dimod.BINARY)
        copied = deepcopy([bqm, [bqm]])

        new = copied[0]
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)

        self.assertIs(new, copied[1][0])

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_subclass(self, name, BQM):
        # copy should respect subclassing
        class SubBQM(BQM):
            pass

        bqm = BQM({'a': -1, 'b': 1}, {'ab': 2}, 6, dimod.BINARY)
        new = bqm.copy()
        self.assertIsNot(bqm, new)
        self.assertEqual(type(bqm), type(new))
        self.assertEqual(bqm, new)


class TestEmpty(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_binary(self, name, BQM):
        bqm = BQM.empty(dimod.BINARY)
        self.assertIsInstance(bqm, BQM)
        assert_consistent_bqm(bqm)
        self.assertIs(bqm.vartype, dimod.BINARY)
        self.assertEqual(bqm.shape, (0, 0))

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_spin(self, name, BQM):
        bqm = BQM.empty(dimod.SPIN)
        self.assertIsInstance(bqm, BQM)
        self.assertIs(bqm.vartype, dimod.SPIN)
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.shape, (0, 0))


class TestEnergies(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_2path(self, name, BQM):
        bqm = BQM([.1, -.2], [[0, -1], [0, 0]], 'SPIN')
        samples = [[-1, -1],
                   [-1, +1],
                   [+1, -1],
                   [+1, +1]]

        energies = bqm.energies(np.asarray(samples))

        np.testing.assert_array_almost_equal(energies, [-.9, .7, 1.3, -1.1])

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_5chain(self, name, BQM):
        arr = np.tril(np.triu(np.ones((5, 5)), 1), 1)
        bqm = BQM(arr, 'BINARY')
        samples = [[0, 0, 0, 0, 0]]

        energies = bqm.energies(np.asarray(samples))
        np.testing.assert_array_almost_equal(energies, [0])

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_dtype(self, name, BQM):
        arr = np.arange(9).reshape((3, 3))
        bqm = BQM(arr, dimod.BINARY)

        samples = [[0, 0, 1], [1, 1, 0]]

        energies = bqm.energies(samples, dtype=np.float32)

        self.assertEqual(energies.dtype, np.float32)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_energy(self, name, BQM):
        arr = np.triu(np.ones((5, 5)))
        bqm = BQM(arr, 'BINARY')
        samples = [[0, 0, 1, 0, 0]]

        energy = bqm.energy(np.asarray(samples))
        self.assertEqual(energy, 1)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_label_mismatch(self, name, BQM):
        arr = np.arange(9).reshape((3, 3))
        bqm = BQM(arr, dimod.BINARY)

        samples = ([[0, 0, 1], [1, 1, 0]], 'abc')

        with self.assertRaises(ValueError):
            bqm.energies(samples)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_length(self, name, BQM):
        arr = np.arange(9).reshape((3, 3))
        bqm = BQM(arr, dimod.BINARY)

        samples = [0, 0]

        with self.assertRaises(ValueError):
            bqm.energies(samples)


class TestFileView(unittest.TestCase):
    @parameterized.expand(
        [(cls.__name__, cls) for cls in BQM_CYTHON_SUBCLASSES])
    def test_empty(self, name, BQM):
        bqm = BQM('SPIN')

        with tempfile.TemporaryFile() as tf:
            with bqm.to_file() as bqmf:
                shutil.copyfileobj(bqmf, tf)
            tf.seek(0)
            new = BQM.from_file(tf)

        self.assertEqual(bqm, new)

    @parameterized.expand(
        [(cls.__name__, cls) for cls in BQM_CYTHON_SUBCLASSES])
    def test_2path(self, name, BQM):
        bqm = BQM([.1, -.2], [[0, -1], [0, 0]], 'SPIN')

        with tempfile.TemporaryFile() as tf:
            with bqm.to_file() as bqmf:
                shutil.copyfileobj(bqmf, tf)
            tf.seek(0)
            new = BQM.from_file(tf)

        self.assertEqual(bqm, new)


class TestFixVariable(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_spin(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        bqm.fix_variable('a', +1)
        self.assertEqual(bqm, BQM({'b': -1}, {}, 1.5, dimod.SPIN))

        bqm = BQM({'a': .5}, {('a', 'b'): -1}, 1.5, dimod.SPIN)
        bqm.fix_variable('a', -1)
        self.assertEqual(bqm, BQM({'b': +1}, {}, 1, dimod.SPIN))

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_binary(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        bqm.fix_variable('a', 1)
        self.assertEqual(bqm, BQM({'b': -1}, {}, 1.5, dimod.BINARY))

        bqm = BQM({'a': .5}, {('a', 'b'): -1}, 1.5, dimod.BINARY)
        bqm.fix_variable('a', 0)
        self.assertEqual(bqm, BQM({'b': 0}, {}, 1.5, dimod.BINARY))

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_cross_type(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        with self.assertRaises(ValueError):
            bqm.fix_variable('a', -1)

        bqm = BQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        with self.assertRaises(ValueError):
            bqm.fix_variable('a', 0)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_missing_variable(self, name, BQM):
        if not BQM.shapeable():
            return
        with self.assertRaises(ValueError):
            BQM('SPIN').fix_variable('a', -1)


class TestFixVariables(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_typical(self, name, BQM):
        if not BQM.shapeable():
            return

        bqm = BQM({'a': -1, 'b': 1, 'c': 3}, {}, dimod.SPIN)

        bqm.fix_variables({'a': 1, 'b': -1})

        self.assertEqual(bqm.linear, {'c': 3})
        self.assertEqual(bqm.quadratic, {})
        self.assertEqual(bqm.offset, -2)


# class TestFlipVariable(unittest.TestCase):
#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_binary(self, name, BQM):
#         bqm = BQM({'a': -1, 'b': 1}, {'ab': -1}, 0, dimod.BINARY)
#         bqm.flip_variable('a')
#         self.assertEqual(bqm, BQM({'a': 1}, {'ab': 1}, -1.0, dimod.BINARY))

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_spin(self, name, BQM):
#         bqm = BQM({'a': -1, 'b': 1}, {'ab': -1}, 1.0, dimod.SPIN)
#         bqm.flip_variable('a')
#         self.assertEqual(bqm, BQM({'a': 1, 'b': 1}, {'ab': 1}, 1.0, dimod.SPIN))


class TestFromNumpyVectors(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_5var_labels(self, _, BQM):
        bqm = BQM.from_numpy_vectors(range(5), ([], [], []), .5, 'SPIN',
                                     variable_order='abcde')
        self.assertEqual(list(bqm.variables), list('abcde'))

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_dtypes(self, _, BQM):
        # we don't test uint64 because when combined with int it gets promoted
        # to float
        integral = [np.uint8, np.uint16, np.uint32,
                    np.int8, np.int16, np.int32, np.int64]
        numeric = integral + [np.float32, np.float64]

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

        with self.subTest('uint64 combos'):
            bqm = BQM.from_numpy_vectors(
                np.asarray(h, dtype=np.uint64),
                (np.asarray(heads, dtype=np.uint64),
                 np.asarray(tails, dtype=np.uint64),
                 np.asarray(values, dtype=np.uint64)),
                0.0, 'SPIN')

            self.assertEqual(bqm.linear, {0: 1, 1: 2, 2: 3})
            self.assertEqual(bqm.adj, {0: {1: 4}, 1: {0: 4, 2: 5}, 2: {1: 5}})

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_empty(self, _, BQM):
        bqm = BQM.from_numpy_vectors([], ([], [], []), 1.5, 'SPIN')
        self.assertEqual(bqm.shape, (0, 0))
        self.assertEqual(bqm.offset, 1.5)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_noncontiguous(self, _, BQM):
        quadratic = np.asarray([[0, 1], [1, 2]])

        bqm = BQM.from_numpy_vectors(
            [], (quadratic[:, 0], quadratic[:, 1], [.5, .6]), 1.5, 'SPIN')

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_oversized_linear(self, _, BQM):
        bqm = BQM.from_numpy_vectors([0, 1, 2], ([], [], []), 1.5, 'SPIN')
        self.assertEqual(bqm.shape, (3, 0))
        self.assertEqual(bqm.linear, {0: 0, 1: 1, 2: 2})
        self.assertEqual(bqm.offset, 1.5)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_undersized_linear(self, _, BQM):
        bqm = BQM.from_numpy_vectors([0, 1], ([3], [4], [1]), 1.5, 'SPIN')
        self.assertEqual(bqm.shape, (5, 1))
        self.assertEqual(bqm.linear, {0: 0, 1: 1, 2: 0, 3: 0, 4: 0})
        self.assertEqual(bqm.offset, 1.5)
        self.assertEqual(bqm.adj, {0: {}, 1: {}, 2: {}, 3: {4: 1}, 4: {3: 1}})


class TestFromQUBO(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_basic(self, name, BQM):
        Q = {(0, 0): -1, (0, 1): -1, (0, 2): -1, (1, 2): 1}
        bqm = BQM.from_qubo(Q)

        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.linear, {0: -1, 1: 0, 2: 0})
        self.assertEqual(bqm.adj, {0: {1: -1, 2: -1},
                                   1: {0: -1, 2: 1},
                                   2: {0: -1, 1: 1}})
        self.assertEqual(bqm.offset, 0)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_with_offset(self, name, BQM):
        Q = {(0, 0): -1, (0, 1): -1, (0, 2): -1, (1, 2): 1}
        bqm = BQM.from_qubo(Q, 1.6)

        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.linear, {0: -1, 1: 0, 2: 0})
        self.assertEqual(bqm.adj, {0: {1: -1, 2: -1},
                                   1: {0: -1, 2: 1},
                                   2: {0: -1, 1: 1}})
        self.assertEqual(bqm.offset, 1.6)


class TestGetLinear(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_disconnected_string_labels(self, name, BQM):
        bqm = BQM({'a': -1, 'b': 1}, {}, dimod.BINARY)
        self.assertEqual(bqm.get_linear('a'), -1)
        self.assertEqual(bqm.get_linear('b'), 1)
        with self.assertRaises(ValueError):
            bqm.get_linear('c')

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_disconnected(self, name, BQM):
        bqm = BQM(5, dimod.SPIN)

        for v in range(5):
            self.assertEqual(bqm.get_linear(v), 0)

        with self.assertRaises(ValueError):
            bqm.get_linear(-1)

        with self.assertRaises(ValueError):
            bqm.get_linear(5)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_dtype(self, name, BQM):
        bqm = BQM(5, dimod.SPIN)

        # np.object_ does not play very nicely, even if it's accurate
        dtype = object if bqm.dtype.type is np.object_ else bqm.dtype.type

        for v in range(5):
            self.assertIsInstance(bqm.get_linear(v), dtype)


class TestGetQuadratic(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_default(self, name, BQM):
        bqm = BQM(5, 'SPIN')  # has no interactions
        with self.assertRaises(ValueError):
            bqm.get_quadratic(0, 1)
        self.assertEqual(bqm.get_quadratic(0, 1, default=5), 5)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_dtype(self, name, BQM):
        bqm = BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]], dimod.SPIN)

        # np.object_ does not play very nicely, even if it's accurate
        dtype = object if bqm.dtype.type is np.object_ else bqm.dtype.type

        self.assertIsInstance(bqm.get_quadratic(0, 1), dtype)
        self.assertIsInstance(bqm.get_quadratic(1, 0), dtype)

        self.assertIsInstance(bqm.get_quadratic(0, 2), dtype)
        self.assertIsInstance(bqm.get_quadratic(2, 0), dtype)


class TestHasVariable(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_typical(self, name, BQM):
        h = OrderedDict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = BQM(h, J, dimod.SPIN)

        self.assertTrue(bqm.has_variable('a'))
        self.assertTrue(bqm.has_variable(1))
        self.assertTrue(bqm.has_variable(3))

        # no false positives
        self.assertFalse(bqm.has_variable(0))
        self.assertFalse(bqm.has_variable(2))


class TestIteration(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_iter_quadratic_neighbours(self, name, BQM):
        bqm = BQM({'ab': -1, 'bc': 21, 'cd': 1}, dimod.SPIN)
        neighbours = set(bqm.iter_quadratic('b'))
        self.assertEqual(neighbours,
                         {('b', 'a', -1), ('b', 'c', 21)})

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_iter_quadratic_neighbours_bunch(self, name, BQM):
        bqm = BQM({'bc': 21, 'cd': 1}, dimod.SPIN)
        self.assertEqual(list(bqm.iter_quadratic(['b', 'c'])),
                         [('b', 'c', 21.0), ('c', 'd', 1.0)])

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_iter_variables(self, name, BQM):
        h = OrderedDict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = BQM(h, J, dimod.SPIN)

        self.assertEqual(list(bqm.iter_variables()), ['a', 1, 3])


class TestLen(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test__len__(self, name, BQM):
        bqm = BQM(np.ones((107, 107)), dimod.BINARY)
        self.assertEqual(len(bqm), 107)


class TestNetworkxGraph(unittest.TestCase):
    # developer note: these tests should be moved to converter tests when
    # the methods are deprecated.

    def setUp(self):
        try:
            import networkx as nx
        except ImportError:
            raise unittest.SkipTest("NetworkX is not installed")

    def test_empty(self):
        import networkx as nx
        G = nx.Graph()
        G.vartype = 'SPIN'
        bqm = dimod.BinaryQuadraticModel.from_networkx_graph(G)
        self.assertEqual(len(bqm), 0)
        self.assertIs(bqm.vartype, dimod.SPIN)

    def test_no_biases(self):
        import networkx as nx
        G = nx.complete_graph(5)
        G.vartype = 'BINARY'
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
        BQM = model.to_networkx_graph()

        self.assertEqual(set(graph), set(BQM))
        for u, v in graph.edges:
            self.assertIn(u, BQM[v])

        for v, bias in model.linear.items():
            self.assertEqual(bias, BQM.nodes[v]['bias'])


class TestNumpyMatrix(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_to_numpy_matrix(self, name, BQM):
        # integer-indexed, binary bqm
        linear = {v: v * .01 for v in range(10)}
        quadratic = {(v, u): u * v * .01 for u, v in itertools.combinations(linear, 2)}
        quadratic[(0, 1)] = quadratic[(1, 0)]
        del quadratic[(1, 0)]
        offset = 1.2
        vartype = dimod.BINARY
        bqm = BQM(linear, quadratic, offset, vartype)

        M = bqm.to_numpy_matrix()

        self.assertTrue(np.array_equal(M, np.triu(M)))  # upper triangular

        for (row, col), bias in np.ndenumerate(M):
            if row == col:
                self.assertEqual(bias, linear[row])
            else:
                self.assertTrue((row, col) in quadratic or (col, row) in quadratic)
                self.assertFalse((row, col) in quadratic and (col, row) in quadratic)

                if row > col:
                    self.assertEqual(bias, 0)
                else:
                    if (row, col) in quadratic:
                        self.assertEqual(quadratic[(row, col)], bias)
                    else:
                        self.assertEqual(quadratic[(col, row)], bias)

        #

        # integer-indexed, not contiguous
        bqm = BQM({}, {(0, 3): -1}, 0.0, dimod.BINARY)

        with self.assertRaises(ValueError):
            M = bqm.to_numpy_matrix()

        #

        # string-labeled, variable_order provided
        linear = {'a': -1}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3}
        bqm = BQM(linear, quadratic, 0.0, dimod.BINARY)

        with self.assertRaises(ValueError):
            bqm.to_numpy_matrix(['a', 'c'])  # incomplete variable order

        M = bqm.to_numpy_matrix(['a', 'c', 'b'])

        self.assertTrue(np.array_equal(M, [[-1., 1.2, 0.], [0., 0., 0.3], [0., 0., 0.]]))

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_functional(self, name, BQM):
        bqm = BQM({'a': -1}, {'ac': 1.2, 'bc': .3}, dimod.BINARY)

        order = ['a', 'b', 'c']

        M = bqm.to_numpy_matrix(variable_order=order)

        new = BQM.from_numpy_matrix(M, variable_order=order)

        assert_consistent_bqm(new)
        self.assertEqual(bqm, new)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_from_numpy_matrix(self, name, BQM):

        linear = {'a': -1}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3}
        bqm = BQM(linear, quadratic, 0.0, dimod.BINARY)

        variable_order = ['a', 'c', 'b']

        M = bqm.to_numpy_matrix(variable_order=variable_order)

        new_bqm = BQM.from_numpy_matrix(M, variable_order=variable_order)

        self.assertEqual(bqm, new_bqm)

        #

        if not BQM.shapeable():
            # this part only applies to shapeable
            return

        # zero-interactions get ignored unless provided in interactions
        linear = {'a': -1}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3, ('a', 'b'): 0}
        bqm = BQM(linear, quadratic, 0.0, dimod.BINARY)
        variable_order = ['a', 'c', 'b']
        M = bqm.to_numpy_matrix(variable_order=variable_order)

        new_bqm = BQM.from_numpy_matrix(M, variable_order=variable_order)

        self.assertNotIn(('a', 'b'), new_bqm.quadratic)
        self.assertNotIn(('b', 'a'), new_bqm.quadratic)

        new_bqm = BQM.from_numpy_matrix(M, variable_order=variable_order, interactions=quadratic)

        self.assertEqual(bqm, new_bqm)

        #

        M = np.asarray([[0, 1], [0, 0]])
        bqm = BQM.from_numpy_matrix(M)
        self.assertEqual(bqm, BQM({0: 0, 1: 0}, {(0, 1): 1}, 0, dimod.BINARY))


class TestNormalize(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_normalize(self, name, BQM):
        bqm = BQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.normalize(.5)
        self.assertAlmostEqual(bqm.linear, {0: -.5, 1: .5})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -.25})
        self.assertAlmostEqual(bqm.offset, .25)
        assert_consistent_bqm(bqm)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_return_value(self, name, BQM):
        bqm = BQM({0: 2}, {(0, 1): 2}, 'SPIN')

        self.assertEqual(bqm.normalize([-1, 1]), .5)


class TestOffset(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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


# class TestPickle(unittest.TestCase):
#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_picklable(self, name, BQM):
#         import pickle

#         bqm = BQM({'a': -1, 'b': 1}, {'ab': 2}, 6, dimod.BINARY)
#         new = pickle.loads(pickle.dumps(bqm))
#         self.assertIs(type(bqm), type(new))
#         self.assertEqual(bqm, new)


class TestRemoveInteraction(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_basic(self, name, BQM):
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


class TestRemoveInteractionsFrom(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_basic(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest
        bqm = BQM(np.triu(np.ones((3, 3))), dimod.BINARY)

        bqm.remove_interactions_from([(0, 2), (2, 1)])

        self.assertEqual(bqm.num_interactions, 1)


class TestRemoveVariable(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_labelled(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_provided(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest
        bqm = BQM('SPIN')
        bqm.add_variable('a')
        bqm.add_variable('b')
        bqm.add_variable('c')

        bqm.remove_variable('b')
        assert_consistent_bqm(bqm)

        # maintained order
        self.assertEqual(list(bqm.iter_variables()), ['a', 'c'])

        with self.assertRaises(ValueError):
            bqm.remove_variable('b')

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_unlabelled(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest
        bqm = BQM(2, dimod.BINARY)
        self.assertEqual(bqm.remove_variable(), 1)
        assert_consistent_bqm(bqm)
        self.assertEqual(bqm.remove_variable(), 0)
        assert_consistent_bqm(bqm)
        with self.assertRaises(ValueError):
            bqm.remove_variable()
        assert_consistent_bqm(bqm)


class TestRelabel(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

        self.assertEqual(new, test)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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
        self.assertEqual(test, bqm)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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
        self.assertEqual(old, bqm)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

        self.assertEqual(newlinear, newmodel.linear)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

        self.assertEqual(newlinear, bqm.linear)


class TestScale(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_basic(self, name, BQM):
        # does not change shape
        bqm = BQM(np.triu(np.ones((3, 3))), dimod.BINARY)

        self.assertEqual(bqm.get_linear(0), 1)
        bqm.set_linear(0, .5)
        self.assertEqual(bqm.get_linear(0), .5)


class TestSetQuadratic(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_set_quadratic_exception(self, name, BQM):
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


class TestShape(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_3x3array(self, name, BQM):
        bqm = BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]], dimod.BINARY)

        self.assertEqual(bqm.shape, (3, 2))
        self.assertEqual(bqm.num_variables, 3)
        self.assertEqual(bqm.num_interactions, 2)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_disconnected(self, name, BQM):
        bqm = BQM(5, dimod.BINARY)

        self.assertEqual(bqm.shape, (5, 0))
        self.assertEqual(bqm.num_variables, 5)
        self.assertEqual(bqm.num_interactions, 0)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_empty(self, name, BQM):
        self.assertEqual(BQM(dimod.SPIN).shape, (0, 0))
        self.assertEqual(BQM(0, dimod.SPIN).shape, (0, 0))

        self.assertEqual(BQM(dimod.SPIN).num_variables, 0)
        self.assertEqual(BQM(0, dimod.SPIN).num_variables, 0)

        self.assertEqual(BQM(dimod.SPIN).num_interactions, 0)
        self.assertEqual(BQM(0, dimod.SPIN).num_interactions, 0)


# class TestToIsing(unittest.TestCase):
#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_spin(self, name, BQM):
#         linear = {0: 7.1, 1: 103}
#         quadratic = {(0, 1): .97}
#         offset = 0.3
#         vartype = dimod.SPIN

#         model = BQM(linear, quadratic, offset, vartype)

#         h, J, off = model.to_ising()

#         self.assertEqual(off, offset)
#         self.assertEqual(linear, h)
#         self.assertEqual(quadratic, J)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_to_ising_binary_to_ising(self, name, BQM):
#         linear = {0: 7.1, 1: 103}
#         quadratic = {(0, 1): .97}
#         offset = 0.3
#         vartype = dimod.BINARY

#         model = BQM(linear, quadratic, offset, vartype)

#         h, J, off = model.to_ising()

#         for spins in itertools.product((-1, 1), repeat=len(model)):
#             spin_sample = dict(zip(range(len(spins)), spins))
#             bin_sample = {v: (s + 1) // 2 for v, s in spin_sample.items()}

#             # calculate the qubo's energy
#             energy = off
#             for (u, v), bias in J.items():
#                 energy += spin_sample[u] * spin_sample[v] * bias
#             for v, bias in h.items():
#                 energy += spin_sample[v] * bias

#             # and the energy of the model
#             self.assertAlmostEqual(energy, model.energy(bin_sample))


# class TestVartypeViews(unittest.TestCase):
#     # SpinView and BinaryView


#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_add_offset_binary(self, name, BQM):
#         bqm = BQM({'a': -1}, {'ab': 2}, 1.5, dimod.SPIN)

#         bqm.binary.add_offset(2)
#         self.assertEqual(bqm.offset, 3.5)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_add_offset_spin(self, name, BQM):
#         bqm = BQM({'a': -1}, {'ab': 2}, 1.5, dimod.BINARY)

#         bqm.spin.add_offset(2)
#         self.assertEqual(bqm.offset, 3.5)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_binary_binary(self, name, BQM):
#         bqm = BQM(dimod.BINARY)
#         self.assertIs(bqm.binary, bqm)
#         self.assertIs(bqm.binary.binary, bqm)  # and so on

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_spin_spin(self, name, BQM):
#         bqm = BQM(dimod.SPIN)
#         self.assertIs(bqm.spin, bqm)
#         self.assertIs(bqm.spin.spin, bqm)  # and so on

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_simple_binary(self, name, BQM):
#         bqm = BQM({'a': 1, 'b': -3, 'c': 2}, {'ab': -5, 'bc': 6}, 16, 'SPIN')

#         assert_consistent_bqm(bqm.binary)
#         self.assertIs(bqm.binary.vartype, dimod.BINARY)
#         binary = bqm.change_vartype(dimod.BINARY, inplace=False)
#         self.assertEqual(binary, bqm.binary)
#         self.assertNotEqual(binary, bqm)
#         self.assertIs(bqm.binary.spin, bqm)
#         self.assertIs(bqm.binary.binary, bqm.binary)  # and so on

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_simple_spin(self, name, BQM):
#         bqm = BQM({'a': 1, 'b': -3, 'c': 2}, {'ab': -5, 'bc': 6}, 16, 'BINARY')

#         assert_consistent_bqm(bqm.spin)
#         self.assertIs(bqm.spin.vartype, dimod.SPIN)
#         spin = bqm.change_vartype(dimod.SPIN, inplace=False)
#         self.assertEqual(spin, bqm.spin)
#         self.assertNotEqual(spin, bqm)
#         self.assertIs(bqm.spin.binary, bqm)
#         self.assertIs(bqm.spin.spin, bqm.spin)  # and so on

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_copy_binary(self, name, BQM):
#         bqm = BQM({'a': 1, 'b': -3, 'c': 2}, {'ab': -5, 'bc': 6}, 16, 'SPIN')
#         new = bqm.binary.copy()
#         self.assertIsNot(new, bqm.binary)
#         self.assertIsInstance(new, BQM)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_copy_spin(self, name, BQM):
#         bqm = BQM({'a': 1, 'b': -3, 'c': 2}, {'ab': -5, 'bc': 6}, 16, 'BINARY')
#         new = bqm.spin.copy()
#         self.assertIsNot(new, bqm.spin)
#         self.assertIsInstance(new, BQM)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_offset_binary(self, name, BQM):
#         bqm = BQM({'a': 1}, {'ab': 2}, 3, dimod.SPIN)

#         bqm.binary.offset -= 2
#         self.assertEqual(bqm.offset, 1)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_offset_spin(self, name, BQM):
#         bqm = BQM({'a': 1}, {'ab': 2}, 3, dimod.BINARY)

#         bqm.spin.offset -= 2
#         self.assertEqual(bqm.offset, 1)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_set_linear_binary(self, name, BQM):
#         bqm = BQM({0: 1, 1: -3, 2: 2}, {(0, 1): -5, (1, 2): 6}, 16, dimod.SPIN)

#         view = bqm.binary
#         copy = bqm.change_vartype(dimod.BINARY, inplace=False)

#         view.set_linear(0, .5)
#         copy.set_linear(0, .5)

#         self.assertEqual(view.get_linear(0), .5)
#         self.assertEqual(copy.get_linear(0), .5)

#         self.assertEqual(view.spin, copy.spin)
#         self.assertEqual(view.binary, copy.binary)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_set_linear_spin(self, name, BQM):
#         bqm = BQM({0: 1, 1: -3, 2: 2}, {(0, 1): -5, (1, 2): 6}, 16, dimod.BINARY)

#         view = bqm.spin
#         copy = bqm.change_vartype(dimod.SPIN, inplace=False)

#         view.set_linear(0, .5)
#         copy.set_linear(0, .5)

#         self.assertEqual(view.get_linear(0), .5)
#         self.assertEqual(copy.get_linear(0), .5)

#         self.assertEqual(view.spin, copy.spin)
#         self.assertEqual(view.binary, copy.binary)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_set_offset_binary(self, name, BQM):
#         bqm = BQM({0: 1, 1: -3, 2: 2}, {(0, 1): -5, (1, 2): 6}, 16, dimod.SPIN)

#         view = bqm.binary
#         copy = bqm.change_vartype(dimod.BINARY, inplace=False)

#         view.offset = .5
#         copy.offset = .5

#         self.assertEqual(view.offset, .5)
#         self.assertEqual(copy.offset, .5)

#         self.assertEqual(view.spin, copy.spin)
#         self.assertEqual(view.binary, copy.binary)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_set_offset_spin(self, name, BQM):
#         bqm = BQM({0: 1, 1: -3, 2: 2}, {(0, 1): -5, (1, 2): 6}, 16, dimod.BINARY)

#         view = bqm.spin
#         copy = bqm.change_vartype(dimod.SPIN, inplace=False)

#         view.offset = .5
#         copy.offset = .5

#         self.assertEqual(view.offset, .5)
#         self.assertEqual(copy.offset, .5)

#         self.assertEqual(view.spin, copy.spin)
#         self.assertEqual(view.binary, copy.binary)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_set_quadratic_binary(self, name, BQM):
#         bqm = BQM({0: 1, 1: -3, 2: 2}, {(0, 1): -5, (1, 2): 6}, 16, dimod.SPIN)

#         view = bqm.binary
#         copy = bqm.change_vartype(dimod.BINARY, inplace=False)

#         view.set_quadratic(0, 1, -1)
#         copy.set_quadratic(0, 1, -1)

#         self.assertEqual(view.get_quadratic(0, 1), -1)
#         self.assertEqual(copy.get_quadratic(0, 1), -1)

#         self.assertEqual(view.spin, copy.spin)
#         self.assertEqual(view.binary, copy.binary)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_set_quadratic_spin(self, name, BQM):
#         bqm = BQM({0: 1, 1: -3, 2: 2}, {(0, 1): -5, (1, 2): 6}, 16, dimod.BINARY)

#         view = bqm.spin
#         copy = bqm.change_vartype(dimod.SPIN, inplace=False)

#         view.set_quadratic(0, 1, -1)
#         copy.set_quadratic(0, 1, -1)

#         self.assertEqual(view.get_quadratic(0, 1), -1)
#         self.assertEqual(copy.get_quadratic(0, 1), -1)

#         self.assertEqual(view.spin, copy.spin)
#         self.assertEqual(view.binary, copy.binary)

#     @parameterized.expand([(cls.__name__, cls, inplace)
#                            for (cls, inplace)
#                            in itertools.product(BQM_SUBCLASSES, [False, True])])
#     def test_relabel_variables_binary(self, name, BQM, inplace):
#         # to get a BinaryView, construct in SPIN, and ask for binary
#         linear = {0: 1, 1: -3, 2: 2}
#         quadratic = {(0, 1): -5, (1, 2): 6}
#         offset = 16
#         vartype = dimod.SPIN
#         view = BQM(linear, quadratic, offset, vartype).binary

#         # relabel view
#         mapping = {0: 'a', 1: 'b', 2: 'c'}
#         new = view.relabel_variables(mapping, inplace=inplace)
#         assert_consistent_bqm(new)
#         if inplace:
#             self.assertIs(view, new)
#         else:
#             self.assertIsNot(view, new)

#         # check that new model is correct
#         linear = {'a': 1, 'b': -3, 'c': 2}
#         quadratic = {'ab': -5, 'bc': 6}
#         offset = 16
#         vartype = dimod.SPIN
#         test = BQM(linear, quadratic, offset, vartype).binary
#         self.assertEqual(new, test)

#     @parameterized.expand([(cls.__name__, cls, inplace)
#                            for (cls, inplace)
#                            in itertools.product(BQM_SUBCLASSES, [False, True])])
#     def test_relabel_variables_spin(self, name, BQM, inplace):
#         # to get a SpinView, construct in BINARY, and ask for spin
#         linear = {0: 1, 1: -3, 2: 2}
#         quadratic = {(0, 1): -5, (1, 2): 6}
#         offset = 16
#         vartype = dimod.BINARY
#         view = BQM(linear, quadratic, offset, vartype).spin

#         # relabel view
#         mapping = {0: 'a', 1: 'b', 2: 'c'}
#         new = view.relabel_variables(mapping, inplace=inplace)
#         assert_consistent_bqm(new)
#         if inplace:
#             self.assertIs(view, new)
#         else:
#             self.assertIsNot(view, new)

#         # check that new model is correct
#         linear = {'a': 1, 'b': -3, 'c': 2}
#         quadratic = {'ab': -5, 'bc': 6}
#         offset = 16
#         vartype = dimod.BINARY
#         test = BQM(linear, quadratic, offset, vartype).spin
#         self.assertEqual(new, test)


class TestToNumpyVectors(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_array_dense(self, name, BQM):
        bqm = BQM(np.arange(9).reshape((3, 3)), 'BINARY')

        ldata, (irow, icol, qdata), off = bqm.to_numpy_vectors()

        np.testing.assert_array_equal(ldata, [0, 4, 8])

        self.assertEqual(len(irow), len(icol))
        self.assertEqual(len(icol), len(qdata))
        self.assertEqual(len(qdata), len(bqm.quadratic))
        for u, v, bias in zip(irow, icol, qdata):
            self.assertAlmostEqual(bqm.adj[u][v], bias)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_empty(self, name, BQM):
        bqm = BQM('SPIN')
        h, (i, j, values), off = bqm.to_numpy_vectors()

        np.testing.assert_array_equal(h, [])
        np.testing.assert_array_equal(i, [])
        np.testing.assert_array_equal(j, [])
        np.testing.assert_array_equal(values, [])
        self.assertEqual(off, 0)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_sort_indices(self, name, BQM):
        bqm = BQM.from_ising({}, {(0, 1): .5, (3, 2): -1, (0, 3): 1.5})

        h, (i, j, values), off = bqm.to_numpy_vectors(sort_indices=True)

        np.testing.assert_array_equal(h, [0, 0, 0, 0])
        np.testing.assert_array_equal(i, [0, 0, 2])
        np.testing.assert_array_equal(j, [1, 3, 3])
        np.testing.assert_array_equal(values, [.5, 1.5, -1])


# class TestToQUBO(unittest.TestCase):
#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_binary(self, name, BQM):
#         linear = {0: 0, 1: 0}
#         quadratic = {(0, 1): 1}
#         offset = 0.0
#         vartype = dimod.BINARY

#         model = BQM(linear, quadratic, offset, vartype)

#         Q, off = model.to_qubo()

#         self.assertEqual(off, offset)
#         self.assertEqual({(0, 0): 0, (1, 1): 0, (0, 1): 1}, Q)

#     @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
#     def test_spin(self, name, BQM):
#         linear = {0: .5, 1: 1.3}
#         quadratic = {(0, 1): -.435}
#         offset = 1.2
#         vartype = dimod.SPIN

#         model = BQM(linear, quadratic, offset, vartype)

#         Q, off = model.to_qubo()

#         for spins in itertools.product((-1, 1), repeat=len(model)):
#             spin_sample = dict(zip(range(len(spins)), spins))
#             bin_sample = {v: (s + 1) // 2 for v, s in spin_sample.items()}

#             # calculate the qubo's energy
#             energy = off
#             for (u, v), bias in Q.items():
#                 energy += bin_sample[u] * bin_sample[v] * bias

#             # and the energy of the model
#             self.assertAlmostEqual(energy, model.energy(spin_sample))


class TestUpdate(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_cross_vartype(self, name, BQM):
        if not BQM.shapeable():
            return
        binary = BQM({'a': .3}, {('a', 'b'): -1}, 0, dimod.BINARY)
        spin = BQM({'c': -1}, {('b', 'c'): 1}, 1.2, dimod.SPIN)

        binary.update(spin)

        target = BQM({'a': .3, 'b': -2, 'c': -4}, {'ab': -1, 'cb': 4},
                     3.2, dimod.BINARY)

        print(binary, target)

        self.assertEqual(binary, target)


class TestViews(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_adj_setitem(self, name, BQM):
        bqm = BQM({'ab': -1}, 'SPIN')
        bqm.adj['a']['b'] = 5
        self.assertEqual(bqm.adj['a']['b'], 5)
        assert_consistent_bqm(bqm)  # all the other cases

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_adj_neighborhoods(self, name, BQM):
        bqm = BQM({'ab': -1, 'ac': -1, 'bc': -1, 'cd': -1}, 'SPIN')

        self.assertEqual(len(bqm.adj['a']), 2)
        self.assertEqual(len(bqm.adj['b']), 2)
        self.assertEqual(len(bqm.adj['c']), 3)
        self.assertEqual(len(bqm.adj['d']), 1)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_linear_delitem(self, name, BQM):
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

        assert_consistent_bqm(bqm)

        with self.assertRaises(KeyError):
            del bqm.linear[2]

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_linear_setitem(self, name, BQM):
        bqm = BQM({'ab': -1}, dimod.SPIN)
        bqm.linear['a'] = 5
        self.assertEqual(bqm.get_linear('a'), 5)
        assert_consistent_bqm(bqm)

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_linear_sum(self, name, BQM):
    #     bqm = BQM.from_ising({'a': -1, 'b': 2}, {'ab': 1, 'bc': 1})
    #     self.assertEqual(bqm.linear.sum(), 1)
    #     self.assertEqual(bqm.linear.sum(start=5), 6)

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_neighborhood_max(self, name, BQM):
    #     bqm = BQM.from_ising({}, {'ab': 1, 'ac': 2, 'bc': 3})
    #     self.assertEqual(bqm.adj['a'].max(), 2)
    #     self.assertEqual(bqm.adj['b'].max(), 3)
    #     self.assertEqual(bqm.adj['c'].max(), 3)

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_neighborhood_max_empty(self, name, BQM):
    #     bqm = BQM.from_ising({'a': 1}, {})

    #     with self.assertRaises(ValueError):
    #         bqm.adj['a'].max()

    #     self.assertEqual(bqm.adj['a'].max(default=5), 5)

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_CYTHON_SUBCLASSES])
    # def test_neighborhood_max_cybqm(self, name, BQM):
    #     bqm = BQM.from_ising({}, {'ab': 1, 'ac': 2, 'bc': 3})

    #     def _max(*args, **kwargs):
    #         raise Exception('boom')

    #     with unittest.mock.patch('builtins.max', _max):
    #         bqm.adj['a'].max()

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_neighborhood_min(self, name, BQM):
    #     bqm = BQM.from_ising({}, {'ab': -1, 'ac': -2, 'bc': -3})
    #     self.assertEqual(bqm.adj['a'].min(), -2)
    #     self.assertEqual(bqm.adj['b'].min(), -3)
    #     self.assertEqual(bqm.adj['c'].min(), -3)

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_neighborhood_min_empty(self, name, BQM):
    #     bqm = BQM.from_ising({'a': 1}, {})

    #     with self.assertRaises(ValueError):
    #         bqm.adj['a'].min()

    #     self.assertEqual(bqm.adj['a'].min(default=5), 5)

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_CYTHON_SUBCLASSES])
    # def test_neighborhood_min_cybqm(self, name, BQM):
    #     bqm = BQM.from_ising({}, {'ab': 1, 'ac': 2, 'bc': 3})

    #     def _min(*args, **kwargs):
    #         raise Exception('boom')

    #     with unittest.mock.patch('builtins.min', _min):
    #         bqm.adj['a'].min()

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_neighborhood_sum(self, name, BQM):
    #     bqm = BQM.from_ising({}, {'ab': -1, 'ac': -2, 'bc': -3})
    #     self.assertEqual(bqm.adj['a'].sum(), -3)
    #     self.assertEqual(bqm.adj['b'].sum(), -4)
    #     self.assertEqual(bqm.adj['c'].sum(), -5)

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_neighborhood_sum_empty(self, name, BQM):
    #     bqm = BQM.from_ising({'a': 1}, {})
    #     self.assertEqual(bqm.adj['a'].sum(), 0)
    #     self.assertEqual(bqm.adj['a'].sum(start=5), 5)

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_CYTHON_SUBCLASSES])
    # def test_neighborhood_sum_cybqm(self, name, BQM):
    #     bqm = BQM.from_ising({}, {'ab': 1, 'ac': 2, 'bc': 3})

    #     def _sum(*args, **kwargs):
    #         raise Exception('boom')

    #     with unittest.mock.patch('builtins.sum', _sum):
    #         bqm.adj['a'].sum()

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_CYTHON_SUBCLASSES])
    # def test_quadratic_sum_cybqm(self, name, BQM):
    #     # make sure it doesn't use python's sum
    #     bqm = BQM.from_ising({'a': -1, 'b': 2}, {'ab': -1, 'bc': 6})

    #     def _sum(*args, **kwargs):
    #         raise Exception('boom')

    #     with unittest.mock.patch('builtins.sum', _sum):
    #         bqm.linear.sum()

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_quadratic_delitem(self, name, BQM):
        if not BQM.shapeable():
            raise unittest.SkipTest
        bqm = BQM([[0, 1, 2, 3, 4],
                   [0, 6, 7, 8, 9],
                   [0, 0, 10, 11, 12],
                   [0, 0, 0, 13, 14],
                   [0, 0, 0, 0, 15]], 'SPIN')
        del bqm.quadratic[0, 1]
        self.assertEqual(set(bqm.iter_neighbors(0)), set([2, 3, 4]))
        assert_consistent_bqm(bqm)

        with self.assertRaises(KeyError):
            del bqm.quadratic[0, 1]

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_quadratic_setitem(self, name, BQM):
        bqm = BQM({'ab': -1}, dimod.SPIN)
        bqm.quadratic[('a', 'b')] = 5
        self.assertEqual(bqm.get_quadratic('a', 'b'), 5)
        assert_consistent_bqm(bqm)

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_quadratic_sum(self, name, BQM):
    #     bqm = BQM.from_ising({'a': -1, 'b': 2}, {'ab': -1, 'bc': 6})
    #     self.assertEqual(bqm.quadratic.sum(), 5)
    #     self.assertEqual(bqm.quadratic.sum(start=5), 10)

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_CYTHON_SUBCLASSES])
    # def test_quadratic_sum_cybqm(self, name, BQM):
    #     # make sure it doesn't use python's sum
    #     bqm = BQM.from_ising({'a': -1, 'b': 2}, {'ab': -1, 'bc': 6})

    #     def _sum(*args, **kwargs):
    #         raise Exception('boom')

    #     with unittest.mock.patch('builtins.sum', _sum):
    #         bqm.quadratic.sum()

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_lin_minmax(self, name, BQM):
    #     num_vars = 10
    #     D = np.arange(num_vars*num_vars).reshape((num_vars, num_vars))
    #     bqm = BQM(D, 'SPIN') 

    #     lmin = min(bqm.linear.values())
    #     self.assertEqual(lmin, bqm.linear.min())

    #     lmax = max(bqm.linear.values())
    #     self.assertEqual(lmax, bqm.linear.max())

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_quad_minmax(self, name, BQM):
    #     num_vars = 10
    #     D = np.arange(num_vars*num_vars).reshape((num_vars, num_vars))
    #     bqm = BQM(D, 'SPIN') 

    #     qmin = min(bqm.quadratic.values())
    #     self.assertEqual(qmin, bqm.quadratic.min())
       
    #     qmax = max(bqm.quadratic.values())
    #     self.assertEqual(qmax, bqm.quadratic.max())

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_lin_minmax_empty(self, name, BQM):
    #     bqm = BQM('SPIN') 

    #     # Test when default is not set
    #     with self.assertRaises(ValueError):
    #         bqm.linear.min()

    #     with self.assertRaises(ValueError):
    #         bqm.linear.max()
        
    #     # Test when default is set
    #     self.assertEqual(bqm.linear.min(default=1), 1)
    #     self.assertEqual(bqm.linear.max(default=2), 2)

    # @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    # def test_quad_minmax_empty(self, name, BQM):
    #     bqm = BQM(500, 'SPIN') 

    #     # Test when default is not set
    #     with self.assertRaises(ValueError):
    #         bqm.quadratic.min()

    #     with self.assertRaises(ValueError):
    #         bqm.quadratic.max()

    #     # Test when default is set
    #     self.assertEqual(bqm.quadratic.min(default=1), 1)
    #     self.assertEqual(bqm.quadratic.max(default=2), 2)


class TestConstraint(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SHAPEABLE_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SHAPEABLE_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SHAPEABLE_SUBCLASSES])
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

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SHAPEABLE_SUBCLASSES])
    def test_unknown_variable(self, name, BQM):
        bqm = BQM('BINARY')
        with self.assertRaises(ValueError):
            bqm.add_linear_equality_constraint(
                [(0, 0)], lagrange_multiplier=1, constant=-1)

    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SHAPEABLE_SUBCLASSES])
    def test_out_of_range_variable(self, name, BQM):
        bqm = BQM('BINARY')
        u = bqm.add_variable(5)
        with self.assertRaises(ValueError):
            bqm.add_linear_equality_constraint(
                [(u + 1, 0)], lagrange_multiplier=1, constant=-1)
