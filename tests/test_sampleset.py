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

import collections.abc as abc
import concurrent.futures
import fractions
import json
import pickle
import unittest

from collections import OrderedDict

import numpy as np

import dimod


try:
    import pandas as pd
    _pandas = True
except ImportError:
    _pandas = False


class Test_as_samples(unittest.TestCase):
    # tests for as_samples function

    def test_copy(self):
        # Check case where samples_like is a tuple
        samples_like = np.ones((5, 5))
        labels = list('abcde')
        arr, lab = dimod.as_samples((samples_like, labels))
        np.testing.assert_array_equal(arr, np.ones((5, 5)))
        self.assertEqual(lab, list('abcde'))
        self.assertIs(labels, lab)
        self.assertTrue(np.shares_memory(arr, samples_like))

        arr, lab = dimod.as_samples((samples_like, labels), copy=True)
        self.assertFalse(np.shares_memory(arr, samples_like))

        # Check case where samples_like is a dimod.SampleSet
        sampleset = dimod.SampleSet.from_samples([{'a': -1, 'b': +1},
                                                  {'a': +1, 'b': +1}],
                                                 'SPIN',
                                                 energy=[-1.0, 1.0])

        arr, lab = dimod.as_samples(sampleset)
        self.assertTrue(np.shares_memory(arr, sampleset.record.sample))
        arr, lab = dimod.as_samples(sampleset, copy=True)
        self.assertFalse(np.shares_memory(arr, sampleset.record.sample))

        arr, lab = dimod.as_samples(sampleset, dtype=np.int8)
        self.assertTrue(np.shares_memory(arr, sampleset.record.sample))
        arr, lab = dimod.as_samples(sampleset, dtype=np.int8, copy=True)
        self.assertFalse(np.shares_memory(arr, sampleset.record.sample))

    def test_dict_with_inconsistent_labels(self):
        with self.assertRaises(ValueError):
            with self.assertWarns(DeprecationWarning):
                dimod.as_samples(({'a': -1}, 'b'))

    def test_dict_with_labels(self):
        with self.assertWarns(DeprecationWarning):
            arr, labels = dimod.as_samples(({'a': -1}, 'a'))
        np.testing.assert_array_equal(arr, [[-1]])
        self.assertEqual(labels, ['a'])

    def test_empty_dict(self):
        # one sample, no labels
        arr, labels = dimod.as_samples({})
        np.testing.assert_array_equal(arr, np.zeros((1, 0)))
        self.assertEqual(labels, [])

    def test_empty_list(self):
        # no samples, no labels
        arr, labels = dimod.as_samples([])
        np.testing.assert_array_equal(arr, np.zeros((0, 0)))
        self.assertEqual(labels, [])

    def test_empty_list_labelled(self):
        # no samples, no labels
        arr, labels = dimod.as_samples(([], []))
        np.testing.assert_array_equal(arr, np.zeros((0, 0)))
        self.assertEqual(labels, [])

        # no samples, 1 label
        arr, labels = dimod.as_samples(([], ['a']))
        np.testing.assert_array_equal(arr, np.zeros((0, 1)))
        self.assertEqual(labels, ['a'])

        # no samples, 2 labels
        arr, labels = dimod.as_samples(([], ['a', 'b']))
        np.testing.assert_array_equal(arr, np.zeros((0, 2)))
        self.assertEqual(labels, ['a', 'b'])

    def test_empty_ndarray(self):
        arr, labels = dimod.as_samples(np.ones(0))
        np.testing.assert_array_equal(arr, np.zeros((0, 0)))
        self.assertEqual(labels, [])

    def test_float_array(self):
        arr, labels = dimod.as_samples(([1.5, 3], 'ab'))
        np.testing.assert_array_equal(arr, [[1.5, 3]])
        self.assertEqual(labels, ['a', 'b'])

    def test_float_dict(self):
        arr, labels = dimod.as_samples({'a': 1.5, 'b': 3})
        np.testing.assert_array_equal(arr, [[1.5, 3]])
        self.assertEqual(labels, ['a', 'b'])

    def test_iterator(self):
        arr, labels = dimod.as_samples(([-1] for _ in range(10)))
        np.testing.assert_array_equal(arr, -np.ones((10, 1)))
        self.assertEqual(labels, [0])

    def test_iterator_empty(self):
        arr, labels = dimod.as_samples(iter([]))

        np.testing.assert_array_equal(arr, np.empty((0, 0)))
        self.assertEqual(labels, [])

    def test_iterator_labelled(self):
        with self.assertRaises(TypeError):
            dimod.as_samples((([-1] for _ in range(10)), 'a'))

    def test_labels_type(self):
        with self.subTest('str'):
            arr = np.ones((5, 5))
            lbl = 'abcde'
            samples, labels = dimod.as_samples((arr, lbl),
                                               labels_type=dimod.variables.Variables)
            self.assertIsInstance(labels, dimod.variables.Variables)
            self.assertEqual(lbl, labels)
            self.assertIs(arr, samples)

        with self.subTest('Variables'):
            arr = np.ones((5, 5))
            lbl = dimod.variables.Variables('abcde')
            samples, labels = dimod.as_samples((arr, lbl),
                                               labels_type=dimod.variables.Variables)
            self.assertIsInstance(labels, dimod.variables.Variables)
            self.assertEqual(lbl, 'abcde')
            self.assertIs(arr, samples)
            self.assertIs(lbl, labels)

        with self.subTest('mapping'):
            samples, labels = dimod.as_samples({'a': 1, 'b': 1},
                                               labels_type=dimod.variables.Variables)
            self.assertIsInstance(labels, dimod.variables.Variables)
            self.assertEqual(set(labels), set('ab'))

        with self.subTest('sampleset'):
            sampleset = dimod.SampleSet.from_samples({'a': 1, 'b': 1}, vartype='BINARY', energy=0)

            samples, labels = dimod.as_samples(sampleset, labels_type=dimod.variables.Variables)
            self.assertIsInstance(labels, dimod.variables.Variables)
            self.assertEqual(set(labels), set('ab'))

    def test_list_of_empty(self):
        arr, labels = dimod.as_samples([[], [], []])
        np.testing.assert_array_equal(arr, np.empty((3, 0)))
        self.assertEqual(labels, [])

        arr, labels = dimod.as_samples([{}, {}, {}])
        np.testing.assert_array_equal(arr, np.empty((3, 0)))
        self.assertEqual(labels, [])

        arr, labels = dimod.as_samples(np.empty((3, 0)))
        np.testing.assert_array_equal(arr, np.empty((3, 0)))
        self.assertEqual(labels, [])

    def test_list_discrete(self):
        arr, labels = dimod.as_samples([int(1 << 8)])

        self.assertEqual(arr.dtype, np.int16)
        np.testing.assert_array_equal(arr, [[int(1 << 8)]])

    def test_mixed_sampletype(self):
        s1 = [0, 1]
        s2 = OrderedDict([(1, 0), (0, 1)])
        s3 = OrderedDict([(0, 1), (1, 0)])

        arr, labels = dimod.as_samples([s1, s2, s3])
        np.testing.assert_array_equal(arr, [[0, 1], [1, 0], [1, 0]])
        self.assertEqual(labels, [0, 1])

    def test_ndarray(self):
        arr, labels = dimod.as_samples(np.ones(5, dtype=np.int32))
        np.testing.assert_array_equal(arr, np.ones((1, 5)))
        self.assertEqual(labels, list(range(5)))
        self.assertEqual(arr.dtype, np.int32)

    def test_ndarray_labelled(self):
        arr, labels = dimod.as_samples((np.ones(5, dtype=np.int32), 'abcde'))
        np.testing.assert_array_equal(arr, np.ones((1, 5)))
        self.assertEqual(labels, ['a', 'b', 'c', 'd', 'e'])
        self.assertEqual(arr.dtype, np.int32)


class TestChangeVartype(unittest.TestCase):
    def test_non_blocking(self):

        future = concurrent.futures.Future()

        sampleset = dimod.SampleSet.from_future(future)

        # shouldn't block or raise
        new = sampleset.change_vartype(dimod.BINARY)

        future.set_result(dimod.SampleSet.from_samples({'a': -1},
                                                       dimod.SPIN,
                                                       energy=1))

        np.testing.assert_array_equal(new.record.sample, [[0]])


class TestConstruction(unittest.TestCase):
    def test_from_samples(self):

        nv = 5
        sample_sets = [dimod.SampleSet.from_samples(np.ones((nv, nv), dtype='int8'), dimod.SPIN, energy=np.ones(nv)),
                       dimod.SampleSet.from_samples([[1]*nv for _ in range(nv)], dimod.SPIN, energy=np.ones(nv)),
                       dimod.SampleSet.from_samples([{v: 1 for v in range(nv)} for _ in range(nv)], dimod.SPIN,
                                                    energy=np.ones(nv)),
                       dimod.SampleSet.from_samples((np.ones((nv, nv), dtype='int8'), list(range(nv))), dimod.SPIN,
                                                    energy=np.ones(nv)),
                       ]

        # all should be the same
        self.assertEqual(sample_sets[1:], sample_sets[:-1])

    def test_from_samples_str_labels(self):
        nv = 5
        alpha = 'abcde'
        sample_sets = [dimod.SampleSet.from_samples([{v: 1 for v in alpha} for _ in range(nv)], dimod.SPIN,
                                                    energy=np.ones(nv)),
                       dimod.SampleSet.from_samples((np.ones((nv, nv), dtype='int8'), alpha), dimod.SPIN,
                                                    energy=np.ones(nv)),
                       ]

        # all should be the same
        self.assertEqual(sample_sets[1:], sample_sets[:-1])

    def test_from_samples_single_sample(self):
        ss0 = dimod.SampleSet.from_samples(([-1, +1], 'ab'), dimod.SPIN, energy=1.0)
        ss1 = dimod.SampleSet.from_samples({'a': -1, 'b': +1}, dimod.SPIN, energy=1.0)

        self.assertEqual(ss0, ss1)

        ss2 = dimod.SampleSet.from_samples([-1, +1], dimod.SPIN, energy=1.0)
        ss3 = dimod.SampleSet.from_samples({0: -1, 1: +1}, dimod.SPIN, energy=1.0)

        self.assertEqual(ss2, ss3)

    def test_from_samples_iterator(self):
        ss0 = dimod.SampleSet.from_samples(np.ones((100, 5), dtype='int8'), dimod.BINARY, energy=np.ones(100))

        # should work for iterable
        ss1 = dimod.SampleSet.from_samples(list(ss0.samples()), dimod.BINARY, energy=ss0.record.energy)

        self.assertEqual(len(ss0), len(ss1))
        self.assertEqual(ss0, ss1)

    def test_from_samples_fields_single(self):
        ss = dimod.SampleSet.from_samples({'a': 1, 'b': -1}, dimod.SPIN, energy=1.0, a=5, b='b')

        self.assertIn('a', ss.record.dtype.fields)
        self.assertIn('b', ss.record.dtype.fields)
        self.assertTrue(all(ss.record.a == [5]))
        self.assertTrue(all(ss.record.b == ['b']))

    def test_from_samples_fields_multiple(self):
        ss = dimod.SampleSet.from_samples(np.ones((2, 5)), dimod.BINARY, energy=[0, 0], a=[-5, 5], b=['a', 'b'])

        self.assertIn('a', ss.record.dtype.fields)
        self.assertIn('b', ss.record.dtype.fields)
        self.assertTrue(all(ss.record.a == [-5, 5]))
        self.assertTrue(all(ss.record.b == ['a', 'b']))

    def test_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            dimod.SampleSet.from_samples(np.ones((3, 5)), dimod.SPIN, energy=[5, 5])

    def test_shorter_samples(self):
        ss = dimod.SampleSet.from_samples(np.ones((100, 5), dtype='int8'), dimod.BINARY, energy=np.ones(100))

        self.assertEqual(len(list(ss.samples(n=1))), 1)

    def test_from_samples_empty(self):

        self.assertEqual(len(dimod.SampleSet.from_samples([], dimod.SPIN, energy=[], a=1)), 0)

        self.assertEqual(len(dimod.SampleSet.from_samples(np.empty((0, 0)), dimod.SPIN, energy=[], a=1)), 0)

    def test_from_samples_with_aggregation(self):
        samples = dimod.SampleSet.from_samples(([[-1, 1], [-1, 1]], 'ab'), dimod.SPIN, energy=[0.0, 0.0],
                                               aggregate_samples=True)
        self.assertEqual(samples.aggregate(),
                         dimod.SampleSet.from_samples(([-1, 1], 'ab'), dimod.SPIN, energy=0.0, num_occurrences=2))

    def test_from_bqm_single_sample(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1})
        samples = dimod.SampleSet.from_samples_bqm({'a': -1, 'b': 1}, bqm)
        self.assertEqual(samples,
                         dimod.SampleSet.from_samples(([-1, 1], 'ab'), dimod.SPIN, energy=1))

    def test_from_bqm_with_sorting(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {(0, 1): -1, (1, 2): -1})

        raw = np.triu(np.ones((3, 3)))
        variables = [2, 1, 0]

        samples = dimod.SampleSet.from_samples_bqm((raw, variables), bqm)
        self.assertEqual(samples.variables, [0, 1, 2])
        np.testing.assert_array_equal(np.flip(raw, 1), samples.record.sample)

    def test_from_bqm_empty_list(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1})
        sampleset = dimod.SampleSet.from_samples_bqm([], bqm)
        self.assertEqual(len(sampleset), 0)
        self.assertEqual(set(sampleset.variables), set('ab'))

    def test_from_cqm(self):
        cqm = dimod.ConstrainedQuadraticModel()
        x, y, z = dimod.Binaries(['x', 'y', 'z'])
        cqm.set_objective(x*y + 2*y*z)
        cqm.add_constraint(x*y == 1, label='constraint_1')
        samples = dimod.SampleSet.from_samples_cqm({'x': 0, 'y': 1, 'z': 1}, cqm)
        self.assertEqual(samples,
                         dimod.SampleSet.from_samples(
                            ([0, 1, 1], 'xyz'), dimod.INTEGER, energy=2, is_feasible=False,
                            is_satisfied=[False], info={'constraint_labels': ['constraint_1']}))
        self.assertEqual(samples.record.is_satisfied.dtype, np.dtype(bool))
        self.assertEqual(samples.record.is_feasible.dtype, np.dtype(bool))

    def test_from_cqm_empty_list(self):
        cqm = dimod.ConstrainedQuadraticModel()
        cqm.set_objective(dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1}))
        sampleset = dimod.SampleSet.from_samples_cqm([], cqm)
        self.assertEqual(len(sampleset), 0)
        self.assertEqual(set(sampleset.variables), set('ab'))
        self.assertEqual(sampleset.record.is_satisfied.dtype, np.dtype(bool))
        self.assertEqual(sampleset.record.is_feasible.dtype, np.dtype(bool))
        self.assertEqual(sampleset.record.is_satisfied.shape, (0, 0))

    def test_from_samples_cqm_soft(self):
        cqm = dimod.ConstrainedQuadraticModel()
        x, y = dimod.Binaries('xy')
        cqm.set_objective(x-y)
        cqm.add_constraint(x+y == 2, label='c0', weight=2)
        cqm.add_constraint(x+y == 1, label='c1', weight=3)
        cqm.add_constraint(3*x <= 1, label='c2', weight=4, penalty='quadratic')
        cqm.add_constraint(x+y == 2, label='c3')  # hard

        sampleset = dimod.SampleSet.from_samples_cqm([{'x': 1, 'y': 1}, {'x': 0, 'y': 1}], cqm)

        self.assertEqual(sampleset.info['constraint_labels'], ['c0', 'c1', 'c2', 'c3'])
        np.testing.assert_array_equal(sampleset.record.is_feasible, [True, False])
        np.testing.assert_array_equal(sampleset.record.is_satisfied,
                                      [[True, False, False, True],
                                       [False, True, True, False]])
        np.testing.assert_array_equal(sampleset.record.energy, [19, 1])

    def test_from_sample_cqm_soft_single_constraint(self):
        x, y = dimod.Binaries('xy')
        with self.subTest(">="):
            cqm = dimod.ConstrainedQuadraticModel()
            c0 = cqm.add_constraint(x + y >= 2, weight=5)
            sampleset = dimod.SampleSet.from_samples_cqm([{'x': 0, 'y': 0}, {'x': 1, 'y': 1}], cqm)
            np.testing.assert_array_equal(sampleset.record.is_feasible, [True, True])
            np.testing.assert_array_equal(sampleset.record.is_satisfied, [[False], [True]])
            np.testing.assert_array_equal(sampleset.record.energy, [10, 0])

        with self.subTest("=="):
            cqm = dimod.ConstrainedQuadraticModel()
            c0 = cqm.add_constraint(x + y == 2, weight=5)
            sampleset = dimod.SampleSet.from_samples_cqm([{'x': 0, 'y': 0}, {'x': 1, 'y': 1}], cqm)
            np.testing.assert_array_equal(sampleset.record.is_feasible, [True, True])
            np.testing.assert_array_equal(sampleset.record.is_satisfied, [[False], [True]])
            np.testing.assert_array_equal(sampleset.record.energy, [10, 0])

        with self.subTest("<="):
            cqm = dimod.ConstrainedQuadraticModel()
            c0 = cqm.add_constraint(x + y <= 0, weight=5)
            sampleset = dimod.SampleSet.from_samples_cqm([{'x': 0, 'y': 0}, {'x': 1, 'y': 1}], cqm)
            np.testing.assert_array_equal(sampleset.record.is_feasible, [True, True])
            np.testing.assert_array_equal(sampleset.record.is_satisfied, [[True], [False]])
            np.testing.assert_array_equal(sampleset.record.energy, [0, 10])

    def test_from_sample_cqm_soft_single_constraint_non_zero_activity(self):
        x, y = dimod.Binaries('xy')
        with self.subTest(">="):
            cqm = dimod.ConstrainedQuadraticModel()
            c0 = cqm.add_constraint(x + y >= 1, weight=5)
            sampleset = dimod.SampleSet.from_samples_cqm([{'x': 0, 'y': 0}, {'x': 1, 'y': 1}], cqm)
            np.testing.assert_array_equal(sampleset.record.is_feasible, [True, True])
            np.testing.assert_array_equal(sampleset.record.is_satisfied, [[False], [True]])
            np.testing.assert_array_equal(sampleset.record.energy, [5, 0])

        with self.subTest("=="):
            cqm = dimod.ConstrainedQuadraticModel()
            c0 = cqm.add_constraint(x + y == 1, weight=5)
            sampleset = dimod.SampleSet.from_samples_cqm([{'x': 0, 'y': 0}, {'x': 1, 'y': 1}], cqm)
            np.testing.assert_array_equal(sampleset.record.is_feasible, [True, True])
            np.testing.assert_array_equal(sampleset.record.is_satisfied, [[False], [False]])
            np.testing.assert_array_equal(sampleset.record.energy, [5, 5])

        with self.subTest("<="):
            cqm = dimod.ConstrainedQuadraticModel()
            c0 = cqm.add_constraint(x + y <= 1, weight=5)
            sampleset = dimod.SampleSet.from_samples_cqm([{'x': 0, 'y': 0}, {'x': 1, 'y': 1}], cqm)
            np.testing.assert_array_equal(sampleset.record.is_feasible, [True, True])
            np.testing.assert_array_equal(sampleset.record.is_satisfied, [[True], [False]])
            np.testing.assert_array_equal(sampleset.record.energy, [0, 5])


class TestDiscreteSampleSet(unittest.TestCase):
    def test_aggregate(self):
        samples = [{'a': 1, 'b': 56}, {'a': 1, 'b': 56}, {'a': 1, 'b': 3}]
        ss = dimod.SampleSet.from_samples(samples, 'DISCRETE',
                                          energy=[2, 2, 3])

        new = ss.aggregate()

        np.testing.assert_array_equal(new.record.sample, [[1, 56], [1, 3]])

    def test_data(self):
        ss = dimod.SampleSet.from_samples(([[0, 107, 236], [3, 21, 1]], 'abc'),
                                          'DISCRETE', energy=[2, 1])

        self.assertEqual(list(ss.data(['sample', 'energy'])),
                         [({'a': 3, 'b': 21, 'c': 1}, 1),
                          ({'a': 0, 'b': 107, 'c': 236}, 2)])

    def test_from_samples_list(self):
        ss = dimod.SampleSet.from_samples([[0, 107, 236], [3, 321, 1]],
                                          'DISCRETE', energy=[2, 1])

        self.assertIs(ss.vartype, dimod.DISCRETE)
        np.testing.assert_array_equal(ss.record.sample,
                                      [[0, 107, 236], [3, 321, 1]])
        np.testing.assert_array_equal(ss.record.energy, [2, 1])

    def test_samples(self):
        ss = dimod.SampleSet.from_samples([[0, 107, 236], [3, 321, 1]],
                                          'DISCRETE', energy=[1, 2])

        np.testing.assert_array_equal(ss.samples()[:, [0, 1, 2]],
                                      [[0, 107, 236], [3, 321, 1]])

    def test_serializable(self):
        samples = [{'a': 1, 'b': 56}, {'a': 1, 'b': 56}, {'a': 1, 'b': 3}]
        ss = dimod.SampleSet.from_samples(samples, 'DISCRETE',
                                          energy=[2, 2, 3])

        new = dimod.SampleSet.from_serializable(ss.to_serializable())


class TestEq(unittest.TestCase):
    def test_ordered(self):
        # samplesets should be equal regardless of variable order
        ss0 = dimod.SampleSet.from_samples(([-1, 1], 'ab'), dimod.SPIN, energy=0.0)
        ss1 = dimod.SampleSet.from_samples(([1, -1], 'ba'), dimod.SPIN, energy=0.0)
        ss2 = dimod.SampleSet.from_samples(([1, -1], 'ab'), dimod.SPIN, energy=0.0)
        ss3 = dimod.SampleSet.from_samples(([1, -1], 'ac'), dimod.SPIN, energy=0.0)

        self.assertEqual(ss0, ss1)
        self.assertNotEqual(ss0, ss2)
        self.assertNotEqual(ss1, ss3)


class TestFilter(unittest.TestCase):
    def test_simple(self):
        sampleset = dimod.SampleSet(np.rec.array([
            ([0., 1.], 0., 1,  True, [True, True]),
            ([1., 0.], 0., 1,  True, [True, True]),
            ([1., 1.], 0., 1,  False, [False, True]),
            ([0., 0.], 0., 1,  False, [True, False])],
            dtype=[('sample', '<f8', (2,)), ('energy', '<f8'), ('num_occurrences', '<i4'),
                   ('is_feasible', '?'), ('is_satisfied', '?', (2,))]),
            ['x', 'y'],
            {},
            'INTEGER')

        new = sampleset.filter(lambda d: d.is_feasible)

        self.assertEqual(len(new), 2)
        np.testing.assert_array_equal(new.record.sample, [[0, 1], [1, 0]])


class TestAggregate(unittest.TestCase):
    def test_aggregate_simple(self):
        samples = dimod.SampleSet.from_samples(([[-1, 1], [-1, 1]], 'ab'), dimod.SPIN, energy=[0.0, 0.0])

        self.assertEqual(samples.aggregate(),
                         dimod.SampleSet.from_samples(([-1, 1], 'ab'), dimod.SPIN, energy=0.0, num_occurrences=2))

        # original should not be changed
        self.assertEqual(samples,
                         dimod.SampleSet.from_samples(([[-1, 1], [-1, 1]], 'ab'), dimod.SPIN, energy=[0.0, 0.0]))

    def test_order_preservation_2x2_unique(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1}, 0, dimod.SPIN)

        # these are unique so order should be preserved
        ss1 = dimod.SampleSet.from_samples_bqm([{'a': 1, 'b': -1},
                                                {'a': -1, 'b': 1}],
                                               bqm)
        ss2 = ss1.aggregate()

        self.assertEqual(ss1, ss2)

    def test_order_preservation(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1}, 0, dimod.SPIN)

        ss1 = dimod.SampleSet.from_samples_bqm([{'a': 1, 'b': -1},
                                                {'a': -1, 'b': 1},
                                                {'a': 1, 'b': -1}],
                                               bqm)
        ss2 = ss1.aggregate()

        target = dimod.SampleSet.from_samples_bqm([{'a': 1, 'b': -1},
                                                   {'a': -1, 'b': 1}],
                                                  bqm,
                                                  num_occurrences=[2, 1])

        self.assertEqual(target, ss2)

    def test_order_preservation_doubled(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': 1, 'bc': -1}, 0, dimod.SPIN)
        ss1 = dimod.SampleSet.from_samples_bqm(([[1, 1, 0],
                                                 [1, 0, 0],
                                                 [0, 0, 0],
                                                 [0, 0, 0],
                                                 [1, 1, 0],
                                                 [1, 0, 0],
                                                 [0, 0, 0]], 'abc'),
                                               bqm)

        target = dimod.SampleSet.from_samples_bqm(([[1, 1, 0],
                                                    [1, 0, 0],
                                                    [0, 0, 0]], 'abc'),
                                                  bqm,
                                                  num_occurrences=[2, 2, 3])

        self.assertEqual(target, ss1.aggregate())

    def test_num_occurences(self):
        samples = [[-1, -1, +1],
                   [-1, +1, +1],
                   [-1, +1, +1],
                   [-1, -1, -1],
                   [-1, +1, +1]]
        agg_samples = [[-1, -1, +1],
                       [-1, +1, +1],
                       [-1, -1, -1]]
        labels = 'abc'

        sampleset = dimod.SampleSet.from_samples((samples, labels), energy=0,
                                                 vartype=dimod.SPIN)
        aggregated = dimod.SampleSet.from_samples((agg_samples, labels), energy=0,
                                                  vartype=dimod.SPIN,
                                                  num_occurrences=[1, 3, 1])

        self.assertEqual(sampleset.aggregate(), aggregated)


class TestAppend(unittest.TestCase):
    def test_sampleset1_append1(self):
        sampleset = dimod.SampleSet.from_samples({'a': -1, 'b': 1}, dimod.SPIN, energy=0)
        new = dimod.append_variables(sampleset, {'c': -1, 'd': -1})

        target = dimod.SampleSet.from_samples({'a': -1, 'b': 1, 'c': -1, 'd': -1}, dimod.SPIN, energy=0)

        self.assertEqual(new, target)

    def test_sampleset2_append1(self):
        sampleset = dimod.SampleSet.from_samples([{'a': -1, 'b': 1}, {'a': -1, 'b': -1}],
                                                 dimod.SPIN, energy=0)
        new = dimod.append_variables(sampleset, {'c': -1, 'd': -1})

        target = dimod.SampleSet.from_samples([{'a': -1, 'b': 1, 'c': -1, 'd': -1},
                                               {'a': -1, 'b': -1, 'c': -1, 'd': -1}],
                                              dimod.SPIN, energy=0)

        self.assertEqual(new, target)

    def test_sampleset2_append2(self):
        sampleset = dimod.SampleSet.from_samples([{'a': -1, 'b': 1}, {'a': -1, 'b': -1}],
                                                 dimod.SPIN, energy=0)
        new = dimod.append_variables(sampleset, [{'c': -1, 'd': -1}, {'c': 1, 'd': 1}])

        target = dimod.SampleSet.from_samples([{'a': -1, 'b': 1, 'c': -1, 'd': -1},
                                               {'a': -1, 'b': -1, 'c': 1, 'd': 1}],
                                              dimod.SPIN, energy=0)

        self.assertEqual(new, target)

    def test_sampleset2_append3(self):
        sampleset = dimod.SampleSet.from_samples([{'a': -1, 'b': 1}, {'a': -1, 'b': -1}],
                                                 dimod.SPIN, energy=0)

        with self.assertRaises(ValueError):
            dimod.append_variables(sampleset, [{'c': -1, 'd': -1}, {'c': 1, 'd': 1}, {'c': -1, 'd': -1}])

    def test_overlapping_variables(self):
        sampleset = dimod.SampleSet.from_samples([{'a': -1, 'b': 1}, {'a': -1, 'b': -1}],
                                                 dimod.SPIN, energy=0)

        with self.assertRaises(ValueError):
            dimod.append_variables(sampleset, [{'c': -1, 'd': -1, 'a': -1}])

    def test_two_samplesets(self):
        sampleset0 = dimod.SampleSet.from_samples([{'a': -1, 'b': 1}, {'a': -1, 'b': -1}],
                                                  dimod.SPIN, energy=[-2, 2])
        sampleset1 = dimod.SampleSet.from_samples([{'c': -1, 'd': 1}, {'c': -1, 'd': -1}],
                                                  dimod.SPIN, energy=[-1, 1])

        target = dimod.SampleSet.from_samples([{'a': -1, 'b': 1, 'c': -1, 'd': 1},
                                               {'a': -1, 'b': -1, 'c': -1, 'd': -1}],
                                              dimod.SPIN, energy=[-2, 2])

        self.assertEqual(dimod.append_variables(sampleset0, sampleset1), target)

class TestAppendVectors(unittest.TestCase):
    def test_scalar(self):
        sampleset = dimod.SampleSet.from_samples([[-1,  1], [-1,  1]], energy=[1, 1], vartype='SPIN') 

        vect = [2] * len(sampleset.record.energy)
        sampleset = dimod.append_data_vectors(sampleset, new=vect)

        for s in sampleset.data_vectors['new']:
            self.assertEqual(s, 2)

    def test_array(self):
        sampleset = dimod.SampleSet.from_samples([[-1,  1], [-1,  1]], energy=[1, 1], vartype='SPIN') 

        vect0 = [np.array([0, 1, 2]), np.array([1, 2, 3])]
        vect1 = [[0, 1, 2], [1, 2, 3]]
        vect2 = [(0, 1, 2), (1, 2, 3)]

        sampleset = dimod.append_data_vectors(sampleset, arrs=vect0, lists=vect1, tups=vect2)
        
        a = sampleset.data_vectors['arrs']
        l = sampleset.data_vectors['lists']
        t = sampleset.data_vectors['tups']

        np.testing.assert_array_equal(a, vect0)
        np.testing.assert_array_equal(l, vect0)
        np.testing.assert_array_equal(t, vect0)

    def test_invalid(self):
        sampleset = dimod.SampleSet.from_samples([[-1,  1], [-1,  1]], energy=[1, 1], vartype='SPIN') 

        # incorrect length
        with self.assertRaises(ValueError):
            sampleset = dimod.append_data_vectors(sampleset, new=[1, 2, 3])

        # appending vector with field name that already exists
        with self.assertRaises(ValueError):
            sampleset = dimod.append_data_vectors(sampleset, energy=[1, 2])

        # invalid type
        with self.assertRaises(ValueError):
            sampleset = dimod.append_data_vectors(sampleset, sets=[{0}, {1}])


class TestDropVariables(unittest.TestCase):
    def test_generator(self):
        sampleset = dimod.ExactSolver().sample_ising({}, {'ab': 1, 'bc': 1})
        new = dimod.drop_variables(sampleset, (v for v in 'cb'))

        self.assertEqual(new.variables, 'a')

        for datum, new_datum in zip(sampleset.data(sorted_by=None), new.data(sorted_by=None)):
            self.assertEqual(datum.energy, new_datum.energy)
            self.assertEqual(datum.num_occurrences, new_datum.num_occurrences)
            self.assertEqual(datum.sample['a'], new_datum.sample['a'])

    def test_superset(self):
        sampleset = dimod.ExactSolver().sample_ising({}, {'ab': 1, 'bc': 1})
        new = dimod.drop_variables(sampleset, 'cefg')

        self.assertEqual(new.variables, 'ab')

        for datum, new_datum in zip(sampleset.data(sorted_by=None), new.data(sorted_by=None)):
            self.assertEqual(datum.energy, new_datum.energy)
            self.assertEqual(datum.num_occurrences, new_datum.num_occurrences)
            self.assertEqual(datum.sample['a'], new_datum.sample['a'])
            self.assertEqual(datum.sample['b'], new_datum.sample['b'])

    def test_variables_were_dropped(self):
        samples = np.triu(np.ones((5, 5)))
        labels = 'abcde'
        sampleset = dimod.SampleSet.from_samples((samples, labels), vartype='BINARY', energy=1)
        variables_to_drop = "ae"
        expected = set([label for label in labels if label not in variables_to_drop])
        obtained = set(dimod.sampleset.drop_variables(sampleset, variables_to_drop).variables)
        self.assertEqual(obtained, expected)


class TestFromFuture(unittest.TestCase):
    def test_default(self):

        future = concurrent.futures.Future()

        response = dimod.SampleSet.from_future(future)

        self.assertIsInstance(response, dimod.SampleSet)
        self.assertFalse(hasattr(response, '_record'))  # should not have a record yet

        self.assertFalse(response.done())

        # make future return a Response
        future.set_result(dimod.SampleSet.from_samples([-1, -1, 1], energy=.5, info={'another_field': .5}, vartype=dimod.SPIN))

        self.assertTrue(response.done())

        # accessing response.record should resolve the future
        np.testing.assert_array_equal(response.record.sample,
                                      np.array([[-1, -1, 1]]))
        np.testing.assert_array_equal(response.record.energy,
                                      np.array([.5]))
        np.testing.assert_array_equal(response.record.num_occurrences,
                                      np.array([1]))
        self.assertEqual(response.info, {'another_field': .5})
        self.assertIs(response.vartype, dimod.SPIN)
        self.assertEqual(response.variables, [0, 1, 2])

    def test_typical(self):
        result = {'occurrences': [1],
                  'active_variables': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                                       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                       33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                       48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                                       63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                                       78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                                       93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
                                       106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
                                       118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
                  'num_occurrences': [1],
                  'num_variables': 128,
                  'format': 'qp',
                  'timing': {},
                  'solutions': [[1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1,
                                 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1,
                                 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                                 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1,
                                 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1,
                                 -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]],
                  'energies': [-704.0],
                  'samples': [[1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                               -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1,
                               1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                               -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1,
                               1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1,
                               -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1,
                               1, 1, 1, 1, 1, 1, -1, -1, -1, -1]]}

        future = concurrent.futures.Future()

        def result_to_response(future):
            result = future.result()
            return dimod.SampleSet.from_samples(result['solutions'],
                                                energy=result['energies'],
                                                num_occurrences=result['occurrences'],
                                                vartype=dimod.SPIN)

        response = dimod.SampleSet.from_future(future, result_hook=result_to_response)

        future.set_result(result)

        matrix = response.record.sample

        np.testing.assert_equal(matrix, result['samples'])

    def test_cloud_client_future(self):
        answer = dimod.SampleSet.from_samples(samples_like=[[1]], vartype='SPIN', energy=[1])
        pid = 'problem-id'

        future = concurrent.futures.Future()
        future.wait_id = lambda timeout=None: pid

        ss = dimod.SampleSet.from_future(future)

        with self.subTest('unresolved sampleset from qpu computation has a functional wait_id'):
            self.assertFalse(ss.done())
            self.assertTrue(hasattr(ss, 'wait_id'))
            self.assertEqual(ss.wait_id(), pid)

        future.set_result(answer)
        ss.resolve()

        with self.subTest('resolved sampleset caches wait_id result'):
            self.assertTrue(ss.done())
            self.assertTrue(hasattr(ss, 'wait_id'))
            self.assertEqual(ss.wait_id(), pid)
            self.assertFalse(hasattr(ss, '_future'))


class TestKeepVariables(unittest.TestCase):
    def test_empty(self):
        sampleset = dimod.ExactSolver().sample_ising({}, {'ab': 1, 'bc': 1})
        new = dimod.keep_variables(sampleset, [])
        self.assertEqual(new.record.sample.shape, (len(sampleset), 0))

    def test_generator(self):
        sampleset = dimod.ExactSolver().sample_ising({}, {'ab': 1, 'bc': 1})
        new = dimod.keep_variables(sampleset, (v for v in 'cb'))

        self.assertEqual(new.variables, 'cb')

        for datum, new_datum in zip(sampleset.data(sorted_by=None), new.data(sorted_by=None)):
            self.assertEqual(datum.energy, new_datum.energy)
            self.assertEqual(datum.num_occurrences, new_datum.num_occurrences)
            self.assertEqual(datum.sample['b'], new_datum.sample['b'])
            self.assertEqual(datum.sample['c'], new_datum.sample['c'])

    def test_simple(self):
        sampleset = dimod.ExactSolver().sample_ising({}, {'ab': 1, 'bc': 1})
        new = dimod.keep_variables(sampleset, 'bc')

        self.assertEqual(new.variables, 'bc')

        for datum, new_datum in zip(sampleset.data(sorted_by=None), new.data(sorted_by=None)):
            self.assertEqual(datum.energy, new_datum.energy)
            self.assertEqual(datum.num_occurrences, new_datum.num_occurrences)
            self.assertEqual(datum.sample['b'], new_datum.sample['b'])
            self.assertEqual(datum.sample['c'], new_datum.sample['c'])

    def test_superset(self):
        sampleset = dimod.ExactSolver().sample_ising({}, {'ab': 1, 'bc': 1})
        with self.assertRaises(ValueError):
            dimod.keep_variables(sampleset, 'bcd')


class TestLowest(unittest.TestCase):
    def test_all_equal(self):
        sampleset = dimod.ExactSolver().sample_ising({}, {'ab': 0})
        self.assertEqual(sampleset, sampleset.lowest())

    def test_empty(self):
        sampleset = dimod.SampleSet.from_samples(([], 'ab'), energy=[], vartype=dimod.SPIN)
        self.assertEqual(sampleset, sampleset.lowest())

    def test_tolerance(self):
        sampleset = dimod.ExactSolver().sample_ising({'a': .001}, {('a', 'b'): -1})

        self.assertEqual(sampleset.lowest(atol=.1), sampleset.truncate(2))
        self.assertEqual(sampleset.lowest(atol=0), sampleset.truncate(1))


class TestPickle(unittest.TestCase):
    def test_without_future(self):
        sampleset = dimod.SampleSet.from_samples([{'a': -1, 'b': 1},
                                                  {'a': -1, 'b': -1}],
                                                 dimod.SPIN, energy=0)
        sampleset.info.update({'a': 5})

        new = pickle.loads(pickle.dumps(sampleset))

        self.assertEqual(new, sampleset)
        self.assertEqual(new.info, {'a': 5})

    def test_integer(self):
        sampleset = dimod.SampleSet.from_samples([0, 1], vartype='INTEGER', energy=1)
        new = pickle.loads(pickle.dumps(sampleset))
        self.assertEqual(new, sampleset)


class TestTruncate(unittest.TestCase):
    def test_typical(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({v: -1 for v in range(100)}, {})
        samples = dimod.SampleSet.from_samples_bqm(np.tril(np.ones(100)), bqm.binary)

        # by default should be in reverse order
        new = samples.truncate(10)
        self.assertEqual(len(new), 10)
        for n, sample in enumerate(new.samples()):
            for v, val in sample.items():
                if v > 100 - n - 1:
                    self.assertEqual(val, 0)
                else:
                    self.assertEqual(val, 1)

    def test_unordered(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({v: -1 for v in range(100)}, {})
        samples = dimod.SampleSet.from_samples_bqm(np.triu(np.ones(100)), bqm.binary)

        # now undordered
        new = samples.truncate(10, sorted_by=None)
        self.assertEqual(len(new), 10)
        for n, sample in enumerate(new.samples()):
            for v, val in sample.items():
                if v < n:
                    self.assertEqual(val, 0)
                else:
                    self.assertEqual(val, 1)


class TestSlice(unittest.TestCase):
    def test_typical(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({v: -1 for v in range(100)}, {})
        sampleset = dimod.SampleSet.from_samples_bqm(np.tril(np.ones(100)), bqm.binary)

        # `:10` is equal to `truncate(10)`
        self.assertEqual(sampleset.slice(10), sampleset.truncate(10))

    def test_unordered(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({v: -1 for v in range(100)}, {})
        sampleset = dimod.SampleSet.from_samples_bqm(np.triu(np.ones(100)), bqm.binary)

        # `:10` but for the unordered case
        self.assertEqual(sampleset.slice(10, sorted_by=None), sampleset.truncate(10, sorted_by=None))

    def test_null_slice(self):
        energies = list(range(10))
        sampleset = dimod.SampleSet.from_samples(np.ones((10, 1)), dimod.SPIN, energy=energies)

        self.assertTrue((sampleset.slice().record.energy == energies).all())

    def test_slice_stop_only(self):
        energies = list(range(10))
        sampleset = dimod.SampleSet.from_samples(np.ones((10, 1)), dimod.SPIN, energy=energies)

        self.assertTrue((sampleset.slice(3).record.energy == energies[:3]).all())
        self.assertTrue((sampleset.slice(-3).record.energy == energies[:-3]).all())

    def test_slice_range(self):
        energies = list(range(10))
        sampleset = dimod.SampleSet.from_samples(np.ones((10, 1)), dimod.SPIN, energy=energies)

        self.assertTrue((sampleset.slice(3, 5).record.energy == energies[3:5]).all())
        self.assertTrue((sampleset.slice(3, -3).record.energy == energies[3:-3]).all())
        self.assertTrue((sampleset.slice(-3, None).record.energy == energies[-3:]).all())

    def test_slice_stride(self):
        energies = list(range(10))
        sampleset = dimod.SampleSet.from_samples(np.ones((10, 1)), dimod.SPIN, energy=energies)

        self.assertTrue((sampleset.slice(3, -3, 2).record.energy == energies[3:-3:2]).all())
        self.assertTrue((sampleset.slice(None, None, 2).record.energy == energies[::2]).all())
        self.assertTrue((sampleset.slice(None, None, -1).record.energy == energies[::-1]).all())

    def test_custom_ordering(self):
        custom = list(range(10))
        sampleset = dimod.SampleSet.from_samples(np.ones((10, 1)), dimod.SPIN, energy=None, custom=custom)

        self.assertTrue((sampleset.slice(3, sorted_by='custom').record.custom == custom[:3]).all())
        self.assertTrue((sampleset.slice(3, -3, sorted_by='custom').record.custom == custom[3:-3]).all())
        self.assertTrue((sampleset.slice(None, None, -1, sorted_by='custom').record.custom == custom[::-1]).all())

    def test_kwargs(self):
        sampleset = dimod.SampleSet.from_samples(np.ones((10, 1)), dimod.SPIN, energy=None)

        with self.assertRaises(TypeError):
            sampleset.slice(1, sortedby='invalid-kwarg')


class TestIteration(unittest.TestCase):
    def test_data_reverse(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1})
        sampleset = dimod.SampleSet.from_samples_bqm([{'a': -1, 'b': 1}, {'a': 1, 'b': 1}], bqm)

        samples = list(sampleset.data())
        reversed_samples = list(sampleset.data(reverse=True))
        self.assertEqual(samples, list(reversed(reversed_samples)))

    def test_iterator(self):
        # deprecated feature
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1})
        sampleset = dimod.SampleSet.from_samples_bqm([{'a': -1, 'b': 1}, {'a': 1, 'b': 1}], bqm)
        self.assertIsInstance(sampleset.samples(), abc.Iterator)
        self.assertIsInstance(sampleset.samples(n=2), abc.Iterator)
        with self.assertWarns(DeprecationWarning):
            spl = next(sampleset.samples())
        self.assertEqual(spl, {'a': 1, 'b': 1})


class TestRelabelVariables(unittest.TestCase):
    def test_copy(self):
        bqm = dimod.BinaryQuadraticModel({'a': .1}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        samples = [{'a': -1, 'b': +1}, {'a': +1, 'b': -1},
                   {'a': -1, 'b': +1}, {'a': -1, 'b': -1}]
        energies = [bqm.energy(sample) for sample in samples]
        response = dimod.SampleSet.from_samples(samples, energy=energies, vartype=dimod.SPIN)

        new_response = response.relabel_variables({'a': 0, 'b': 1}, inplace=False)

        # original response should not change
        for sample in response:
            self.assertIn(sample, samples)

        for sample in new_response:
            self.assertEqual(set(sample), {0, 1})

    def test_docstring(self):
        response = dimod.SampleSet.from_samples([{'a': -1}, {'a': +1}], energy=[-1, 1], vartype=dimod.SPIN)
        new_response = response.relabel_variables({'a': 0}, inplace=False)

    def test_partial_inplace(self):
        mapping = {0: '3', 1: 4, 2: 5, 3: 6, 4: 7, 5: '1', 6: '2', 7: '0'}

        response = dimod.SampleSet.from_samples([[-1, +1, -1, +1, -1, +1, -1, +1]], energy=-1, vartype=dimod.SPIN)

        new_response = response.relabel_variables(mapping, inplace=False)

        for new_sample, sample in zip(new_response, response):
            for v, val in sample.items():
                self.assertIn(mapping[v], new_sample)
                self.assertEqual(new_sample[mapping[v]], val)

            self.assertEqual(len(sample), len(new_sample))

    def test_partial(self):

        mapping = {0: '3', 1: 4, 2: 5, 3: 6, 4: 7, 5: '1', 6: '2', 7: '0'}

        response = dimod.SampleSet.from_samples([[-1, +1, -1, +1, -1, +1, -1, +1]], energy=-1, vartype=dimod.SPIN)
        response2 = dimod.SampleSet.from_samples([[-1, +1, -1, +1, -1, +1, -1, +1]], energy=-1, vartype=dimod.SPIN)

        response.relabel_variables(mapping, inplace=True)

        for new_sample, sample in zip(response, response2):
            for v, val in sample.items():
                self.assertIn(mapping[v], new_sample)
                self.assertEqual(new_sample[mapping[v]], val)

            self.assertEqual(len(sample), len(new_sample))

    def test_non_blocking_copy(self):

        future = concurrent.futures.Future()

        sampleset = dimod.SampleSet.from_future(future)

        new = sampleset.relabel_variables({0: 'a'}, inplace=False)  # should not block or raise

        future.set_result(dimod.SampleSet.from_samples({0: -1},
                                                       dimod.SPIN,
                                                       energy=1))

        self.assertIsNot(new, sampleset)
        self.assertEqual(new.variables, ['a'])
        self.assertEqual(sampleset.variables, [0])

    def test_non_blocking_inplace(self):

        future = concurrent.futures.Future()

        sampleset = dimod.SampleSet.from_future(future)

        new = sampleset.relabel_variables({0: 'a'}, inplace=True)  # should not block or raise

        future.set_result(dimod.SampleSet.from_samples({0: -1},
                                                       dimod.SPIN,
                                                       energy=1))

        self.assertIs(new, sampleset)
        self.assertEqual(new.variables, ['a'])
        self.assertEqual(sampleset.variables, ['a'])


class TestSerialization(unittest.TestCase):
    def test_empty_with_bytes(self):
        sampleset = dimod.SampleSet.from_samples([], dimod.BINARY, energy=[])

        dct = sampleset.to_serializable(use_bytes=True)

        new = dimod.SampleSet.from_serializable(dct)

        self.assertEqual(sampleset, new)

    def test_triu_with_bytes(self):
        num_variables = 50
        num_samples = 50
        samples = 2*np.triu(np.ones((num_samples, num_variables)), -4) - 1
        bqm = dimod.BinaryQuadraticModel.from_ising({v: .1*v for v in range(num_variables)}, {})
        sampleset = dimod.SampleSet.from_samples_bqm(samples, bqm)

        dct = sampleset.to_serializable(use_bytes=True)

        new = dimod.SampleSet.from_serializable(dct)

        self.assertEqual(sampleset, new)

    def test_3path_with_bytes(self):
        samples = [[-1, -1, -1, 1], [1, 1, 1, -1]]
        sampleset = dimod.SampleSet.from_samples(samples, energy=0,
                                                 vartype=dimod.SPIN)

        dct = sampleset.to_serializable(use_bytes=True)

        new = dimod.SampleSet.from_serializable(dct)

        self.assertEqual(sampleset, new)

    def test_functional_simple_shapes(self):
        for ns in range(1, 9):
            for nv in range(1, 15):

                raw = np.random.randint(2, size=(ns, nv))

                if ns % 2:
                    vartype = dimod.SPIN
                    raw = 2 * raw - 1
                else:
                    vartype = dimod.BINARY

                samples = dimod.SampleSet.from_samples(raw, vartype, energy=np.ones(ns))
                new_samples = dimod.SampleSet.from_serializable(samples.to_serializable())
                self.assertEqual(samples, new_samples)

    def test_functional_with_info(self):
        sampleset = dimod.SampleSet.from_samples([[-1, 1], [1, -1]], energy=-1,
                                                 vartype=dimod.SPIN,
                                                 info={'hello': 'world'})

        new = dimod.SampleSet.from_serializable(sampleset.to_serializable())

        self.assertEqual(new.info, sampleset.info)

    def test_functional_json(self):
        nv = 4
        ns = 7

        raw = np.random.randint(2, size=(ns, nv))

        samples = dimod.SampleSet.from_samples(raw, dimod.BINARY, energy=np.ones(ns))

        s = json.dumps(samples.to_serializable())
        new_samples = dimod.SampleSet.from_serializable(json.loads(s))
        self.assertEqual(samples, new_samples)

    def test_functional_str(self):
        nv = 4
        ns = 7

        raw = np.random.randint(2, size=(ns, nv))

        samples = dimod.SampleSet.from_samples((raw, 'abcd'), dimod.BINARY, energy=np.ones(ns))

        s = json.dumps(samples.to_serializable())
        new_samples = dimod.SampleSet.from_serializable(json.loads(s))
        self.assertEqual(samples, new_samples)

    def test_tuple_variable_labels(self):
        sampleset = dimod.SampleSet.from_samples(([], [(0, 0), (0, 1), ("a", "b", 2)]), dimod.BINARY, energy=[])

        json_str = json.dumps(sampleset.to_serializable())

        new = dimod.SampleSet.from_serializable(json.loads(json_str))

        self.assertEqual(sampleset, new)

    def test_tuple_variable_labels_nested(self):
        variables = [((0, 1), 0), (('a', (0, 'a')), 1), ("a", "b", 2)]
        sampleset = dimod.SampleSet.from_samples(([], variables), dimod.BINARY,
                                                 energy=[])

        json_str = json.dumps(sampleset.to_serializable())

        new = dimod.SampleSet.from_serializable(json.loads(json_str))

        self.assertEqual(sampleset, new)

    def test_numpy_variable_labels(self):
        h = {0: 0, 1: 1, np.int64(2): 2, np.float32(3): 3,
             fractions.Fraction(4, 1): 4, fractions.Fraction(5, 2): 5,
             '6': 6}

        sampleset = dimod.NullSampler().sample_ising(h, {})

        json.dumps(sampleset.to_serializable())

    def test_non_integer_samples_bool(self):
        samples = np.ones((5, 5), dtype=bool)
        sampleset = dimod.SampleSet.from_samples(samples, 'BINARY', 1)

        new = dimod.SampleSet.from_serializable(sampleset.to_serializable())

        self.assertEqual(sampleset, new)

    def test_non_integer_samples_float(self):
        samples = np.ones((5, 5), dtype=np.float32)
        sampleset = dimod.SampleSet.from_samples(samples, 'BINARY', 1)

        new = dimod.SampleSet.from_serializable(sampleset.to_serializable())

        self.assertEqual(sampleset, new)

    def test_unpacked(self):
        # dev note: we are using an unsupported back door that allows
        # samplesets to handle integer variables. This support could
        # disappear at any time
        samples = np.arange(25).reshape((5, 5))
        sampleset = dimod.SampleSet.from_samples(samples, 'BINARY', 1)

        s = sampleset.to_serializable(use_bytes=False, pack_samples=False)
        new = dimod.SampleSet.from_serializable(s)

        np.testing.assert_array_equal(sampleset.record, new.record)

    def test_unpacked_bytes(self):
        # dev note: we are using an unsupported back door that allows
        # samplesets to handle integer variables. This support could
        # disappear at any time
        samples = np.arange(25).reshape((5, 5))
        sampleset = dimod.SampleSet.from_samples(samples, 'BINARY', 1)

        s = sampleset.to_serializable(use_bytes=True, pack_samples=False)
        new = dimod.SampleSet.from_serializable(s)

        np.testing.assert_array_equal(sampleset.record, new.record)


@unittest.skipUnless(_pandas, "no pandas present")
class TestPandas(unittest.TestCase):
    def test_simple(self):
        samples = dimod.SampleSet.from_samples(([[-1, 1, -1], [-1, -1, 1]], 'abc'),
                                               dimod.SPIN, energy=[-.5, .5])
        df = samples.to_pandas_dataframe()

        other = pd.DataFrame([[-1, 1, -1, -.5, 1], [-1, -1, 1, .5, 1]],
                             columns=['a', 'b', 'c', 'energy', 'num_occurrences'])

        pd.testing.assert_frame_equal(df, other, check_dtype=False)

    def test_sample_column(self):
        samples = dimod.SampleSet.from_samples(([[-1, 1, -1], [-1, -1, 1]], 'abc'),
                                               dimod.SPIN, energy=[-.5, .5])
        df = samples.to_pandas_dataframe(sample_column=True)

        other = pd.DataFrame([[{'a': -1, 'b': 1, 'c': -1}, -0.5, 1],
                              [{'a': -1, 'b': -1, 'c': 1}, 0.5, 1]],
                             columns=['sample', 'energy', 'num_occurrences'])

        pd.testing.assert_frame_equal(df, other, check_dtype=False)


class TestFirst(unittest.TestCase):
    # SampleSet.first property
    def test_empty(self):
        with self.assertRaises(ValueError):
            dimod.SampleSet.from_samples([], dimod.SPIN, energy=[]).first


class TestDataVectors(unittest.TestCase):
    # SampleSet.data_vectors property
    def test_empty(self):
        ss = dimod.SampleSet.from_samples([], dimod.SPIN, energy=[])

        self.assertEqual(set(ss.data_vectors), {'energy', 'num_occurrences'})
        for field, vector in ss.data_vectors.items():
            np.testing.assert_array_equal(vector, [])

    def test_view(self):
        # make sure that the vectors are views
        ss = dimod.SampleSet.from_samples([[-1, 1], [1, 1]], dimod.SPIN, energy=[5, 5])

        self.assertEqual(set(ss.data_vectors), {'energy', 'num_occurrences'})
        for field, vector in ss.data_vectors.items():
            np.shares_memory(vector, ss.record)


class Test_concatenate(unittest.TestCase):
    def test_simple(self):
        ss0 = dimod.SampleSet.from_samples([-1, +1], dimod.SPIN, energy=-1)
        ss1 = dimod.SampleSet.from_samples([+1, -1], dimod.SPIN, energy=-1)
        ss2 = dimod.SampleSet.from_samples([[+1, +1], [-1, -1]], dimod.SPIN, energy=[1, 1])

        comb = dimod.concatenate((ss0, ss1, ss2))

        out = dimod.SampleSet.from_samples([[-1, +1], [+1, -1], [+1, +1], [-1, -1]], dimod.SPIN, energy=[-1, -1, 1, 1])

        self.assertEqual(comb, out)
        np.testing.assert_array_equal(comb.record.sample, out.record.sample)

    def test_variables_order(self):
        ss0 = dimod.SampleSet.from_samples(([-1, +1], 'ab'), dimod.SPIN, energy=-1)
        ss1 = dimod.SampleSet.from_samples(([-1, +1], 'ba'), dimod.SPIN, energy=-1)
        ss2 = dimod.SampleSet.from_samples(([[+1, +1], [-1, -1]], 'ab'), dimod.SPIN, energy=[1, 1])

        comb = dimod.concatenate((ss0, ss1, ss2))

        out = dimod.SampleSet.from_samples(([[-1, +1], [+1, -1], [+1, +1], [-1, -1]], 'ab'),
                                           dimod.SPIN, energy=[-1, -1, 1, 1])

        self.assertEqual(comb, out)
        np.testing.assert_array_equal(comb.record.sample, out.record.sample)

    def test_variables_order(self):
        ss0 = dimod.SampleSet.from_samples(([-1, +1], 'ab'), dimod.SPIN, energy=-1)
        ss1 = dimod.SampleSet.from_samples(([-1, +1], 'ba'), dimod.SPIN, energy=-1)
        ss2 = dimod.SampleSet.from_samples(([[+1, +1], [-1, -1]], 'ab'), dimod.SPIN, energy=[1, 1])

        comb = dimod.concatenate((ss0, ss1, ss2))

        out = dimod.SampleSet.from_samples(([[-1, +1], [+1, -1], [+1, +1], [-1, -1]], 'ab'),
                                           dimod.SPIN, energy=[-1, -1, 1, 1])

        self.assertEqual(comb, out)
        np.testing.assert_array_equal(comb.record.sample, out.record.sample)

    def test_variables_order_and_vartype(self):
        ss0 = dimod.SampleSet.from_samples(([-1, +1], 'ab'), dimod.SPIN, energy=-1)
        ss1 = dimod.SampleSet.from_samples(([-1, +1], 'ba'), dimod.SPIN, energy=-1)
        ss2 = dimod.SampleSet.from_samples(([[1, 1], [0, 0]], 'ab'), dimod.BINARY, energy=[1, 1])

        comb = dimod.concatenate((ss0, ss1, ss2))

        out = dimod.SampleSet.from_samples(([[-1, +1], [+1, -1], [+1, +1], [-1, -1]], 'ab'),
                                           dimod.SPIN, energy=[-1, -1, 1, 1])

        self.assertEqual(comb, out)
        np.testing.assert_array_equal(comb.record.sample, out.record.sample)

    def test_empty(self):
        with self.assertRaises(ValueError):
            dimod.concatenate([])


class TestInferVartype(unittest.TestCase):
    def test_array_ambiguous_all_1s(self):
        arr = np.ones((5, 5))
        self.assertIs(dimod.sampleset.infer_vartype(arr), None)

    def test_array_ambiguous_empty(self):
        arr = []
        self.assertIs(dimod.sampleset.infer_vartype(arr), None)

    def test_array_binary(self):
        arr = np.triu(np.ones((5, 5)))
        self.assertIs(dimod.sampleset.infer_vartype(arr), dimod.BINARY)

    def test_array_invalid(self):
        arr = [1, 2, 1]
        with self.assertRaises(ValueError):
            dimod.sampleset.infer_vartype(arr)

    def test_array_spin(self):
        arr = 2*np.triu(np.ones((5, 5)))-1
        self.assertIs(dimod.sampleset.infer_vartype(arr), dimod.SPIN)

    def test_sampleset_binary(self):
        ss = dimod.SampleSet.from_samples(([[1, 1], [0, 0]], 'ab'),
                                          dimod.BINARY, energy=[1, 1])
        self.assertIs(dimod.sampleset.infer_vartype(ss), dimod.BINARY)

    def test_sampleset_spin(self):
        ss = dimod.SampleSet.from_samples(([[1, 1], [-1, -1]], 'ab'),
                                          dimod.SPIN, energy=[1, 1])
        self.assertIs(dimod.sampleset.infer_vartype(ss), dimod.SPIN)
