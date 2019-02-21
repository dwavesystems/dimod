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
# ================================================================================================
import unittest
import json

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

    def test_copy_false(self):
        samples_like = np.ones((5, 5))
        labels = list('abcde')
        arr, lab = dimod.as_samples((samples_like, labels))
        np.testing.assert_array_equal(arr, np.ones((5, 5)))
        self.assertEqual(lab, list('abcde'))
        self.assertIs(labels, lab)
        self.assertTrue(np.shares_memory(arr, samples_like))

    def test_dict_with_inconsistent_labels(self):
        with self.assertRaises(ValueError):
            dimod.as_samples(({'a': -1}, 'b'))

    def test_dict_with_labels(self):
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

    def test_iterator(self):
        with self.assertRaises(TypeError):
            dimod.as_samples(([-1] for _ in range(10)))

    def test_iterator_labelled(self):
        with self.assertRaises(TypeError):
            dimod.as_samples(([-1] for _ in range(10)), 'a')

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

        # ss0.samples() is an iterator, so let's just use that
        with self.assertRaises(TypeError):
            dimod.SampleSet.from_samples(ss0.samples(), dimod.BINARY, energy=ss0.record.energy)

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


class TestAggregate(unittest.TestCase):
    def test_aggregate_simple(self):
        samples = dimod.SampleSet.from_samples(([[-1, 1], [-1, 1]], 'ab'), dimod.SPIN, energy=[0.0, 0.0])

        self.assertEqual(samples.aggregate(),
                         dimod.SampleSet.from_samples(([-1, 1], 'ab'), dimod.SPIN, energy=0.0, num_occurrences=2))

        # original should not be changed
        self.assertEqual(samples,
                         dimod.SampleSet.from_samples(([[-1, 1], [-1, 1]], 'ab'), dimod.SPIN, energy=[0.0, 0.0]))


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


class TestIteration(unittest.TestCase):
    def test_data_reverse(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1})
        sampleset = dimod.SampleSet.from_samples_bqm([{'a': -1, 'b': 1}, {'a': 1, 'b': 1}], bqm)

        samples = list(sampleset.data())
        reversed_samples = list(sampleset.data(reverse=True))
        self.assertEqual(samples, list(reversed(reversed_samples)))


class TestSerialization(unittest.TestCase):
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
