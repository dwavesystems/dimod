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

import numpy as np

import dimod


try:
    import pandas as pd
    _pandas = True
except ImportError:
    _pandas = False


class TestSampleSet(unittest.TestCase):
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

    def test_eq_ordered(self):
        # samplesets should be equal regardless of variable order
        ss0 = dimod.SampleSet.from_samples(([-1, 1], 'ab'), dimod.SPIN, energy=0.0)
        ss1 = dimod.SampleSet.from_samples(([1, -1], 'ba'), dimod.SPIN, energy=0.0)
        ss2 = dimod.SampleSet.from_samples(([1, -1], 'ab'), dimod.SPIN, energy=0.0)
        ss3 = dimod.SampleSet.from_samples(([1, -1], 'ac'), dimod.SPIN, energy=0.0)

        self.assertEqual(ss0, ss1)
        self.assertNotEqual(ss0, ss2)
        self.assertNotEqual(ss1, ss3)

    def test_shorter_samples(self):
        ss = dimod.SampleSet.from_samples(np.ones((100, 5), dtype='int8'), dimod.BINARY, energy=np.ones(100))

        self.assertEqual(len(list(ss.samples(n=1))), 1)

    def test_from_samples_empty(self):

        self.assertEqual(len(dimod.SampleSet.from_samples([], dimod.SPIN, energy=[], a=1)), 0)

        self.assertEqual(len(dimod.SampleSet.from_samples({}, dimod.SPIN, energy=[], a=1)), 0)

        self.assertEqual(len(dimod.SampleSet.from_samples(np.empty((0, 0)), dimod.SPIN, energy=[], a=1)), 0)

    def test_from_samples_with_aggregation(self):
        samples = dimod.SampleSet.from_samples(([[-1, 1], [-1, 1]], 'ab'), dimod.SPIN, energy=[0.0, 0.0],
                                               aggregate_samples=True)
        self.assertEqual(samples.aggregate(),
                         dimod.SampleSet.from_samples(([-1, 1], 'ab'), dimod.SPIN, energy=0.0, num_occurrences=2))

    def test_aggregate_simple(self):
        samples = dimod.SampleSet.from_samples(([[-1, 1], [-1, 1]], 'ab'), dimod.SPIN, energy=[0.0, 0.0])

        self.assertEqual(samples.aggregate(),
                         dimod.SampleSet.from_samples(([-1, 1], 'ab'), dimod.SPIN, energy=0.0, num_occurrences=2))

        # original should not be changed
        self.assertEqual(samples,
                         dimod.SampleSet.from_samples(([[-1, 1], [-1, 1]], 'ab'), dimod.SPIN, energy=[0.0, 0.0]))


class TestSampleSetSerialization(unittest.TestCase):

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
class TestSampleSet_to_pandas_dataframe(unittest.TestCase):
    def test_simple(self):
        samples = dimod.SampleSet.from_samples(([[-1, 1, -1], [-1, -1, 1]], 'abc'),
                                               dimod.SPIN, energy=[-.5, .5])
        df = samples.to_pandas_dataframe()

        other = pd.DataFrame([[-1, 1, -1, -.5, 1], [-1, -1, 1, .5, 1]],
                             columns=['a', 'b', 'c', 'energy', 'num_occurrences'])

        pd.testing.assert_frame_equal(df, other, check_dtype=False)


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
