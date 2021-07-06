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

import unittest

import numpy as np

from dimod.views.samples import SamplesArray


class TestSamplesArray(unittest.TestCase):
    # SampleSet.samples
    def setUp(self):
        self.s3x3 = SamplesArray(np.triu(np.ones((3, 3), dtype='int8')),
                                 ['a', 0, ('b', 1)])

    def test_singlerow(self):
        self.assertEqual(self.s3x3[0], {'a': 1, 0: 1, ('b', 1): 1})
        self.assertEqual(self.s3x3[1], {'a': 0, 0: 1, ('b', 1): 1})
        self.assertEqual(self.s3x3[2], {'a': 0, 0: 0, ('b', 1): 1})

    def test_multirow_slice(self):
        new = self.s3x3[:]

        np.testing.assert_array_equal(new._samples, self.s3x3._samples)
        self.assertEqual(new._variables, self.s3x3._variables)

    def test_multirow_slice_partial(self):
        new = self.s3x3[1:]

        np.testing.assert_array_equal(new._samples, [[0, 1, 1], [0, 0, 1]])
        self.assertEqual(new._variables, self.s3x3._variables)

    def test_multirow_index_array(self):
        new = self.s3x3[[0, 2]]

        np.testing.assert_array_equal(new._samples, [[1, 1, 1], [0, 0, 1]])
        self.assertEqual(new._variables, self.s3x3._variables)

    def test_too_many_indices(self):
        with self.assertRaises(IndexError):
            self.s3x3[0, 1, 2]

    def test_singlerow_single_column(self):
        self.assertEqual(self.s3x3[0, 'a'], 1)
        self.assertEqual(self.s3x3[1, 'a'], 0)
        self.assertEqual(self.s3x3[0, 0], 1)
        self.assertEqual(self.s3x3[2, 0], 0)
        self.assertEqual(self.s3x3[0, ('b', 1)], 1)

    def test_multirow_singlecolumn(self):
        np.testing.assert_array_equal(self.s3x3[:, 'a'], [1, 0, 0])
        np.testing.assert_array_equal(self.s3x3[1:, 'a'], [0, 0])
        np.testing.assert_array_equal(self.s3x3[[0, 2], 0], [1, 0])

    def test_singlerow_multicolumn(self):
        np.testing.assert_array_equal(self.s3x3[1, ['a', 0]], [0, 1])

    def test_multirow_multicolumn(self):
        np.testing.assert_array_equal(self.s3x3[:, ['a', 0]], [[1, 1], [0, 1], [0, 0]])
        np.testing.assert_array_equal(self.s3x3[1:, ['a', 0]], [[0, 1], [0, 0]])
        np.testing.assert_array_equal(self.s3x3[0:3:2, ['a', 0]], [[1, 1], [0, 0]])
        np.testing.assert_array_equal(self.s3x3[[0, 2], ['a', 0]], [[1, 1], [0, 0]])

    def test_column_miss(self):
        with self.assertRaises(KeyError):
            self.s3x3[0, 'c']
        with self.assertRaises(KeyError):
            self.s3x3[0:2, ['a', 'c']]
        with self.assertRaises(KeyError):
            self.s3x3[0, 3]

    def test_column_slice(self):
        with self.assertRaises(KeyError):
            self.s3x3[0, :]
