# Copyright 2021 D-Wave Systems Inc.
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
import parameterized

import dimod

from dimod.cyutilities import coo_sort


class TestCOOSort(unittest.TestCase):
    def test_row(self):
        row = np.asarray([3, 2, 1], dtype=int)
        col = np.asarray([1, 2, 3], dtype=int)
        data = np.asarray([13, 22, 13], dtype=float)

        coo_sort(row, col, data)

        np.testing.assert_array_equal(row, [1, 1, 2])
        np.testing.assert_array_equal(col, [3, 3, 2])
        np.testing.assert_array_equal(data, [13, 13, 22])

    def test_random(self):
        rng = np.random.default_rng(42)

        row = np.arange(100, dtype=np.int64)
        col = np.arange(100, dtype=np.int64)

        for _ in range(10):
            rng.shuffle(row)
            rng.shuffle(col)

            data = np.asarray(row*col, dtype=float)

            coo_sort(row, col, data)

            pairs = list(zip(row, col))
            for i in range(len(pairs) - 1):
                self.assertLessEqual(pairs[i], pairs[i+1])
            np.testing.assert_array_equal(row * col, data)


class TestVartypeInfo(unittest.TestCase):
    @parameterized.parameterized.expand([(np.float32,), (np.float64,)])
    def test_binary(self, dtype):
        info = dimod.vartype_info('BINARY', dtype=dtype)
        self.assertEqual(info.default_min, 0)
        self.assertEqual(info.default_max, 1)
        self.assertEqual(info.min, 0)
        self.assertEqual(info.max, 1)

    def test_integer(self):
        with self.subTest('float32'):
            info = dimod.vartype_info('INTEGER', dtype=np.float32)
            self.assertEqual(info.default_min, 0)
            self.assertEqual(info.default_max, 16777215)
            self.assertEqual(info.min, -16777215)
            self.assertEqual(info.max, 16777215)

        with self.subTest('float64'):
            info = dimod.vartype_info('INTEGER', dtype=np.float64)
            self.assertEqual(info.default_min, 0)
            self.assertEqual(info.default_max, 9007199254740991)
            self.assertEqual(info.min, -9007199254740991)
            self.assertEqual(info.max, 9007199254740991)

        with self.subTest('default'):
            info = dimod.vartype_info('INTEGER')
            self.assertEqual(info.default_min, 0)
            self.assertEqual(info.default_max, 9007199254740991)
            self.assertEqual(info.min, -9007199254740991)
            self.assertEqual(info.max, 9007199254740991)

    @parameterized.parameterized.expand([(np.float32,), (np.float64,)])
    def test_spin(self, dtype):
        info = dimod.vartype_info('SPIN', dtype=dtype)
        self.assertEqual(info.default_min, -1)
        self.assertEqual(info.default_max, 1)
        self.assertEqual(info.min, -1)
        self.assertEqual(info.max, 1)
