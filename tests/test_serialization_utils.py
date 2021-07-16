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

from dimod.serialization.utils import serialize_ndarray, deserialize_ndarray


class TestNdArraySerialization(unittest.TestCase):
    def test_functional_empty(self):
        arr = np.empty((0, 0))

        obj = serialize_ndarray(arr)
        new = deserialize_ndarray(obj)
        np.testing.assert_array_equal(arr, new)
        self.assertEqual(arr.dtype, new.dtype)

        obj = serialize_ndarray(arr, use_bytes=True)
        self.assertIsInstance(obj['data'], bytes)
        new = deserialize_ndarray(obj)
        np.testing.assert_array_equal(arr, new)
        self.assertEqual(arr.dtype, new.dtype)

    def test_functional_3x3triu(self):
        arr = np.triu(np.ones((3, 3)))

        obj = serialize_ndarray(arr)
        new = deserialize_ndarray(obj)
        np.testing.assert_array_equal(arr, new)
        self.assertEqual(arr.dtype, new.dtype)

        obj = serialize_ndarray(arr, use_bytes=True)
        self.assertIsInstance(obj['data'], bytes)
        new = deserialize_ndarray(obj)
        np.testing.assert_array_equal(arr, new)
        self.assertEqual(arr.dtype, new.dtype)
