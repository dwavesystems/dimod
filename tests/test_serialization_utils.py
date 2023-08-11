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

import numbers
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

    def test_replacing_floats_with_ints(self):

        floating_dtypes = [np.float16, np.float32, np.float64]

        if int(np.__version__.split(".")[1]) >= 22:
            # Numpy<1.22.0 didn't support `is_integer()` on floating types
            # so float128 etc don't work out-of-the-box because `tolist()`
            # doesn't convert those to Python float.
            floating_dtypes.append(np.longdouble)

        for dtype in floating_dtypes:
            with self.subTest(f"{dtype}, all integer"):
                arr = np.ones(3, dtype=dtype)
                arr[0] = 2
                arr[1] = -0.0

                obj = serialize_ndarray(arr)

                # test the round trip
                new = deserialize_ndarray(obj)
                np.testing.assert_array_equal(arr, new)
                self.assertEqual(arr.dtype, new.dtype)  # original vartype is restored

                # test the ones that can be are mapped to int
                self.assertIsInstance(obj["data"][0], int)
                self.assertIsInstance(obj["data"][1], int)
                self.assertIsInstance(obj["data"][2], int)

            with self.subTest(f"{dtype}, all float"):
                arr = np.empty(3, dtype=dtype)
                arr[0] = 1.5
                arr[1] = float("inf")
                arr[2] = float("nan")

                obj = serialize_ndarray(arr)

                # test the round trip
                new = deserialize_ndarray(obj)
                np.testing.assert_array_equal(arr, new)
                self.assertEqual(arr.dtype, new.dtype)  # original vartype is restored

                # test the ones that can be are mapped to int
                self.assertIsInstance(obj["data"][0], numbers.Real)
                self.assertIsInstance(obj["data"][1], numbers.Real)
                self.assertIsInstance(obj["data"][2], numbers.Real)

            with self.subTest(f"{dtype}, mixed"):
                arr = np.ones(3, dtype=dtype)
                arr[0] = 1.5
                arr[1] = -0.0

                obj = serialize_ndarray(arr)

                # test the round trip
                new = deserialize_ndarray(obj)
                np.testing.assert_array_equal(arr, new)
                self.assertEqual(arr.dtype, new.dtype)  # original vartype is restored

                # test the ones that can be are mapped to int
                self.assertIsInstance(obj["data"][0], numbers.Real)
                self.assertIsInstance(obj["data"][1], int)
                self.assertIsInstance(obj["data"][2], int)

        with self.subTest("complex, mixed"):
            arr = np.ones(3, dtype=complex)
            arr[0] = 1.5
            arr[1] = -0.0

            obj = serialize_ndarray(arr)

            # test the round trip
            new = deserialize_ndarray(obj)
            np.testing.assert_array_equal(arr, new)
            self.assertEqual(arr.dtype, new.dtype)

            # in this case everything is kept as a complex number
            self.assertIsInstance(obj["data"][0], complex)
            self.assertIsInstance(obj["data"][1], complex)
            self.assertIsInstance(obj["data"][2], complex)

        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            with self.subTest(dtype):
                arr = np.empty(3, dtype=dtype)
                arr[0] = 2
                arr[1] = 0
                arr[2] = -1

                obj = serialize_ndarray(arr)

                # test the round trip
                new = deserialize_ndarray(obj)
                np.testing.assert_array_equal(arr, new)
                self.assertEqual(arr.dtype, new.dtype)  # original vartype is restored

                # test the ones that can be are mapped to int
                self.assertIsInstance(obj["data"][0], int)
                self.assertIsInstance(obj["data"][1], int)
                self.assertIsInstance(obj["data"][2], int)
