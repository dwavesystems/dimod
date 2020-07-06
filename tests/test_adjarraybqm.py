# Copyright 2020 D-Wave Systems Inc.
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
import unittest

import numpy as np

from dimod import AdjArrayBQM


class TestData(unittest.TestCase):

    def test_empty(self):
        bqm = AdjArrayBQM('SPIN')

        data = bqm.data

        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 5)

        nstarts, lbiases, neighborhoods, qbiases, offset = map(np.asarray, data)

        np.testing.assert_array_equal(nstarts, [])
        np.testing.assert_array_equal(lbiases, [])
        np.testing.assert_array_equal(neighborhoods, [])
        np.testing.assert_array_equal(qbiases, [])
        np.testing.assert_array_equal(offset, [0])

        new = AdjArrayBQM.from_data(data, 'SPIN')
        self.assertEqual(bqm, new)

    def test_scalar(self):
        bqm = AdjArrayBQM(1, 'SPIN')

        data = bqm.data

        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 5)
        self.assertTrue(all(isinstance(buff, memoryview) for buff in data))

        new = AdjArrayBQM.from_data(data, 'SPIN')
        self.assertEqual(bqm, new)

    def test_square(self):
        bqm = AdjArrayBQM(np.arange(5), np.arange(25).reshape((5, 5)), 'SPIN')

        data = bqm.data

        self.assertIsInstance(data, tuple)
        self.assertEqual(len(data), 5)
        self.assertTrue(all(isinstance(buff, memoryview) for buff in data))

        new = AdjArrayBQM.from_data(data, 'SPIN')
        self.assertEqual(bqm, new)
