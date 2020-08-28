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
#
# =============================================================================
import unittest

import numpy as np

from dimod.bqm.adjdictbqm import AdjDictBQM


class TestObjectDtype(unittest.TestCase):
    # AdjDictBQM has an object dtype so it has some special cases that need
    # to be tested

    def test_dtypes_array_like_ints(self):
        # these should stay as python ints
        obj = [[0, 1], [1, 2]]

        bqm = AdjDictBQM(obj, 'BINARY')

        for _, bias in bqm.quadratic.items():
            self.assertIsInstance(bias, int)

    def test_dtypes_ndarray_ints(self):
        # these should stay as python ints
        obj = np.asarray([[0, 1], [1, 2]], dtype=np.int32)

        bqm = AdjDictBQM(obj, 'BINARY')

        for _, bias in bqm.quadratic.items():
            self.assertIsInstance(bias, np.int32)
