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

try:
    import bson
except ImportError:
    _bson_imported = False
else:
    _bson_imported = True

from dimod.serialization.utils import array2bytes, bytes2array


@unittest.skipUnless(_bson_imported, "no pymongo bson installed")
class TestBson(unittest.TestCase):
    def test_functional_bson_binary(self):
        arr = np.ones((5, 5))

        b = array2bytes(arr, bytes_type=bson.Binary)

        np.testing.assert_array_equal(arr, bytes2array(b))
