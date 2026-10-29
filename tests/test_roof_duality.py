# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import dimod

from dwave.preprocessing.lower_bounds import roof_duality

class TestRoofDuality(unittest.TestCase):
    def test_empty(self):
        bqm = dimod.BinaryQuadraticModel('BINARY')
        lb, fixed = roof_duality(bqm, strict=True)
        self.assertEqual(fixed, {})
        self.assertEqual(lb, 0.0)

        lb, fixed = roof_duality(bqm, strict=False)
        self.assertEqual(fixed, {})
        self.assertEqual(lb, 0.0)

    def test_all_zero(self):
        num_vars = 3

        bqm = dimod.BinaryQuadraticModel(num_vars, 'BINARY')
        lb, fixed = roof_duality(bqm, strict=True)
        self.assertEqual(fixed, {})
        self.assertEqual(lb, 0.0)

        lb, fixed = roof_duality(bqm, strict=False)
        self.assertEqual(len(fixed), num_vars)
        for val in fixed.values():
            self.assertEqual(val, 1)
        self.assertEqual(lb, 0.0)

    def test_3path(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 10}, {'ab': -1, 'bc': 1})
        lb, fixed = roof_duality(bqm)
        self.assertEqual(fixed, {'a': -1, 'b': -1, 'c': 1})
        self.assertEqual(lb, -12.0)

    def test_object(self):
        bqm = dimod.DictBQM.from_ising({'a': 10}, {'ab': -1, 'bc': 1})
        lb, fixed = roof_duality(bqm)
        self.assertEqual(fixed, {'a': -1, 'b': -1, 'c': 1})
        self.assertEqual(lb, -12.0)
