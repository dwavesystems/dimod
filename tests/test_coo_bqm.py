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

from dimod.bqm.coo_bqm import CooBQM


class TestFromIsing(unittest.TestCase):
    def test_empty(self):
        bqm = CooBQM.from_ising({}, {})

    def test_integer_labelled(self):
        bqm = CooBQM.from_ising({0: -1}, {(0, 1): 1})

        self.assertEqual(bqm.linear, {0: -1, 1: 0})
        self.assertEqual(bqm.quadratic, {(0, 1): 1})

    def test_str_labelled(self):
        bqm = CooBQM.from_ising({'a': -1}, {('a', 'b'): 1})

        self.assertEqual(bqm.linear, {'a': -1, 'b': 0})
        self.assertEqual(bqm.quadratic, {('a', 'b'): 1})
        self.assertEqual(bqm.quadratic, {('a', 'b'): 1})
