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

from dimod.compatibility23 import SortKey


class TestSortKey(unittest.TestCase):
    def test_range(self):
        self.assertLess(SortKey(range(0, 3)), SortKey(range(2, 4)))

    def test_tuple_list(self):
        self.assertLess(SortKey([0, 2]), SortKey((0, 2)))

    def test_nested(self):
        s0 = [[0, 1], 1]
        s1 = [(0, 1), 1]
        self.assertLess(SortKey(s0), SortKey(s1))
