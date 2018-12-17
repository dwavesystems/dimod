# Copyright 2018 D-Wave Systems Inc.
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
# ================================================================================================

import unittest

from dimod.variables import Variables


class TestVariables(unittest.TestCase):
    def test_duplicates(self):
        # should have no duplicates
        variables = Variables(['a', 'b', 'c', 'b'])
        self.assertEqual(list(variables), ['a', 'b', 'c'])

    def test_iterable(self):
        variables = Variables('abcdef')
        self.assertEqual(list(variables), list('abcdef'))

    def test_index(self):
        variables = Variables(range(5))
        self.assertEqual(variables.index(4), 4)

    def test_count(self):
        variables = Variables([1, 1, 1, 4, 5])
        self.assertEqual(list(variables), [1, 4, 5])
        for v in range(10):
            if v in variables:
                self.assertEqual(variables.count(v), 1)
            else:
                self.assertEqual(variables.count(v), 0)

    def test_len(self):
        variables = Variables(range(5))
        self.assertEqual(len(variables), 5)
        variables = Variables('aaaaa')
        self.assertEqual(len(variables), 1)
