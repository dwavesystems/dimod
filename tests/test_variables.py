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
# =============================================================================
import unittest

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from dimod.variables import Variables


class TestDuplicates(unittest.TestCase):
    def test_duplicates(self):
        # should have no duplicates
        variables = Variables(['a', 'b', 'c', 'b'])
        self.assertEqual(list(variables), ['a', 'b', 'c'])

    def test_count(self):
        variables = Variables([1, 1, 1, 4, 5])
        self.assertEqual(list(variables), [1, 4, 5])
        for v in range(10):
            if v in variables:
                self.assertEqual(variables.count(v), 1)
            else:
                self.assertEqual(variables.count(v), 0)

    def test_len(self):
        variables = Variables('aaaaa')
        self.assertEqual(len(variables), 1)


class TestRelabel(unittest.TestCase):
    def test_permissive(self):
        variables = Variables([0, 1])

        # relabels a non-existant variable 2
        variables.relabel({0: 'a', 1: 'b', 2: 'c'})

        self.assertEqual(variables, Variables('ab'))


class TestList(unittest.TestCase):
    iterable = list(range(5))

    def test_index_api(self):
        variables = Variables(self.iterable)
        self.assertTrue(hasattr(variables, 'index'))
        self.assertTrue(callable(variables.index))
        self.assertTrue(isinstance(variables.index, abc.Mapping))

    def test_contains_unhashable(self):
        variables = Variables(self.iterable)
        self.assertFalse([] in variables)

    def test_count_unhashable(self):
        variables = Variables(self.iterable)
        self.assertEqual(variables.count([]), 0)

    def test_index(self):
        variables = Variables(self.iterable)
        for idx, v in enumerate(self.iterable):
            self.assertEqual(variables.index(v), idx)

    def test_iterable(self):
        variables = Variables(self.iterable)
        self.assertEqual(list(variables), list(self.iterable))

    def test_equality(self):
        variables = Variables(self.iterable)
        self.assertEqual(variables, self.iterable)

    def test_len(self):
        variables = Variables(self.iterable)
        self.assertEqual(len(variables), len(self.iterable))

    def test_relabel_conflict(self):
        variables = Variables(self.iterable)

        iterable = self.iterable

        # want a relabelling with identity relabels and that maps to the same
        # set of labels as the original
        target = [iterable[-i] for i in range(len(iterable))]

        mapping = dict(zip(iterable, target))

        variables.relabel(mapping)

        self.assertEqual(variables, target)

    def test_relabel_not_hashable(self):
        variables = Variables(self.iterable)
        mapping = {v: [v] for v in variables}
        with self.assertRaises(ValueError):
            variables.relabel(mapping)


class TestMixed(TestList):
    # misc hashable objects
    iterable = [0, ('b',), 2.1, 'c', frozenset('d')]


class TestRange(TestList):
    iterable = range(5)


class TestString(TestList):
    iterable = 'abcde'
