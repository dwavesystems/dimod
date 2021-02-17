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

import collections.abc as abc
import decimal
import fractions
import itertools
import unittest

import numpy as np

from parameterized import parameterized_class

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

    def test_unlike_types_eq_hash(self):
        zeros = [0, 0.0, np.int8(0), np.float64(0),
                 fractions.Fraction(0), decimal.Decimal(0)]
        for perm in itertools.permutations(zeros, len(zeros)):
            variables = Variables(perm)
            self.assertEqual(len(variables), len(set(zeros)))


class TestIndex(unittest.TestCase):
    def test_permissive(self):
        variables = Variables()

        with self.assertRaises(ValueError):
            variables.index(0)

        self.assertEqual(variables.index(0, permissive=True), 0)
        self.assertEqual(variables.index(0, permissive=True), 0)
        self.assertEqual(variables.index('a', permissive=True), 1)


class TestPrint(unittest.TestCase):
    def test_pprint(self):
        import pprint

        variables = Variables(range(10))
        variables._append('a')  # make not range

        string = pprint.pformat(variables, width=20)
        target = '\n'.join(
            ["Variables([0,",
             "           1,",
             "           2,",
             "           3,",
             "           4,",
             "           5,",
             "           6,",
             "           7,",
             "           8,",
             "           9,",
             "           'a'])"])
        self.assertEqual(string, target)

    def test_repr_empty(self):
        variables = Variables()
        self.assertEqual(repr(variables), 'Variables()')

    def test_repr_mixed(self):
        variables = Variables('abc')
        self.assertEqual(repr(variables), "Variables(['a', 'b', 'c'])")

    def test_repr_range(self):
        self.assertEqual(repr(Variables(range(10))),
                         'Variables({!r})'.format(list(range(10))))
        self.assertEqual(repr(Variables(range(11))), 'Variables(range(0, 11))')


class TestRelabel(unittest.TestCase):
    def test_permissive(self):
        variables = Variables([0, 1])

        # relabels a non-existant variable 2
        variables._relabel({0: 'a', 1: 'b', 2: 'c'})

        self.assertEqual(variables, Variables('ab'))


@parameterized_class(
    [dict(name='list', iterable=list(range(5))),
     dict(name='string', iterable='abcde'),
     dict(name='range', iterable=range(5)),
     dict(name='mixed', iterable=[0, ('b',), 2.1, 'c', frozenset('d')]),
     dict(name='floats', iterable=[0., 1., 2., 3., 4.]),
     ],
    class_name_func=lambda cls, i, inpt: '%s_%s' % (cls.__name__, inpt['name'])
    )
class TestIterable(unittest.TestCase):

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

        variables._relabel(mapping)

        self.assertEqual(variables, target)

    def test_relabel_not_hashable(self):
        variables = Variables(self.iterable)
        mapping = {v: [v] for v in variables}
        with self.assertRaises(ValueError):
            variables._relabel(mapping)
