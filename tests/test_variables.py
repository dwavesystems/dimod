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
import copy
import decimal
import fractions
import itertools
import unittest
import unittest.mock

import numpy as np

from parameterized import parameterized_class

from dimod.variables import Variables


class TestAppend(unittest.TestCase):
    def test_conflict(self):
        variables = Variables()
        variables._append(1)
        variables._append()  # should take the label 0
        variables._append()

        self.assertEqual(variables, [1, 0, 2])


class TestConstruction(unittest.TestCase):
    @unittest.mock.patch("dimod.variables.Variables._append")
    def test_range(self, mock):
        # test that we bypass the append method

        class Boom(Exception):
            pass

        def boom(*args, **kwargs):
            raise Boom()

        mock.side_effect = boom

        times = []
        for n in [0, 1, 10, 100, 1000, 10000]:
            variables = Variables(range(n))
            self.assertEqual(variables, range(n))

        # test that the test works
        with self.assertRaises(Boom):
            Variables('abc')

    def test_range_negative(self):
        variables = Variables(range(-10))
        self.assertEqual(variables, [])
        self.assertEqual(variables, range(-10))

    def test_variables(self):
        v = Variables('abc')
        self.assertEqual(v, 'abc')
        self.assertEqual(Variables(v), v)


class TestCopy(unittest.TestCase):
    def test_copy(self):
        variables = Variables('abc')
        new = copy.copy(variables)
        variables._relabel({'a': 0})  # should not change the copy
        self.assertIsNot(new, variables)
        self.assertEqual(new, 'abc')
        self.assertIsInstance(new, Variables)

    def test_deepcopy_memo(self):
        variables = Variables('abc')
        new = copy.deepcopy([variables, variables])
        self.assertIs(new[0], new[1])
        self.assertIsNot(new[0], variables)
        self.assertIsInstance(new[0], Variables)


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


class TestGetItem(unittest.TestCase):
    def test_empty(self):
        variables = Variables()
        with self.assertRaises(IndexError):
            variables[0]
        with self.assertRaises(IndexError):
            variables[-1]

    def test_values(self):
        variables = Variables('abc')

        self.assertEqual(variables[0], 'a')
        self.assertEqual(variables[1], 'b')
        self.assertEqual(variables[2], 'c')
        with self.assertRaises(IndexError):
            variables[3]
        self.assertEqual(variables[-1], 'c')
        self.assertEqual(variables[-2], 'b')
        self.assertEqual(variables[-3], 'a')
        with self.assertRaises(IndexError):
            variables[-4]


class TestIndex(unittest.TestCase):
    def test_permissive(self):
        variables = Variables()

        with self.assertRaises(ValueError):
            variables.index(0)

        self.assertEqual(variables.index(0, permissive=True), 0)
        self.assertEqual(variables.index(0, permissive=True), 0)
        self.assertEqual(variables.index('a', permissive=True), 1)


class TestPop(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(IndexError):
            Variables()._pop()

    def test_simple(self):
        variables = Variables('abc')
        self.assertEqual(variables._pop(), 'c')
        self.assertEqual(variables, 'ab')

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
    def test_duplicate_target(self):
        # see https://github.com/dwavesystems/dimod/issues/1110
        variables = Variables('ab')
        with self.assertRaises(ValueError):
            variables._relabel({'a': 'c', 'b': 'c'})

    def test_permissive(self):
        variables = Variables([0, 1])

        # relabels a non-existant variable 2
        variables._relabel({0: 'a', 1: 'b', 2: 'c'})

        self.assertEqual(variables, Variables('ab'))

    def test_swap(self):
        variables = Variables([1, 0, 3, 4, 5])
        variables._relabel({5: 3, 3: 5})
        self.assertEqual(variables, [1, 0, 5, 4, 3])


class TestRemove(unittest.TestCase):
    def test_exceptions(self):
        variables = Variables([0, 1, 'a', 3, 4])

        with self.assertRaises(ValueError):
            variables._remove(2)

        with self.assertRaises(ValueError):
            variables._remove('hello')

    def test_last(self):
        variables = Variables(['i', 'x', 's'])
        variables._remove('s')
        self.assertEqual(variables, 'ix')

    def test_typical(self):
        variables = Variables(['i', 'x', 's'])
        variables._remove('i')
        self.assertEqual(variables, 'xs')

    def test_range(self):
        variables = Variables(range(10))
        variables._remove(3)
        self.assertEqual(variables, [0, 1, 2, 4, 5, 6, 7, 8, 9])


class TestSlice(unittest.TestCase):
    def test_slice(self):
        variables = Variables('abd')

        self.assertEqual(variables[:3], 'abd')
        self.assertEqual(variables[:1], 'a')
        self.assertEqual(variables[:], 'abd')
        self.assertIsInstance(variables[:2], Variables)
        self.assertEqual(variables[1::2], 'b')
        self.assertEqual(variables[::2], 'ad')


@parameterized_class(
    [dict(name='list', iterable=list(range(5))),
     dict(name='string', iterable='abcde'),
     dict(name='range', iterable=range(5)),
     dict(name='range_reversed', iterable=range(4, -1, -1)),
     dict(name='range_start', iterable=range(2, 7)),
     dict(name='range_step', iterable=range(0, 10, 2)),
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
