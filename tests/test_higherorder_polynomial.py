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
# ============================================================================
import itertools
import unittest

import numpy as np

import dimod
from dimod.higherorder import BinaryPolynomial


class TestConstruction(unittest.TestCase):
    # just that things don't fall down, we'll test correctness when
    # testing other attributes
    def test_from_dict(self):
        BinaryPolynomial({'a': -1, tuple(): 1.3, 'bc': -1, ('a', 'b'): 1}, 'SPIN')

    def test_from_iterator(self):
        BinaryPolynomial(((term, -1) for term in itertools.combinations(range(100), 2)), 'SPIN')

    def test_aggregation(self):
        poly = BinaryPolynomial({'ab': 1, 'ba': 1, ('a', 'b'): 1, ('b', 'a'): 1}, 'SPIN')
        self.assertEqual(poly, BinaryPolynomial({'ab': 4}, 'SPIN'))


class Test__contains__(unittest.TestCase):
    def test_single_term(self):
        poly = BinaryPolynomial({('a', 'b'): 1}, 'SPIN')
        self.assertIn('ab', poly)
        self.assertIn('ba', poly)
        self.assertIn(('a', 'b'), poly)
        self.assertIn(('b', 'a'), poly)


class Test__len__(unittest.TestCase):
    def test_single_term(self):
        poly = BinaryPolynomial({('a', 'b'): 1}, 'SPIN')
        self.assertEqual(len(poly), 1)

    def test_repeated_term(self):
        poly = BinaryPolynomial({('a', 'b'): 1, 'ba': 1}, 'BINARY')
        self.assertEqual(len(poly), 1)


class Test__getitems__(unittest.TestCase):
    def test_repeated_term(self):
        poly = BinaryPolynomial({'ab': 1, 'ba': 1, ('a', 'b'): 1, ('b', 'a'): 1}, 'BINARY')
        self.assertEqual(poly['ab'], 4)


class Test_energies(unittest.TestCase):
    def test_single_variable(self):
        poly = BinaryPolynomial({'a': -1}, 'SPIN')

        energies = poly.energies(([[-1], [1]], ['a']))
        np.testing.assert_array_equal(energies, [1, -1])
