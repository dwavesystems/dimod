# Copyright 2021 D-Wave Systems Inc.
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

import itertools
import unittest

import dimod


class TestComparison(unittest.TestCase):
    def test_repr(self):
        x, y = dimod.Binaries('xy')
        i, = dimod.Integers('i')

        self.assertEqual(repr(x + y <= 5), "Le(BinaryQuadraticModel({'x': 1.0, 'y': 1.0}, {}, 0.0, 'BINARY'), 5)")
        self.assertEqual(repr(x + y == 5), "Eq(BinaryQuadraticModel({'x': 1.0, 'y': 1.0}, {}, 0.0, 'BINARY'), 5)")
        self.assertEqual(repr(x + y >= 5), "Ge(BinaryQuadraticModel({'x': 1.0, 'y': 1.0}, {}, 0.0, 'BINARY'), 5)")
        self.assertEqual(repr(x + i <= 5), "Le(QuadraticModel({'x': 1.0, 'i': 1.0}, {}, 0.0, {'x': 'BINARY', 'i': 'INTEGER'}, dtype='float64'), 5)")
        self.assertEqual(repr(x + i == 5), "Eq(QuadraticModel({'x': 1.0, 'i': 1.0}, {}, 0.0, {'x': 'BINARY', 'i': 'INTEGER'}, dtype='float64'), 5)")
        self.assertEqual(repr(x + i >= 5), "Ge(QuadraticModel({'x': 1.0, 'i': 1.0}, {}, 0.0, {'x': 'BINARY', 'i': 'INTEGER'}, dtype='float64'), 5)")

    def test_str(self):
        x, y = dimod.Binaries('xy')
        i, = dimod.Integers('i')

        self.assertEqual(str(x + y <= 5), "x + y <= 5")
        self.assertEqual(str(x + y == 5), "x + y == 5")
        self.assertEqual(str(x + y >= 5), "x + y >= 5")
        self.assertEqual(str(x + i <= 5), "x + i <= 5")
        self.assertEqual(str(x + i == 5), "x + i == 5")
        self.assertEqual(str(x + i >= 5), "x + i >= 5")

    def test_to_polystring(self):
        x, y = dimod.Binaries('xy')
        i, = dimod.Integers('i')

        self.assertEqual((x + y <= 5).to_polystring(), "x + y <= 5")
        self.assertEqual((x + y == 5).to_polystring(), "x + y == 5")
        self.assertEqual((x + y >= 5).to_polystring(), "x + y >= 5")
        self.assertEqual((x + i <= 5).to_polystring(), "x + i <= 5")
        self.assertEqual((x + i == 5).to_polystring(), "x + i == 5")
        self.assertEqual((x + i >= 5).to_polystring(), "x + i >= 5")

        self.assertEqual((x + y <= 5).to_polystring(lambda v: f'T{v}'), "Tx + Ty <= 5")


class TestExpressions(unittest.TestCase):
    def test_add_permutations(self):
        x = dimod.Binary('x')
        i = dimod.Integer('i')
        s = dimod.Spin('s')

        for perm in itertools.permutations([x, i, s, 1]):
            qm = sum(perm)

            self.assertEqual(qm.linear, {'x': 1, 'i': 1, 's': 1})
            self.assertEqual(qm.offset, 1)
            self.assertEqual(qm.quadratic, {})

    def test_expressions(self):
        i = dimod.Integer('i')
        j = dimod.Integer('j')

        qm = (i - 1)*(j - 1)

        self.assertEqual(qm.linear, {'i': -1, 'j': -1})
        self.assertEqual(qm.quadratic, {('i', 'j'): 1})
        self.assertEqual(qm.offset, 1)

    def test_mul_permutations(self):
        x = dimod.Binary('x')
        i = dimod.Integer('i')
        s = dimod.Spin('s')

        for t0, t1 in itertools.permutations([x, i, s, 1], 2):
            qm = t0*t1

    def test_sub_permutations(self):
        x = dimod.Binary('x')
        i = dimod.Integer('i')
        s = dimod.Spin('s')

        for t0, t1 in itertools.permutations([x, i, s, 1], 2):
            qm = t0 - t1


class QuickSum(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(dimod.quicksum([]), 0)

    def test_promotion(self):
        x = dimod.Binary('x')
        i = dimod.Integer('i')
        s = dimod.Spin('s')

        for perm in itertools.permutations([2*x, i, s, 1]):
            qm = dimod.quicksum(perm)

            self.assertEqual(qm.linear, {'x': 2, 'i': 1, 's': 1})
            self.assertEqual(qm.offset, 1)
            self.assertEqual(qm.quadratic, {})

    def test_copy(self):
        x = dimod.Binary('x')

        newx = dimod.quicksum([x])

        self.assertIsNot(newx, x)
