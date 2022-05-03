# Copyright 2022 D-Wave Systems Inc.
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

import os
import unittest

import dimod

from dimod.serialization.lp import read_lp_file, read_lp

data_dir = os.path.join(os.path.dirname(__file__), 'data', 'lp')


class TestObjective(unittest.TestCase):
    def test_complex(self):
        lp = """
        minimize
            a + 2 b + [ a^2 + 4 a * b + 7 b^2 ]/2 + 5
        Binary
            a b
        End
        """
        a, b = dimod.Binaries('ab')

        cqm = read_lp(lp)

        self.assertFalse(cqm.constraints)
        self.assertTrue(cqm.objective.is_equal(a + 2*b + (a**2 + 4*a*b + 7*b**2) / 2 + 5))

    def test_linear(self):
        lp = """
        minimize
            obj: x + y
        End
        """

        x, y = dimod.Reals('xy')

        cqm = read_lp(lp)

        self.assertFalse(cqm.constraints)
        self.assertTrue(cqm.objective.is_equal(x + y))

    def test_quadratic(self):
        lp = """
        minimize
            obj: [ x * y] / 2
        Binary
            x y
        End
        """

        x, y = dimod.Binaries('xy')

        cqm = read_lp(lp)

        self.assertFalse(cqm.constraints)
        self.assertTrue(cqm.objective.is_equal(x * y / 2))

    def test_quadratic_nospace(self):
        lp = """
        minimize
            obj: [x*y]/2
        Binary
            x y
        End
        """

        x, y = dimod.Binaries('xy')

        cqm = read_lp(lp)

        self.assertFalse(cqm.constraints)
        self.assertTrue(cqm.objective.is_equal(x * y / 2))

    def test_unlabelled(self):
        lp = """
        minimize
            x + y
        End
        """

        x, y = dimod.Reals('xy')

        cqm = read_lp(lp)

        self.assertFalse(cqm.constraints)
        self.assertTrue(cqm.objective.is_equal(x + y))

    # todo
    # def test_scientific_notation(self):
    #     lp = """
    #     Minimize
    #       obj: 2e3 x0 + 2.1e-04 x0^2
    #     Subject To
    #       x0 <= 1
    #     General
    #       x0
    #     End
    #     """

    #     x0 = dimod.Integer('i0')

    #     cqm = read_lp(lp)


class TestReadLp(unittest.TestCase):
    """Test the different APIs"""

    lp = b"""
    Minimize
     obj: x0 + x1 + 3 x2

    Subject To
     c1: x0 + x2 <= 9

    Binary
     x0 x1 x2

    End
    """

    def assert_cqm(self, cqm):
        x0, x1, x2 = dimod.Binaries(['x0', 'x1', 'x2'])

        objective = x0 + x1 + 3*x2
        c1 = x0 + x2 <= 9

        self.assertEqual(objective.linear, cqm.objective.linear)
        self.assertEqual(objective.quadratic, cqm.objective.quadratic)
        self.assertEqual(objective.offset, cqm.objective.offset)
        for v in cqm.variables:
            self.assertIs(cqm.objective.vartype(v), dimod.BINARY)

    def test_bytes(self):
        self.assert_cqm(read_lp(self.lp))

    def test_file_like(self):
        with open(os.path.join(data_dir, '3variable_1constraint_linear.lp'), 'rb') as f:
            self.assert_cqm(read_lp(f))

    def test_string(self):
        self.assert_cqm(read_lp(self.lp.decode()))
