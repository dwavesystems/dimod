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

import io
import os
import unittest

import dimod

from dimod.lp import load, loads

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

        cqm = loads(lp)

        self.assertFalse(cqm.constraints)
        self.assertTrue(cqm.objective.is_equal(a + 2*b + (a**2 + 4*a*b + 7*b**2) / 2 + 5))

    def test_doc(self):
        lp = """
        Minimize
            x0 - 2 x1
        Subject To
            x0 + x1 = 1
        Binary
            x0 x1
        End
        """

        cqm = loads(lp)

    def test_linear(self):
        lp = """
        minimize
            obj: x + y
        End
        """

        x, y = dimod.Reals('xy')

        cqm = loads(lp)

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

        cqm = loads(lp)

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

        cqm = loads(lp)

        self.assertFalse(cqm.constraints)
        self.assertTrue(cqm.objective.is_equal(x * y / 2))

    def test_unlabelled(self):
        lp = """
        minimize
            x + y
        End
        """

        x, y = dimod.Reals('xy')

        cqm = loads(lp)

        self.assertFalse(cqm.constraints)
        self.assertTrue(cqm.objective.is_equal(x + y))

    def test_scientific_notation(self):
        lp = """
        Minimize
          obj: 2e3 x0 + [4.1e-02 x0*x0]/2
        Subject To
          x0 <= 1
        General
          x0
        End
        """
        cqm = loads(lp)

        x0 = dimod.Integer('x0')

        self.assertTrue(cqm.objective.is_equal(2e3 * x0 + (4.1e-2 * x0 * x0) / 2))