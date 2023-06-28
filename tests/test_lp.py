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

import numpy as np

import dimod

from dimod import Binary, BQM, CQM, Integer, Real
from dimod.lp import load, loads
from parameterized import parameterized

data_dir = os.path.join(os.path.dirname(__file__), 'data', 'lp')

LP_BAD_LABELS = [['😜'], ['x*y'], ['a+b'], ['e9'], ['E-24'], ['E8cats'], ['eels'], ['example'],
                 [()], [frozenset()], ['.x'], ['0y'], [""], ['\\'], ['x' * 256], [b'x'], ['π']]
LP_ODD_LABELS = [['$'], ['#'], ['%'], ['&'], ['"']]
LP_TEST_VALUES = [[1e30], [1e-30], [-1e30], [-1e-30], [10], [-10], [0.1], [-0.1]]
LP_MAX_LINE_LEN = 560


class TestLoads(unittest.TestCase):
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

    # Developer note: It would be nice if this failed, but the LP reader that
    # we're wrapping doesn't raise an error in this case but rather ignores
    # it, so we propagate the behavior.
    # def test_invalid_optimization_sense(self):
    #     lp = """
    #     invalid
    #         obj: x + y
    #     End
    #     """

    #     x, y = dimod.Reals('xy')

    #     with self.assertRaises(ValueError):
    #         cqm = loads(lp)

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

    def test_maximize(self):
        lp = """
        Maximize
            x0 - 2 x1 + [ 6 x0*x1 ] / 2
        Subject To
            x0 + x1 = 1
        Binary
            x0 x1
        End
        """

        x0, x1 = dimod.Binaries(['x0', 'x1'])

        cqm = loads(lp)

        self.assertTrue(cqm.objective.is_equal(-x0 + 2*x1 - 3*x0*x1))
        self.assertTrue(cqm.constraints[0].lhs.is_equal(x0 + x1))
        self.assertEqual(cqm.constraints[0].rhs, 1)
        self.assertIs(cqm.constraints[0].sense, dimod.sym.Sense.Eq)

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


class TestDumps(unittest.TestCase):

    def _assert_cqms_are_equivalent(self, cqm, new):
        self.assertEqual({v: cqm.vartype(v) for v in cqm.variables},
                         {v: cqm.vartype(v) for v in new.variables})
        self.assertTrue(cqm.objective.is_equal(new.objective))
        self.assertEqual(set(cqm.constraints), set(new.constraints))

        for key in cqm.constraints:
            a = cqm.constraints[key]
            b = new.constraints[key]
            self.assertEqual(a.sense, b.sense)

            if a.lhs.dtype == np.float32:
                self.assertTrue((a.lhs - a.rhs).is_almost_equal(b.lhs - b.rhs))
            else:
                self.assertTrue((a.lhs - a.rhs).is_equal(b.lhs - b.rhs))

    def test_functional(self):
        cqm = CQM()
        bqm = BQM({'a': -1}, {'ab': 1}, 1.5, 'BINARY')
        cqm.add_constraint(bqm, '<=', label='c0')
        cqm.add_constraint(bqm, '>=', label='c1')
        cqm.set_objective(BQM({'c': -1}, {}, 'BINARY'))
        cqm.add_constraint(Binary('a')*Integer('d')*5 == 3, label='c2')

        new = dimod.lp.loads(dimod.lp.dumps(cqm))

        self._assert_cqms_are_equivalent(cqm, new)

    def test_empty_model(self):
        cqm = CQM()

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    def test_no_constraints(self):
        cqm = CQM()
        cqm.set_objective(BQM({'a': -1, 'b': -1}, {'ab': 1}, 'BINARY'))

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    def test_no_objective(self):
        cqm = CQM()
        bqm = BQM({'a': -1}, {'ab': 1}, 1.5, 'BINARY')
        cqm.add_constraint(bqm, '<=', label='c0')
        cqm.add_constraint(bqm, '>=', label='c1')
        cqm.add_constraint(Binary('a') * Integer('d') * 5 <= 3, label='c2')

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    def test_quadratic_objective(self):
        cqm = CQM()
        cqm.set_objective(BQM({}, {'ab': 1}, 'BINARY'))

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    @parameterized.expand(LP_TEST_VALUES)
    def test_objective_offset(self, value):
        cqm = CQM()
        cqm.set_objective(BQM({}, {'ab': 1}, value, 'BINARY'))

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    def test_spin_variables(self):
        cqm = CQM()
        cqm.set_objective(BQM({}, {'ab': 1}, 'SPIN'))

        with self.assertRaises(ValueError):
            dimod.lp.dumps(cqm)

    @parameterized.expand(LP_BAD_LABELS)
    def test_bad_variable_labels(self, label):
        cqm = CQM()
        cqm.set_objective(BQM({label: 1}, {}, 'BINARY'))

        with self.assertRaises(ValueError):
            dimod.lp.dumps(cqm)

    @parameterized.expand(LP_ODD_LABELS)
    def test_odd_variable_labels_no_constraints(self, label):
        cqm = CQM()
        cqm.set_objective(BQM({label: 1}, {}, 'BINARY'))

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    @parameterized.expand(LP_ODD_LABELS)
    def test_odd_variable_labels_with_constraints(self, label):
        cqm = CQM()
        cqm.set_objective(BQM({label: 1}, {}, 'BINARY'))
        bqm = BQM({'a': -1}, {('a', label): 1}, 1.5, 'BINARY')
        cqm.add_constraint(bqm, '<=', label='c0')

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    @parameterized.expand(LP_BAD_LABELS)
    def test_bad_constraint_labels(self, label):
        cqm = CQM()
        cqm.add_constraint(Binary('a') + Binary('b') == 1, label=label)

        with self.assertRaises(ValueError):
            dimod.lp.dumps(cqm)

    @parameterized.expand(LP_ODD_LABELS)
    def test_odd_constraint_labels(self, label):
        cqm = CQM()
        cqm.add_constraint(Binary('a') + Binary('b') == 1, label=label)

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    def test_long_line_breaking(self):
        cqm = CQM()
        x = dimod.Binary('x' * 255)
        y = dimod.Binary('y' * 255)
        z = dimod.Binary('z' * 255)
        cqm.set_objective(1e30*x*y + 1e30*y*z - 1e30*z*x)
        cqm.add_constraint(1e30*x + 1e30*y + 1e30*z - 1e30*x*y + 1e30*y*z <= 1e30, label='c0')

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

        for line in dimod.lp.dumps(cqm).splitlines():
            self.assertLess(len(line), LP_MAX_LINE_LEN)

    def test_integer_bounds(self):
        LOWER_BOUND = -10
        UPPER_BOUND = 10

        i = Integer('i', lower_bound=LOWER_BOUND)
        j = Integer('j', upper_bound=UPPER_BOUND)
        k = Integer('k', lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND)

        cqm = CQM()
        cqm.set_objective(i + j + k)
        cqm.add_constraint(i*j - k <= 1, label='c0')

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

        self.assertEqual(new.lower_bound('i'), LOWER_BOUND)
        self.assertEqual(new.upper_bound('i'), (1 << 53) - 1)
        self.assertEqual(new.lower_bound('j'), 0)
        self.assertEqual(new.upper_bound('j'), UPPER_BOUND)
        self.assertEqual(new.lower_bound('k'), LOWER_BOUND)
        self.assertEqual(new.upper_bound('k'), UPPER_BOUND)

    def test_real_bounds(self):
        LOWER_BOUND = -10.123
        UPPER_BOUND = 10.456

        x = Real('x', lower_bound=LOWER_BOUND)
        y = Real('y', upper_bound=UPPER_BOUND)
        z = Real('z', lower_bound=LOWER_BOUND, upper_bound=UPPER_BOUND)

        cqm = CQM()
        cqm.set_objective(x + y + z)

        # Real variables cannot have interactions, so add instead of multiply here
        cqm.add_constraint(x + y - z <= 1, label='c0')

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

        self.assertEqual(new.lower_bound('x'), LOWER_BOUND)
        self.assertEqual(new.upper_bound('x'), 1e30)
        self.assertEqual(new.lower_bound('y'), 0)
        self.assertEqual(new.upper_bound('y'), UPPER_BOUND)
        self.assertEqual(new.lower_bound('z'), LOWER_BOUND)
        self.assertEqual(new.upper_bound('z'), UPPER_BOUND)

    @parameterized.expand(LP_TEST_VALUES)
    def test_objective_linear_bias_values(self, value):
        cqm = CQM()
        cqm.set_objective(BQM({'a': value}, {}, 'BINARY'))

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    @parameterized.expand(LP_TEST_VALUES)
    def test_objective_quadratic_bias_values(self, value):
        cqm = CQM()
        cqm.set_objective(BQM([], {'ab': value}, 'BINARY'))

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    @parameterized.expand(LP_TEST_VALUES)
    def test_constraint_linear_coefficient_values(self, value):
        cqm = CQM()
        cqm.add_constraint(Integer('a') * value + Integer('b') <= 1, label='c0')

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    @parameterized.expand(LP_TEST_VALUES)
    def test_constraint_quadratic_coefficient_values(self, value):
        cqm = CQM()
        cqm.add_constraint(Integer('a') * Integer('b') * value <= 1, label='c0')

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    @parameterized.expand(LP_TEST_VALUES)
    def test_constraint_rhs_values(self, value):
        cqm = CQM()
        cqm.add_constraint(Integer('a') + 0.1 * Integer('b') <= value, label='c0')

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    @parameterized.expand(LP_TEST_VALUES)
    def test_constraint_from_model_Float32BQM(self, value):
        cqm = CQM()
        bqm = dimod.Float32BQM({'a': value}, {}, 'BINARY')
        cqm.add_constraint_from_model(bqm, '<=', label='c0')

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    def test_random_labels(self):
        cqm = CQM()
        a, b = dimod.Binaries([None, None])
        i, j = dimod.Integers([None, None])
        cqm.set_objective(a * i - b * j)
        cqm.add_constraint(a + 2 * j <= 1)
        cqm.add_constraint(-b + 2 * i >= 1)

        new = dimod.lp.loads(dimod.lp.dumps(cqm))
        self._assert_cqms_are_equivalent(cqm, new)

    def test_soft_constraint(self):
        cqm = CQM()
        a, b = dimod.Binaries(['a', 'b'])
        cqm.add_constraint(a + 2 * b <= 1, weight=1)
        with self.assertRaises(ValueError):
            dimod.lp.dumps(cqm)
