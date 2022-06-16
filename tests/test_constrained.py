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
import json
import unittest

from textwrap import dedent

import dimod

import os.path as path

import numpy as np

from dimod import BQM, Spin, Binary, CQM, Integer
from dimod.sym import Sense


class TestAddVariable(unittest.TestCase):
    def test_inconsistent_bounds(self):
        cqm = dimod.CQM()
        i = dimod.Integer('i')
        cqm.set_objective(i)
        cqm.set_lower_bound('i', 1)
        cqm.set_upper_bound('i', 5)
        with self.assertRaises(ValueError):
            cqm.add_variable('INTEGER', 'i', lower_bound=-1)
        with self.assertRaises(ValueError):
            cqm.add_variable('INTEGER', 'i', upper_bound=6)

    def test_return_value(self):
        cqm = dimod.CQM()
        self.assertEqual(cqm.add_variable('INTEGER', 'i'), 'i')
        self.assertEqual(cqm.add_variable('INTEGER', 'i'), 'i')

    def test_deprecation(self):
        cqm = dimod.CQM()
        with self.assertWarns(DeprecationWarning):
            cqm.add_variable('a', 'INTEGER')
        with self.assertWarns(DeprecationWarning):
            cqm.add_variable('b', dimod.INTEGER)
        with self.assertWarns(DeprecationWarning):
            cqm.add_variable('c', frozenset((-1, 1)))

        # ambiguous cases should use the new format
        self.assertEqual(cqm.add_variable('SPIN', 'BINARY'), 'BINARY')
        self.assertEqual(cqm.vartype('BINARY'), dimod.SPIN)


class TestAddConstraint(unittest.TestCase):
    def test_bqm(self):
        cqm = CQM()

        bqm = BQM({'a': -1}, {'ab': 1}, 1.5, 'SPIN')
        cqm.add_constraint(bqm, '<=')
        cqm.add_constraint(bqm, '>=')  # add it again

    def test_duplicate(self):
        cqm = CQM()

        bqm = BQM({'a': -1}, {'ab': 1}, 1.5, 'SPIN')

        cqm.add_constraint(bqm <= 5, label='hello')
        with self.assertRaises(ValueError):
            cqm.add_constraint(bqm <= 5, label='hello')

    def test_symbolic(self):
        cqm = CQM()

        a = Spin('a')
        b = Spin('b')
        c = Spin('c')

        cqm.add_constraint(2*a*b + b*c - c + 1 <= 1)

    def test_symbolic_mixed(self):
        cqm = CQM()

        x = Binary('x')
        s = Spin('s')
        i = Integer('i')

        cqm.add_constraint(2*i + s + x <= 2)

        self.assertIs(cqm.vartype('x'), dimod.BINARY)
        self.assertIs(cqm.vartype('s'), dimod.SPIN)
        self.assertIs(cqm.vartype('i'), dimod.INTEGER)

    def test_terms(self):
        cqm = CQM()

        a = cqm.add_variable('BINARY', 'a')
        b = cqm.add_variable('BINARY', 'b')
        c = cqm.add_variable('INTEGER', 'c')

        cqm.add_constraint([(a, b, 1), (b, 2.5,), (3,), (c, 1.5)], sense='<=')

    def test_terms_integer_bounds(self):
        # bug report: https://github.com/dwavesystems/dimod/issues/943
        cqm = CQM()
        i = Integer('i', lower_bound=-1, upper_bound=5)
        cqm.set_objective(i)
        label = cqm.add_constraint([('i', 1)], sense='<=')  # failing in #943

        self.assertEqual(cqm.constraints[label].lhs.lower_bound('i'), -1)
        self.assertEqual(cqm.constraints[label].lhs.upper_bound('i'), 5)


class TestAddDiscrete(unittest.TestCase):
    def test_bqm(self):
        cqm = CQM()
        qm = sum(dimod.Binaries('xyz'))
        c = cqm.add_discrete(qm, label='hello')
        self.assertIn(c, cqm.discrete)
        self.assertTrue(cqm.constraints[c].lhs.is_equal(qm))

    def test_comparison(self):
        cqm = CQM()
        qm = sum(dimod.Binaries('xyz'))
        c = cqm.add_discrete(qm == 1, label='hello')
        self.assertIn(c, cqm.discrete)
        self.assertTrue(cqm.constraints[c].lhs.is_equal(qm))

    def test_exceptions(self):
        cqm = dimod.CQM()
        x, y, z = dimod.Binaries('xyz')
        with self.subTest("too few variables"):
            with self.assertRaises(ValueError):
                cqm.add_discrete('x')
            with self.assertRaises(ValueError):
                cqm.add_discrete(x)
        with self.subTest("wrong sense"):
            with self.assertRaises(ValueError):
                cqm.add_discrete(x + y <= 1)
        with self.subTest("wrong rhs"):
            with self.assertRaises(ValueError):
                cqm.add_discrete(x + y == 2)
        with self.subTest("wrong vartype"):
            s = dimod.Spin('s')
            with self.assertRaises(ValueError):
                cqm.add_discrete(x + s == 1)
        with self.subTest("quadratic"):
            with self.assertRaises(ValueError):
                cqm.add_discrete(x + y + x*y== 1)

    def test_iterator(self):
        cqm = CQM()
        label = cqm.add_discrete(iter(range(10)), label='hello')
        self.assertEqual(cqm.constraints[label].lhs.variables, range(10))

    def test_label(self):
        cqm = CQM()
        label = cqm.add_discrete('abc', label='hello')
        self.assertEqual(label, 'hello')
        self.assertEqual(cqm.variables, 'abc')

    def test_qm(self):
        cqm = CQM()
        qm = dimod.QM.from_bqm(sum(dimod.Binaries('xyz')))
        c = cqm.add_discrete(qm, label='hello')
        self.assertIn(c, cqm.discrete)
        self.assertTrue(cqm.constraints[c].lhs.is_equal(qm))

    def test_qm_overlap(self):
        b1 = dimod.Binary('b1')
        b2 = dimod.Binary('b2')
        b3 = dimod.Binary('b3')

        cqm = dimod.CQM()
        cqm.add_discrete(dimod.quicksum([b1, b2]))
        with self.assertRaises(ValueError):
            cqm.add_discrete(dimod.quicksum([b1, b3]))

    def test_qm_not_binary(self):
        x, y = dimod.Binaries('xy')
        i, j = dimod.Integers('ij')
        ii = dimod.Binary('i')

        cqm = dimod.CQM()
        cqm.add_constraint(i <= 5)
        with self.assertRaises(ValueError):
            cqm.add_discrete(x + y + ii)

        with self.assertRaises(ValueError):
            cqm.add_discrete(x + j)

    def test_simple(self):
        cqm = CQM()
        cqm.add_discrete('abc')


class TestBounds(unittest.TestCase):
    def test_inconsistent(self):
        i0 = Integer('i')
        i1 = Integer('i', upper_bound=1)
        i2 = Integer('i', lower_bound=-2)

        cqm = CQM()
        cqm.set_objective(i0)
        with self.assertRaises(ValueError):
            cqm.add_constraint(i1 <= 1)

        cqm = CQM()
        cqm.add_constraint(i0 <= 1)
        with self.assertRaises(ValueError):
            cqm.set_objective(i2)

    def test_later_defn(self):
        i0 = Integer('i')
        i1 = Integer('i', upper_bound=1)

        cqm = CQM()
        cqm.add_variable('INTEGER', 'i')
        cqm.set_objective(i0)
        with self.assertRaises(ValueError):
            cqm.add_constraint(i1 <= 1)

        cqm.add_variable('INTEGER', 'i')

    def test_setting(self):
        i = Integer('i')
        j = Integer('j', upper_bound=5, lower_bound=-2)
        x = Binary('x')

        cqm = CQM()
        cqm.set_objective(2*i - j + x)
        cqm.add_constraint(i + j <= 5, 'c1')
        cqm.add_constraint(x*j == 3, 'c2')

        with self.assertRaises(ValueError):
            cqm.set_upper_bound('i', -1)  # too low

        cqm.set_upper_bound('i', 1.5)
        self.assertEqual(cqm.upper_bound('i'), 1.5)

        with self.assertRaises(ValueError):
            cqm.set_lower_bound('i', 1.2)
        cqm.set_lower_bound('i', -100)
        self.assertEqual(cqm.lower_bound('i'), -100)

        # can't modify 'x' since it's binary
        with self.assertRaises(ValueError):
            cqm.set_upper_bound('x', 100)
        with self.assertRaises(ValueError):
            cqm.set_lower_bound('x', -100)

        # internal state of the constraints shouldn't matter but just in
        # case
        self.assertEqual(cqm.constraints['c1'].lhs.lower_bound('i'), -100)
        self.assertEqual(cqm.constraints['c1'].lhs.upper_bound('i'), 1.5)


class TestCheckFeasible(unittest.TestCase):
    def test_simple(self):
        x, y, z = dimod.Binaries('xyz')

        cqm = dimod.CQM()
        cqm.add_constraint((x + y + z) * 3 <= 3)

        self.assertTrue(cqm.check_feasible({'x': 1, 'y': 0, 'z': 0}))
        self.assertTrue(cqm.check_feasible({'x': 0, 'y': 0, 'z': 0}))
        self.assertFalse(cqm.check_feasible({'x': 1, 'y': 0, 'z': 1}))

    def test_tolerance(self):
        x, y, z = dimod.Binaries('xyz')

        cqm = dimod.CQM()
        cqm.add_constraint((x + y + z) * 3 * .1 <= .9)

        sample = {'x': 1, 'y': 1, 'z': 1}

        self.assertTrue(cqm.check_feasible(sample))


class TestCopy(unittest.TestCase):
    def test_deepcopy(self):
        from copy import deepcopy

        cqm = CQM()

        x = Binary('x')
        s = Spin('s')
        i = Integer('i')

        cqm.set_objective(i + s + x)
        constraint = cqm.add_constraint(i + s + x <= 1)

        new = deepcopy(cqm)

        self.assertTrue(new.objective.is_equal(cqm.objective))
        self.assertTrue(new.constraints[constraint].lhs.is_equal(cqm.constraints[constraint].lhs))


class TestFixVariable(unittest.TestCase):
    def test_typical(self):
        x, y, z = dimod.Binaries('xyz')
        i, j = dimod.Integers('ij')

        cqm = CQM()
        cqm.set_objective(x + 2*y + 3*i + 4*j)
        c0 = cqm.add_constraint(3*i + 2*x*j + 5*i*j <= 5, label='c0')
        c1 = cqm.add_discrete('xyz', label='c1')

        with self.assertRaises(ValueError):
            cqm.fix_variable('not a variable', 1)

        fixed = cqm.fix_variable('x', 0)

        self.assertNotIn('x', cqm.variables)
        self.assertEqual(fixed, {})
        self.assertTrue(cqm.objective.is_equal(0 + 2*y + 3*i + 4*j + 0*z))
        self.assertTrue(cqm.constraints[c0].lhs.is_equal(3*i + 2*0*j + 5*i*j))
        self.assertTrue(cqm.constraints[c1].lhs.is_equal(0 + y + z))
        self.assertIn(c1, cqm.discrete)

    def test_cascade(self):
        with self.subTest('set discrete to 1'):
            cqm = CQM()
            cqm.add_discrete('abc', label='discrete')
            fixed = cqm.fix_variable('a', 1, cascade=True)
            self.assertEqual(fixed, {'b': 0, 'c': 0})
            self.assertEqual(cqm.variables, [])
            self.assertIn('discrete', cqm.constraints)
            self.assertNotIn('discrete', cqm.discrete)
            self.assertEqual(cqm.constraints['discrete'].lhs.linear, {})
            self.assertEqual(cqm.constraints['discrete'].lhs.offset, 1)

        with self.subTest('set all the 0s'):
            cqm = CQM()
            cqm.add_discrete('abc', label='discrete')
            fixed = cqm.fix_variable('a', 0, cascade=True)
            self.assertEqual(fixed, {})
            fixed = cqm.fix_variable('b', 0, cascade=True)
            self.assertEqual(fixed, {'c': 1})
            self.assertEqual(cqm.variables, [])
            self.assertIn('discrete', cqm.constraints)
            self.assertNotIn('discrete', cqm.discrete)
            self.assertEqual(cqm.constraints['discrete'].lhs.linear, {})
            self.assertEqual(cqm.constraints['discrete'].lhs.offset, 1)

        with self.subTest('one variable equality'):
            cqm = CQM()
            c0 = cqm.add_constraint(dimod.Integer('i') + dimod.Integer('j') == 7)
            fixed = cqm.fix_variable('i', 4, cascade=True)
            self.assertEqual(fixed, {'j': 3})
            qm = dimod.QM()
            qm.offset = 7
            self.assertTrue(cqm.constraints[c0].lhs.is_equal(qm))

    def test_discrete(self):
        cqm = CQM()
        cqm.add_discrete('xyz', label='discrete')

        cqm.fix_variable('x', 0)
        cqm.fix_variable('y', 0)

        self.assertNotIn('discrete', cqm.discrete)  # reduced to size 1


class TestFixVariables(unittest.TestCase):
    def test_typical(self):
        x, y, z = dimod.Binaries('xyz')
        i, j = dimod.Integers('ij')

        cqm = CQM()
        cqm.set_objective(x + 2*y + 3*i + 4*j)
        c0 = cqm.add_constraint(3*i + 2*x*j + 5*i*j <= 5, label='c0')
        c1 = cqm.add_discrete('xyz', label='c1')

        cqm.fix_variables({'x': 1, 'i': 86})

        self.assertTrue(cqm.objective.is_equal(1 + 2*y + 3*86 + 4*j + 0*z))
        self.assertTrue(cqm.constraints[c0].lhs.is_equal(3*86+2*1*j+5*86*j))
        self.assertTrue(cqm.constraints[c1].lhs.is_equal(1+y+z))
        self.assertNotIn(c1, cqm.discrete)

    def test_cascade(self):
        x, y, z = dimod.Binaries('xyz')
        i, j = dimod.Integers('ij')

        cqm = CQM()
        cqm.set_objective(x + 2*y + 3*i + 4*j)
        c0 = cqm.add_constraint(3*i + 2*x*j + 5*i*j <= 5, label='c0')
        c1 = cqm.add_discrete('xyz', label='c1')
        c2 = cqm.add_constraint(i + j == 7, label='c2')

        fixed = cqm.fix_variables({'x': 1, 'i': 7}, cascade=True)
        self.assertEqual({'z': 0, 'y': 0, 'j': 0.0}, fixed)

    def test_cascade_infeas(self):
        x, y, z = dimod.Binaries('xyz')
        i, j = dimod.Integers('ij')

        cqm = CQM()
        cqm.set_objective(x + 2*y + 3*i + 4*j)
        c0 = cqm.add_constraint(3*i + 2*x*j + 5*i*j <= 5, label='c0')
        c1 = cqm.add_discrete('xyz', label='c1')
        c2 = cqm.add_constraint(i + j == 7, label='c2')

        with self.assertRaises(dimod.exceptions.InfeasibileModelError):
            cqm.fix_variables({'x': 1, 'y': 1}, cascade=True)

    def test_cascade_redundant(self):
        x, y, z = dimod.Binaries('xyz')
        i, j = dimod.Integers('ij')

        cqm = CQM()
        cqm.set_objective(x + 2*y + 3*i + 4*j)
        c0 = cqm.add_constraint(3*i + 2*x*j + 5*i*j <= 5, label='c0')
        c1 = cqm.add_discrete('xyz', label='c1')
        c2 = cqm.add_constraint(i + j == 7, label='c2')

        fixed = cqm.fix_variables({'x': 1, 'i': 7, 'y': 0}, cascade=True)
        self.assertEqual({'z': 0, 'y': 0, 'j': 0.0}, fixed)  # y appears because it was fixed twice


class TestFlipVariable(unittest.TestCase):
    def test_exceptions(self):
        x, y = dimod.Binaries('xy')
        i, j = dimod.Integers('ij')
        s, t = dimod.Spins('st')

        cqm = dimod.CQM()
        cqm.set_objective(x + y + i + j + s + t)
        cqm.add_constraint(x + i + s <= 5)

        with self.assertRaises(ValueError):
            cqm.flip_variable('i')  # can't flip integer
        with self.assertRaises(ValueError):
            cqm.flip_variable('not a variable')

    def test_binary(self):
        x, y, z = dimod.Binaries('xyz')
        i, j = dimod.Integers('ij')
        s, t = dimod.Spins('st')

        cqm = dimod.CQM()
        cqm.set_objective(x + y + z + i + j + s + t)
        c = cqm.add_constraint(x + i + s <= 5)
        d = cqm.add_discrete('xyz')

        cqm.flip_variable('x')

        self.assertTrue(cqm.objective.is_equal((1-x) + y + z + i + j + s + t))
        self.assertTrue(cqm.constraints[c].lhs.is_equal((1-x) + i + s))
        self.assertTrue(cqm.constraints[d].lhs.is_equal((1-x) + y + z))
        self.assertNotIn(d, cqm.discrete)


class TestIterViolations(unittest.TestCase):
    def test_no_constraints(self):
        cqm = CQM.from_bqm(-Binary('a') + Binary('a')*Binary('b') + 1.5)
        self.assertEqual(cqm.violations({'a': 1, 'b': 1}), {})

    def test_binary(self):
        a, b, c = dimod.Binaries('abc')

        cqm = CQM()
        cqm.add_constraint(a + b + c == 1, label='onehot')
        cqm.add_constraint(a*b <= 0, label='ab LE')
        cqm.add_constraint(c >= 1, label='c GE')

        sample = {'a': 0, 'b': 0, 'c': 1}  # satisfying sample
        self.assertEqual(cqm.violations(sample), {'ab LE': 0.0, 'c GE': 0.0, 'onehot': 0.0})
        self.assertEqual(cqm.violations(sample, skip_satisfied=True), {})

        sample = {'a': 1, 'b': 0, 'c': 0}  # violates one
        self.assertEqual(cqm.violations(sample), {'ab LE': 0.0, 'c GE': 1.0, 'onehot': 0.0})
        self.assertEqual(cqm.violations(sample, skip_satisfied=True), {'c GE': 1.0})

    def test_integer(self):
        i = dimod.Integer('i', lower_bound=-1000)
        j, k = dimod.Integers('jk')

        cqm = CQM()
        label_le = cqm.add_constraint(i + j*k <= 5)
        label_ge = cqm.add_constraint(i + j >= 1000)

        sample = {'i': 105, 'j': 4, 'k': 5}
        self.assertEqual(cqm.violations(sample), {label_le: 120.0, label_ge: 891.0})

        sample = {'j': -1, 'i': 1004, 'k': 1000}
        self.assertEqual(cqm.violations(sample, clip=False), {label_ge: -3.0, label_le: -1.0})


class TestIsAlmostEqual(unittest.TestCase):
    def test_simple(self):
        i, j = dimod.Integers('ij')
        x, y = dimod.Binaries('xy')

        cqm0 = dimod.CQM()
        cqm0.set_objective(i + 2*j + x*i)
        cqm0.add_constraint(i <= 5, label='a')
        cqm0.add_constraint(y*j >= 4, label='b')

        cqm1 = dimod.CQM()
        cqm1.set_objective(i + 2.0001*j + x*i)
        cqm1.add_constraint(i <= 5, label='a')
        cqm1.add_constraint(1.001*y*j >= 4, label='b')

        self.assertTrue(cqm0.is_almost_equal(cqm1, places=2))
        self.assertFalse(cqm0.is_almost_equal(cqm1, places=5))


class TestIsEqual(unittest.TestCase):
    def test_simple(self):
        i, j = dimod.Integers('ij')
        x, y = dimod.Binaries('xy')

        cqm0 = dimod.CQM()
        cqm0.set_objective(i + 2*j + x*i)
        cqm0.add_constraint(i <= 5, label='a')
        cqm0.add_constraint(y*j >= 4, label='b')

        cqm1 = dimod.CQM()
        cqm1.set_objective(i + 2*j + x*i)
        cqm1.add_constraint(i <= 5, label='a')
        cqm1.add_constraint(y*j >= 4, label='b')

        self.assertTrue(cqm0.is_equal(cqm1))

        cqm1.set_objective(y)
        self.assertFalse(cqm0.is_equal(cqm1))

        cqm1.set_objective(i + 2*j + x*i)
        cqm1.add_constraint(x*y == 1)
        self.assertFalse(cqm0.is_equal(cqm1))


class TestCQMtoBQM(unittest.TestCase):
    def test_empty(self):
        bqm, inverter = dimod.cqm_to_bqm(dimod.CQM())
        self.assertEqual(bqm.shape, (0, 0))
        self.assertEqual(bqm.vartype, dimod.BINARY)

    def test_bqm_objective_only(self):
        x, y, z = dimod.Binaries('xyz')

        cqm = CQM.from_bqm(x*y + 2*y*z + 8*x + 5)

        bqm, inverter = dimod.cqm_to_bqm(cqm)

        self.assertEqual(bqm, x*y + 2*y*z + 8*x + 5)

    def test_qm_objective_only(self):
        i = dimod.Integer('i', upper_bound=7)
        j = dimod.Integer('j', upper_bound=9)
        x = dimod.Binary('x')

        qm = i*j + 5*j*x + 8*i + 3*x + 5
        cqm = CQM.from_qm(qm)

        bqm, inverter = dimod.cqm_to_bqm(cqm)

        sampleset = dimod.ExactSolver().sample(bqm)

        for bin_sample, energy in sampleset.data(['sample', 'energy']):
            int_sample = inverter(bin_sample)
            self.assertEqual(qm.energy(int_sample), energy)

    def test_bqm_equality_constraint(self):
        x, y, z = dimod.Binaries('xyz')

        cqm = CQM()
        cqm.add_constraint(x + y + z == 1)

        bqm, inverter = dimod.cqm_to_bqm(cqm, lagrange_multiplier=1)

        self.assertEqual(bqm, (x + y + z - 1)*(x + y + z - 1))

    def test_bqm_equality_constraint_no_lagrange(self):
        x, y, z = dimod.Binaries('xyz')

        cqm = CQM()
        cqm.add_constraint(x + y + z == 1)

        bqm, inverter = dimod.cqm_to_bqm(cqm)

        self.assertEqual(bqm, (x + y + z - 1)*(x + y + z - 1))

    def test_bqm_equality_constraint_offset(self):
        x, y, z = dimod.Binaries('xyz')

        cqm = CQM()
        cqm.add_constraint(x + y + z - 1 == 2)

        bqm, inverter = dimod.cqm_to_bqm(cqm, lagrange_multiplier=1)

        self.assertEqual(bqm, (x + y + z - 3)*(x + y + z - 3))

    def test_bqm_Le_constraint(self):
        x, y, z = dimod.Binaries('xyz')

        cqm = CQM()
        cqm.add_constraint(x + y + z <= 2)

        bqm, inverter = dimod.cqm_to_bqm(cqm, lagrange_multiplier=1)

        configs = set()
        for sample in dimod.ExactSolver().sample(bqm).lowest().samples():
            self.assertLessEqual(sample['x'] + sample['y'] + sample['z'], 2)
            configs.add((sample['x'], sample['y'], sample['z']))
        self.assertEqual(len(configs), 7)

    def test_bqm_Ge_constraint(self):
        x, y, z = dimod.Binaries('xyz')

        cqm = CQM()
        cqm.add_constraint(x + y + z >= 2)

        bqm, inverter = dimod.cqm_to_bqm(cqm, lagrange_multiplier=1)

        configs = set()
        for sample in dimod.ExactSolver().sample(bqm).lowest().samples():
            self.assertGreaterEqual(sample['x'] + sample['y'] + sample['z'], 2)
            configs.add((sample['x'], sample['y'], sample['z']))
        self.assertEqual(len(configs), 4)

    def test_qm_Ge_constraint(self):
        i = dimod.Integer('i', upper_bound=7)
        j = dimod.Integer('j', upper_bound=9)
        x = dimod.Binary('x')

        cqm = CQM()
        cqm.add_constraint(i + j + x >= 5)

        bqm, inverter = dimod.cqm_to_bqm(cqm, lagrange_multiplier=1)

        for bin_sample in dimod.ExactSolver().sample(bqm).lowest().samples():
            int_sample = inverter(bin_sample)
            self.assertGreaterEqual(int_sample['i'] + int_sample['j'] + int_sample['x'], 5)

    def test_serializable(self):
        i = dimod.Integer('i', upper_bound=7)
        j = dimod.Integer('j', upper_bound=9)
        x = dimod.Binary('x')

        cqm = CQM()
        cqm.add_constraint(i + j + x >= 5)

        bqm, inverter = dimod.cqm_to_bqm(cqm, lagrange_multiplier=1)

        newinverter = dimod.constrained.CQMToBQMInverter.from_dict(
            json.loads(json.dumps(inverter.to_dict())))

        for bin_sample in dimod.ExactSolver().sample(bqm).lowest().samples():
            int_sample = newinverter(bin_sample)
            self.assertGreaterEqual(int_sample['i'] + int_sample['j'] + int_sample['x'], 5)


class TestFromDQM(unittest.TestCase):
    def test_case_label(self):
        dqm = dimod.CaseLabelDQM()
        u = dqm.add_variable({'red', 'green', 'blue'}, shared_labels=True)
        v = dqm.add_variable(['blue', 'yellow', 'brown'], label='v', shared_labels=True)
        dqm.set_linear_case(u, 'red', 1)
        dqm.set_linear_case(v, 'yellow', 2)
        dqm.set_quadratic_case(u, 'green', v, 'blue', -0.5)
        dqm.set_quadratic_case(u, 'blue', v, 'brown', -0.5)

        cqm = CQM.from_discrete_quadratic_model(dqm)

        self.assertEqual(cqm.objective.linear,
                         {(0, 'blue'): 0.0, (0, 'red'): 1.0, (0, 'green'): 0.0,
                          ('v', 'blue'): 0.0, ('v', 'brown'): 0.0, ('v', 'yellow'): 2.0})
        self.assertEqual(cqm.objective.quadratic,
                         {(('v', 'blue'), (0, 'green')): -0.5, (('v', 'brown'), (0, 'blue')): -0.5})
        self.assertEqual(cqm.objective.offset, 0)
        self.assertTrue(all(cqm.objective.vartype(v) is dimod.BINARY
                            for v in cqm.objective.variables))

        self.assertEqual(set(cqm.constraints), set(dqm.variables))

    def test_empty(self):
        dqm = dimod.DiscreteQuadraticModel()

        cqm = CQM.from_discrete_quadratic_model(dqm)

        self.assertEqual(len(cqm.variables), 0)
        self.assertEqual(len(cqm.constraints), 0)

    def test_typical(self):
        dqm = dimod.DQM()
        u = dqm.add_variable(4)
        v = dqm.add_variable(3)
        dqm.set_quadratic(u, v, {(0, 2): -1, (2, 1): 1})
        dqm.set_linear(u, [0, 1, 2, 3])
        dqm.offset = 5

        cqm = CQM.from_discrete_quadratic_model(dqm)

        self.assertEqual(cqm.variables, [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)])

        self.assertEqual(cqm.objective.linear,
                         {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3,
                          (1, 1): 0, (1, 2): 0, (1, 0): 0})
        self.assertEqual(cqm.objective.quadratic, {((1, 1), (0, 2)): 1.0, ((1, 2), (0, 0)): -1.0})
        self.assertEqual(cqm.objective.offset, dqm.offset)

        # keys of constraints are the variables of DQM
        self.assertEqual(set(cqm.constraints), set(dqm.variables))


class TestNumBiases(unittest.TestCase):
    def test_simple(self):
        x, y, z = dimod.Binaries('xyz')
        i, j = dimod.Integers('ij')

        cqm = dimod.CQM()
        cqm.set_objective(x + y + z + i + j + i*x)
        cqm.add_constraint(x + y + z + x*y + z*y <= 5)
        cqm.add_constraint(i + x + i*x == 5)

        self.assertEqual(cqm.num_biases(), 14)
        self.assertEqual(cqm.num_biases(linear_only=True), 10)
        self.assertEqual(cqm.num_biases('BINARY'), 11)
        self.assertEqual(cqm.num_biases('SPIN'), 0)
        self.assertEqual(cqm.num_biases('INTEGER'), 5)
        self.assertEqual(cqm.num_biases('BINARY', linear_only=True), 7)


class TestNumQuadraticVariables(unittest.TestCase):
    def test_simple(self):
        x, y, z = dimod.Binaries('xyz')
        i, j = dimod.Integers('ij')

        cqm = dimod.CQM()
        cqm.set_objective(x + y + z + i + j + i*x)
        cqm.add_constraint(x + y + z + x*y + z*y <= 5)
        cqm.add_constraint(i + x + i*x == 5)

        with self.assertWarns(DeprecationWarning):
            cqm.num_quadratic_variables()

        self.assertEqual(cqm.num_quadratic_variables(include_objective=False), 5)
        self.assertEqual(cqm.num_quadratic_variables(include_objective=True), 7)
        self.assertEqual(cqm.num_quadratic_variables('BINARY', include_objective=False), 4)
        self.assertEqual(cqm.num_quadratic_variables('BINARY', include_objective=True), 5)


class FromQM(unittest.TestCase):
    def test_from_bqm(self):
        bqm = BQM({'a': -1}, {'ab': 1}, 1.5, 'SPIN')
        cqm = CQM.from_bqm(bqm)
        self.assertTrue(cqm.objective.is_equal(dimod.QM.from_bqm(bqm)))

    def test_from_qm(self):
        qm = BQM({'a': -1}, {'ab': 1}, 1.5, 'SPIN') + Integer('i')
        cqm = CQM.from_quadratic_model(qm)
        self.assertTrue(cqm.objective.is_equal(qm))


class TestRelabelConstraints(unittest.TestCase):
    def test_discrete(self):
        x, y = dimod.Binaries('xy')
        cqm = dimod.CQM()

        c0 = cqm.add_constraint(x + y - x*y <= 5, label='c0')
        c1 = cqm.add_discrete(x + y == 1, label='c1')

        cqm.relabel_constraints({c1: 'c2'})

        self.assertEqual(set(cqm.constraints), {'c0', 'c2'})
        self.assertEqual(cqm.discrete, {'c2'})
        self.assertEqual(cqm.constraints['c2'].lhs, x + y)

    def test_superset(self):
        x, y = dimod.Binaries('xy')
        cqm = dimod.CQM()

        c0 = cqm.add_constraint(x + y - x*y <= 5, label='c0')
        c1 = cqm.add_discrete(x + y == 1, label='c1')

        cqm.relabel_constraints({'a': 'b'})

    def test_swap(self):
        x, y = dimod.Binaries('xy')
        cqm = dimod.CQM()

        c0 = cqm.add_constraint(x + y <= 5)
        c1 = cqm.add_constraint(x*y == 1)

        cqm.relabel_constraints({c0: c1, c1: c0})

        self.assertEqual(set(cqm.constraints), {c0, c1})
        self.assertEqual(cqm.constraints[c0].lhs, x*y)
        self.assertEqual(cqm.constraints[c1].lhs, x+y)


class TestRelabelVariables(unittest.TestCase):
    def test_copy(self):
        cqm = CQM()
        u, v, w = dimod.Binaries('uvw')
        cqm.set_objective(6*u + v + w)
        c0 = cqm.add_constraint(u + w <= 5)
        c1 = cqm.add_constraint(w + v == 3)

        new = cqm.relabel_variables({'u': 1, 'v': 0}, inplace=False)

        self.assertIsNot(new, cqm)
        self.assertEqual(new.objective.variables, [1, 0, 'w'])
        self.assertEqual(new.constraints[c0].lhs.variables, [1, 'w'])
        self.assertEqual(new.constraints[c1].lhs.variables, ['w', 0])
        self.assertEqual(cqm.objective.variables, 'uvw')

    def test_inplace(self):
        cqm = CQM()
        u, v, w = dimod.Binaries('uvw')
        cqm.set_objective(6*u + v + w)
        c0 = cqm.add_constraint(u + w <= 5)
        c1 = cqm.add_constraint(w + v == 3)

        new = cqm.relabel_variables({'u': 1, 'v': 0})

        self.assertIs(new, cqm)
        self.assertEqual(cqm.objective.variables, [1, 0, 'w'])
        self.assertEqual(cqm.constraints[c0].lhs.variables, [1, 'w'])
        self.assertEqual(cqm.constraints[c1].lhs.variables, ['w', 0])


class TestRemoveConstraint(unittest.TestCase):
    def test_cascade(self):
        cqm = CQM()
        u, v, w = dimod.Binaries('uvw')

        cqm.set_objective(u)
        cqm.add_constraint(v == 1)
        c0 = cqm.add_constraint(u + v + w == 0, label='c0')
        c1 = cqm.add_constraint(u + v + w == 0, label='c1')

        cqm.remove_constraint(c0, cascade=False)
        self.assertEqual(cqm.variables, ['u', 'v', 'w'])
        cqm.remove_constraint(c1, cascade=True)
        self.assertEqual(cqm.variables, ['u', 'v'])

    def test_simple(self):
        x, y = dimod.Binaries('xy')
        cqm = dimod.CQM()

        c0 = cqm.add_constraint(x + y - x*y <= 5, label='c0')
        c1 = cqm.add_discrete(x + y == 1, label='c1')

        cqm.remove_constraint(c1)

        self.assertEqual(set(cqm.constraints), {c0})
        self.assertEqual(cqm.discrete, set())

        with self.assertRaises(ValueError):
            cqm.remove_constraint('not a constraint')


class TestSerialization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dimod.REAL_INTERACTIONS = True

    @classmethod
    def tearDownClass(cls):
        dimod.REAL_INTERACTIONS = False

    def test_functional(self):
        cqm = CQM()

        bqm = BQM({'a': -1}, {'ab': 1}, 1.5, 'SPIN')
        cqm.add_constraint(bqm, '<=')
        cqm.add_constraint(bqm, '>=')
        cqm.set_objective(BQM({'c': -1}, {}, 'SPIN'))
        cqm.add_constraint(Spin('a')*Integer('d')*5 <= 3)

        new = CQM.from_file(cqm.to_file())

        self.assertTrue(cqm.objective.is_equal(new.objective))
        self.assertEqual(set(cqm.constraints), set(new.constraints))
        for label, constraint in cqm.constraints.items():
            self.assertTrue(constraint.lhs.is_equal(new.constraints[label].lhs))
            self.assertEqual(constraint.rhs, new.constraints[label].rhs)
            self.assertEqual(constraint.sense, new.constraints[label].sense)

    def test_functional_empty(self):
        new = CQM.from_file(CQM().to_file())

    def test_functional_discrete(self):
        cqm = CQM()

        bqm = BQM({'a': -1}, {'ab': 1}, 1.5, 'SPIN')
        cqm.add_constraint(bqm, '<=')
        cqm.add_constraint(bqm, '>=')
        cqm.set_objective(Integer('c'))
        cqm.add_constraint(Spin('a')*Integer('d')*5 <= 3)
        cqm.add_discrete('efg')

        new = CQM.from_file(cqm.to_file())

        self.assertTrue(cqm.objective.is_equal(new.objective))
        self.assertEqual(set(cqm.constraints), set(new.constraints))
        for label, constraint in cqm.constraints.items():
            self.assertTrue(constraint.lhs.is_equal(new.constraints[label].lhs))
            self.assertEqual(constraint.rhs, new.constraints[label].rhs)
            self.assertEqual(constraint.sense, new.constraints[label].sense)
        self.assertSetEqual(cqm.discrete, new.discrete)

    def test_header(self):
        from dimod.serialization.fileview import read_header

        cqm = CQM()

        x = Binary('x')
        s = Spin('s')
        i = Integer('i')

        cqm.set_objective(x + 3*i + s*x)
        cqm.add_constraint(x*s + x <= 5)
        cqm.add_constraint(i*i + i*s <= 4)

        header_info = read_header(cqm.to_file(), b'DIMODCQM')

        self.assertEqual(header_info.data,
                         dict(num_biases=11,
                              num_constraints=2,
                              num_quadratic_variables=4,
                              num_variables=3,
                              num_quadratic_variables_real=0,
                              num_linear_biases_real=0,
                              ))

    def test_header_real(self):
        from dimod.serialization.fileview import read_header

        x, y = dimod.Binaries('xy')
        s = Spin('s')
        i = Integer('i')
        a, b = dimod.Reals('ab')

        cqm = CQM()
        cqm.set_objective(a + b + x + s*i)
        cqm.add_constraint(a + b <= 5)
        cqm.add_constraint(a - i == 4)
        cqm.add_constraint(x + y + x*y == 5)

        self.assertEqual(read_header(cqm.to_file(), b'DIMODCQM').data,
                         dict(num_biases=14,
                              num_constraints=3,
                              num_quadratic_variables=2,
                              num_variables=6,
                              num_quadratic_variables_real=0,
                              num_linear_biases_real=5,
                              ))

        cqm.set_objective(a + b + x + s*i + a*b)
        cqm.add_constraint(3*a*(1 - b) == 4)

        self.assertEqual(read_header(cqm.to_file(), b'DIMODCQM').data,
                         dict(num_biases=18,
                              num_constraints=4,
                              num_quadratic_variables=4,
                              num_variables=6,
                              num_quadratic_variables_real=4,
                              num_linear_biases_real=7,
                              ))


class TestSetObjective(unittest.TestCase):
    def test_bqm(self):
        for dtype in [np.float32, np.float64, object]:
            with self.subTest(dtype=np.dtype(dtype).name):
                bqm = dimod.BQM({'a': 1}, {'ab': 4}, 5, 'BINARY', dtype=dtype)

                cqm = dimod.CQM()

                cqm.set_objective(bqm)

                self.assertEqual(cqm.objective.linear, bqm.linear)
                self.assertEqual(cqm.objective.quadratic, bqm.quadratic)
                self.assertEqual(cqm.objective.offset, bqm.offset)

                # doing it again should do nothing
                cqm.set_objective(bqm)

                self.assertEqual(cqm.objective.linear, bqm.linear)
                self.assertEqual(cqm.objective.quadratic, bqm.quadratic)
                self.assertEqual(cqm.objective.offset, bqm.offset)

    def test_empty(self):
        self.assertEqual(CQM().objective.num_variables, 0)

    def test_set(self):
        cqm = CQM()
        cqm.set_objective(Integer('a') * 5)
        self.assertTrue(cqm.objective.is_equal(Integer('a') * 5))

    def test_terms_objective(self):
        cqm = CQM()

        a = cqm.add_variable('BINARY', 'a')
        b = cqm.add_variable('BINARY', 'b')
        c = cqm.add_variable('INTEGER', 'c')

        cqm.set_objective([(a, b, 1), (b, 2.5,), (3,), (c, 1.5)])
        energy = cqm.objective.energy({'a': 1, 'b': 0, 'c': 10})
        self.assertAlmostEqual(energy, 18)
        energy = cqm.objective.energy({'a': 1, 'b': 1, 'c': 3})
        self.assertAlmostEqual(energy, 11)


class TestSubstituteSelfLoops(unittest.TestCase):
    def test_typical(self):
        i, j, k = dimod.Integers('ijk')

        cqm = CQM()

        cqm.set_objective(2*i*i + i*k)
        label = cqm.add_constraint(-3*i*i + 4*j*j <= 5)

        mapping = cqm.substitute_self_loops()

        self.assertIn('i', mapping)
        self.assertIn('j', mapping)
        self.assertEqual(len(mapping), 2)

        self.assertEqual(cqm.objective.quadratic, {('k', 'i'): 1, (mapping['i'], 'i'): 2})
        self.assertEqual(cqm.constraints[label].lhs.quadratic,
                         {(mapping['i'], 'i'): -3.0, (mapping['j'], 'j'): 4.0})
        self.assertEqual(len(cqm.constraints), 3)

        for v, new in mapping.items():
            self.assertIn(new, cqm.constraints)
            self.assertEqual(cqm.constraints[new].sense, Sense.Eq)
            self.assertEqual(cqm.constraints[new].lhs.linear, {v: 1, new: -1})
            self.assertEqual(cqm.constraints[new].rhs, 0)


class TestCQMFromLPFile(unittest.TestCase):

    def test_linear(self):

        filepath = path.join(path.dirname(path.abspath(__file__)), 'data', 'test_linear.lp')

        with open(filepath) as f:
            with self.assertWarns(DeprecationWarning):
                cqm = CQM.from_lp_file(f)

        self.assertEqual(len(cqm.variables), 3, msg='wrong number of variables')
        self.assertEqual(len(cqm.constraints), 1, msg='wrong number of constraints')

        # check objective
        self.assertAlmostEqual(cqm.objective.get_linear('x0'), 1, msg='linear(x0) should be 1')
        self.assertAlmostEqual(cqm.objective.get_linear('x1'), 1, msg='linear(x1) should be 1')
        self.assertAlmostEqual(cqm.objective.get_linear('x2'), 3, msg='linear(x2) should be 3')

        # check constraint:
        for cname, cmodel in cqm.constraints.items():

            if cname == 'c1':
                self.assertAlmostEqual(cmodel.lhs.get_linear('x0'), 1,
                                       msg='constraint c1, linear(x0) should be 1')
                self.assertAlmostEqual(cmodel.lhs.get_linear('x2'), 1,
                                       msg='constraint c1, linear(x3) should be 1')
                self.assertAlmostEqual(cmodel.lhs.offset, 0, msg='constraint c1, offset should be 0')
                self.assertTrue(cmodel.sense == Sense.Le, msg='constraint c1, should be inequality')
                self.assertAlmostEqual(cmodel.rhs, 9)

            else:
                raise KeyError('Not expected constraint: {}'.format(cname))

    def test_quadratic(self):

        filepath = path.join(path.dirname(path.abspath(__file__)), 'data', 'test_quadratic.lp')

        with open(filepath) as f:
            with self.assertWarns(DeprecationWarning):
                cqm = CQM.from_lp_file(f)

        self.assertEqual(len(cqm.variables), 4, msg='wrong number of variables')
        self.assertEqual(len(cqm.constraints), 2, msg='wrong number of constraints')

        # check objective:
        self.assertAlmostEqual(cqm.objective.get_linear('x0'), 0,
                               msg=' linear(x0) should be 0')
        self.assertAlmostEqual(cqm.objective.get_linear('x1'), 0,
                               msg=' linear(x1) should be 0')
        self.assertAlmostEqual(cqm.objective.get_linear('x2'), 0,
                               msg=' linear(x2) should be 0')
        self.assertAlmostEqual(cqm.objective.get_linear('x3'), 0,
                               msg=' linear(x3) should be 0')

        self.assertAlmostEqual(cqm.objective.get_quadratic('x0', 'x1'), 0.5,
                               msg='quad(x0, x1) should be 0.5')
        self.assertAlmostEqual(cqm.objective.get_quadratic('x0', 'x2'), 0.5,
                               msg='quad(x0, x2) should be 0.5')
        self.assertAlmostEqual(cqm.objective.get_quadratic('x0', 'x3'), 0.5,
                               msg='quad(x0, x3) should be 0.5')
        self.assertAlmostEqual(cqm.objective.get_quadratic('x1', 'x2'), 0.5,
                               msg='quad(x1, x2) should be 0.5')
        self.assertAlmostEqual(cqm.objective.get_quadratic('x1', 'x3'), 0.5,
                               msg='quad(x1, x3) should be 0.5')
        self.assertAlmostEqual(cqm.objective.get_quadratic('x2', 'x3'), 0.5,
                               msg='quad(x2, x3) should be 0.5')

        # check constraints:
        for cname, cmodel in cqm.constraints.items():

            if cname == 'c1':

                self.assertAlmostEqual(cmodel.lhs.get_linear('x0'), 1,
                                       msg='constraint c1, linear(x0) should be 1')
                self.assertAlmostEqual(cmodel.lhs.get_linear('x3'), 1,
                                       msg='constraint c1, linear(x3) should be 1')
                self.assertAlmostEqual(cmodel.lhs.offset, 0,
                                       msg='constraint c1, offset should be 0')
                self.assertTrue(cmodel.sense == Sense.Le,
                                msg='constraint c1, should be <= inequality')
                self.assertAlmostEqual(cmodel.rhs, 1)

            elif cname == 'c2':
                self.assertAlmostEqual(cmodel.lhs.get_linear('x1'), 1,
                                       msg='constraint c2, linear(x1) should be 1')
                self.assertAlmostEqual(cmodel.lhs.get_linear('x2'), 1,
                                       msg='constraint c2, linear(x2) should be 1')
                self.assertAlmostEqual(cmodel.lhs.offset, 0,
                                       msg='constraint c1, offset should be 0')
                self.assertTrue(cmodel.sense == Sense.Ge,
                                msg='constraint c2, should be >= inequality')
                self.assertAlmostEqual(cmodel.rhs, 2)

            else:
                raise KeyError('Not expected constraint: {}'.format(cname))

    def test_integer(self):
        filepath = path.join(path.dirname(path.abspath(__file__)), 'data', 'test_integer.lp')

        with open(filepath) as f:
            with self.assertWarns(DeprecationWarning):
                cqm = CQM.from_lp_file(f)

        self.assertEqual(len(cqm.variables), 5, msg='wrong number of variables')
        self.assertEqual(len(cqm.constraints), 6, msg='wrong number of constraints')

        # check the bounds
        self.assertAlmostEqual(cqm.objective.lower_bound('i0'), 3,
                               msg='lower bound of i0 should be 3')
        self.assertAlmostEqual(cqm.objective.upper_bound('i0'), 15,
                               msg='upper bound of i0 should be 15')
        self.assertAlmostEqual(cqm.objective.lower_bound('i1'), 0,
                               msg='lower bound of i1 should be 0')
        self.assertAlmostEqual(cqm.objective.upper_bound('i1'), 10,
                               msg='upper bound of i1 should be 10')

        # check objective:
        self.assertAlmostEqual(cqm.objective.get_linear('x0'), -4,
                               msg=' linear(x0) should be -4')
        self.assertAlmostEqual(cqm.objective.get_linear('x1'), -9,
                               msg=' linear(x1) should be -9')
        self.assertAlmostEqual(cqm.objective.get_linear('x2'), 0,
                               msg=' linear(x2) should be 0')
        self.assertAlmostEqual(cqm.objective.get_linear('i0'), 6,
                               msg=' linear(i0) should be 6')
        self.assertAlmostEqual(cqm.objective.get_linear('i1'), 1,
                               msg=' linear(i1) should be 1')

        self.assertAlmostEqual(cqm.objective.get_quadratic('x0', 'x1'), 0.5,
                               msg='quad(x0, x1) should be 0.5')
        self.assertAlmostEqual(cqm.objective.get_quadratic('x0', 'x2'), 0.5,
                               msg='quad(x0, x2) should be 0.5')

        # check constraints:
        for cname, cmodel in cqm.constraints.items():

            if cname == 'c1':

                self.assertAlmostEqual(cmodel.lhs.get_linear('x1'), 1,
                                       msg='constraint c1, linear(x1) should be 1')
                self.assertAlmostEqual(cmodel.lhs.get_linear('x2'), 1,
                                       msg='constraint c1, linear(x2) should be 1')
                self.assertAlmostEqual(cmodel.lhs.offset, 0,
                                       msg='constraint c1, offset should be 0')
                self.assertTrue(cmodel.sense == Sense.Ge,
                                msg='constraint c2, should be >= inequality')
                self.assertAlmostEqual(cmodel.rhs, 2)

            elif cname == 'c2':
                self.assertAlmostEqual(cmodel.lhs.get_linear('x0'), 0,
                                       msg='constraint c2, linear(x0) should be 0')
                self.assertAlmostEqual(cmodel.lhs.get_linear('x2'), 0,
                                       msg='constraint c2, linear(x2) should be 0')
                self.assertAlmostEqual(cmodel.lhs.get_quadratic('x0', 'x2'), 1,
                                       msg='quad(x0, x2) should be 1')
                self.assertAlmostEqual(cmodel.lhs.offset, 0,
                                       msg='constraint c2, offset should be -1')
                self.assertTrue(cmodel.sense == Sense.Ge,
                                msg='constraint c2, should be >= inequality')
                self.assertAlmostEqual(cmodel.rhs, 1)

            elif cname == 'c3':
                self.assertAlmostEqual(cmodel.lhs.get_linear('x1'), 3,
                                       msg='constraint c3, linear(x1) should be 3')
                self.assertAlmostEqual(cmodel.lhs.get_quadratic('x0', 'x2'), 6,
                                       msg='constraint c3, quad(x0, x2) should be 6')
                self.assertAlmostEqual(cmodel.lhs.offset, 0,
                                       msg='constraint c3, offset should be 0')
                self.assertTrue(cmodel.sense == Sense.Ge,
                                msg='constraint c3, should be >= inequality')
                self.assertAlmostEqual(cmodel.rhs, -9)

            elif cname == 'c4':
                self.assertAlmostEqual(cmodel.lhs.get_linear('x0'), 5,
                                       msg='constraint c4, linear(x0) should be 5')
                self.assertAlmostEqual(cmodel.lhs.get_linear('i0'), -9,
                                       msg='constraint c4, linear(i0) should be -9')
                self.assertAlmostEqual(cmodel.lhs.offset, 0,
                                       msg='constraint c4, offset should be 0')
                self.assertTrue(cmodel.sense == Sense.Le,
                                msg='constraint c4, should be <= inequality')
                self.assertAlmostEqual(cmodel.rhs, 1)

            elif cname == 'c5':
                self.assertAlmostEqual(cmodel.lhs.get_linear('i0'), -34,
                                       msg='constraint c5, linear(i0) should be -34')
                self.assertAlmostEqual(cmodel.lhs.get_linear('i1'), 26,
                                       msg='constraint c5, linear(i1) should be 26')
                self.assertAlmostEqual(cmodel.lhs.offset, 0,
                                       msg='constraint c5, offset should be -2')
                self.assertTrue(cmodel.sense == Sense.Eq,
                                msg='constraint c5, should be an equality')
                self.assertAlmostEqual(cmodel.rhs, 2)

            elif cname == 'c6':
                self.assertAlmostEqual(cmodel.lhs.get_linear('i0'), 0,
                                       msg='constraint c6, linear(i0) should be 0')
                self.assertAlmostEqual(cmodel.lhs.get_linear('i1'), 0,
                                       msg='constraint c6, linear(i1) should be0')
                self.assertAlmostEqual(cmodel.lhs.offset, 0,
                                       msg='constraint c6, offset should be 0')
                self.assertAlmostEqual(cmodel.lhs.get_quadratic('i0', 'i1'), 1,
                                       msg='constraint c6, quadratic (i0, i1) should be 1')
                self.assertTrue(cmodel.sense == Sense.Ge,
                                msg='constraint c6, should be an inequality')
                self.assertAlmostEqual(cmodel.rhs, 0)

            else:
                raise KeyError('Not expected constraint: {}'.format(cname))

    def test_pure_quadratic(self):
        filepath = path.join(path.dirname(path.abspath(__file__)), 'data', 'test_quadratic_variables.lp')

        with open(filepath) as f:
            with self.assertWarns(DeprecationWarning):
                cqm = CQM.from_lp_file(f)

        self.assertEqual(len(cqm.variables), 2, msg='wrong number of variables')
        self.assertEqual(len(cqm.constraints), 2, msg='wrong number of constraints')

        # check the objective
        self.assertAlmostEqual(cqm.objective.get_linear('x0'), 0.5,
                               msg=' linear(x0) should be 0.5')
        self.assertAlmostEqual(cqm.objective.get_linear('i0'), 0,
                               msg=' linear(i0) should be 0')

        self.assertAlmostEqual(cqm.objective.get_quadratic('i0', 'i0'), 0.5,
                               msg='quad(i0, i0) should be 0.5')

        for cname, cmodel in cqm.constraints.items():

            if cname == 'c1':

                self.assertAlmostEqual(cmodel.lhs.get_linear('x0'), 1,
                                       msg='constraint c1, linear(x0) should be 1')
                self.assertAlmostEqual(cmodel.lhs.offset, 0,
                                       msg='constraint c1, offset should be 0')
                self.assertTrue(cmodel.sense == Sense.Ge,
                                msg='constraint c1, should be >= inequality')
                self.assertAlmostEqual(cmodel.rhs, 1)

            elif cname == 'c2':
                self.assertAlmostEqual(cmodel.lhs.get_linear('i0'), 0,
                                       msg='constraint c2, linear(i0) should be 0')
                self.assertAlmostEqual(cmodel.lhs.get_quadratic('i0', 'i0'), 1,
                                       msg='quad(i0, i0) should be 1')
                self.assertAlmostEqual(cmodel.lhs.offset, 0,
                                       msg='constraint c2, offset should be 0')
                self.assertTrue(cmodel.sense == Sense.Ge,
                                msg='constraint c2, should be >= inequality')
                self.assertAlmostEqual(cmodel.rhs, 25)

            else:
                raise KeyError('Not expected constraint: {}'.format(cname))

    def test_nameless_objective(self):

        filepath = path.join(path.dirname(path.abspath(__file__)), 'data', 'test_nameless_objective.lp')

        with open(filepath) as f:
            with self.assertWarns(DeprecationWarning):
                cqm = CQM.from_lp_file(f)

        self.assertEqual(len(cqm.variables), 3, msg='wrong number of variables')
        self.assertEqual(len(cqm.constraints), 0, msg='expected 0 constraints')

    def test_nameless_constraint(self):

        filepath = path.join(path.dirname(path.abspath(__file__)), 'data', 'test_nameless_constraint.lp')

        with open(filepath) as f:
            with self.assertWarns(DeprecationWarning):
                cqm = CQM.from_lp_file(f)

        self.assertEqual(len(cqm.variables), 3, msg='wrong number of variables')
        self.assertEqual(len(cqm.constraints), 1, msg='expected 1 constraint')

    def test_empty_objective(self):

        # test case where Objective section is missing. This is allowed in LP format,
        # see https://www.gurobi.com/documentation/9.1/refman/lp_format.html)
        filepath = path.join(path.dirname(path.abspath(__file__)), 'data', 'test_empty_objective.lp')

        with open(filepath) as f:
            with self.assertWarns(DeprecationWarning):
                cqm = CQM.from_lp_file(f)

        self.assertEqual(len(cqm.variables), 2, msg='wrong number of variables')
        self.assertEqual(len(cqm.constraints), 1, msg='wrong number of constraints')

        # check that the objective is empty
        self.assertAlmostEqual(cqm.objective.get_linear('x0'), 0,
                               msg=' linear(x0) should be 0')
        self.assertAlmostEqual(cqm.objective.get_linear('x1'), 0,
                               msg=' linear(i0) should be 0')
        for cname, cmodel in cqm.constraints.items():
            if cname == 'c1':
                self.assertAlmostEqual(cmodel.lhs.get_linear('x0'), 1,
                                       msg='constraint c1, linear(x0) should be 1')
                self.assertAlmostEqual(cmodel.lhs.get_linear('x1'), 1,
                                       msg='constraint c1, linear(x1) should be 1')
                self.assertAlmostEqual(cmodel.lhs.offset, 0,
                                       msg='constraint c1, offset should be 0')
                self.assertTrue(cmodel.sense == Sense.Eq,
                                msg='constraint c1, should be equality')
                self.assertAlmostEqual(cmodel.rhs, 1)

            else:
                raise KeyError('Not expected constraint: {}'.format(cname))

    def test_empty_constraint(self):

        filepath = path.join(path.dirname(path.abspath(__file__)), 'data', 'test_empty_constraint.lp')

        with open(filepath, 'r') as f:
            with self.assertWarns(DeprecationWarning):
                with self.assertRaises(ValueError):
                    cqm = CQM.from_lp_file(f)

    def test_quadratic_binary(self):

        filepath = path.join(path.dirname(path.abspath(__file__)), 'data', 'test_quadratic_binary.lp')

        with open(filepath) as f:
            with self.assertWarns(DeprecationWarning):
                cqm = CQM.from_lp_file(f)

        self.assertEqual(len(cqm.variables), 1, msg='wrong number of variables')
        self.assertEqual(len(cqm.constraints), 1, msg='expected 1 constraint')

        self.assertAlmostEqual(cqm.objective.get_linear('x0'), 1.5,
                               msg=' linear(x0) should be 1.5')

        for cname, cmodel in cqm.constraints.items():
            if cname == 'c1':
                self.assertAlmostEqual(cmodel.lhs.get_linear('x0'), 2,
                                       msg='constraint c1, linear(x0) should be 2')

    def test_variable_multiple_times(self):

        filepath = path.join(path.dirname(path.abspath(__file__)), 'data', 'test_variable_multiple_times.lp')

        with open(filepath) as f:
            with self.assertWarns(DeprecationWarning):
                cqm = CQM.from_lp_file(f)

        self.assertEqual(len(cqm.variables), 1, msg='wrong number of variables')
        self.assertEqual(len(cqm.constraints), 1, msg='expected 1 constraint')

        self.assertAlmostEqual(cqm.objective.get_linear('x0'), 3,
                               msg=' linear(x0) should be 3')

        for cname, cmodel in cqm.constraints.items():
            if cname == 'c1':
                self.assertAlmostEqual(cmodel.lhs.get_linear('x0'), 3.5,
                                       msg='constraint c1, linear(x0) should be 3.5')


class TestIterConstraintData(unittest.TestCase):
    def test_iteration_order(self):
        cqm = CQM()
        x, y, z = dimod.Binaries(['x', 'y', 'z'])
        cqm.set_objective(x*y + 2*y*z)
        for a, b, c in itertools.permutations([x, y, z]):
            cqm.add_constraint(a * (b - c) <= 0)
        self.assertEqual(len(cqm.constraints), 6)
        self.assertEqual(list(cqm.constraints.keys()),
            [datum.label for datum in cqm.iter_constraint_data({'x': 1, 'y': 0, 'z': 0})])


class TestStr(unittest.TestCase):
    def test_one(self):
        cqm = CQM()

        m, n = dimod.Binaries(['m', 'n'])
        j = dimod.Integer('j')
        x, y = dimod.Reals(['x', 'y'])

        cqm.set_objective(m*j + 2*n + x)
        cqm.add_constraint(m * (j - n) <= 0, label='c0')
        cqm.add_constraint(n * j + 2 * y >= 1, label='c1')
        cqm.set_lower_bound('j', -10)
        cqm.set_upper_bound('j', 11)
        cqm.set_lower_bound('y', -1.1)
        cqm.set_upper_bound('y', 1.2)

        self.assertEqual(str(cqm), dedent(
            '''
            Objective
              2*Binary('n') + Real('x') + 0*Real('y') + Binary('m')*Integer('j')

            Constraints
              c0: Binary('m')*Integer('j') - Binary('m')*Binary('n') <= 0
              c1: 2*Real('y') + Binary('n')*Integer('j') >= 1

            Bounds
              -10.0 <= Integer('j') <= 11.0
              0.0 <= Real('x') <= 1e+30
              -1.1 <= Real('y') <= 1.2
            ''').lstrip())

    def test_spin_variables_have_no_bounds(self):
        cqm = CQM()
        cqm.set_objective(BQM({'a': 1, 'b': 1}, {}, 'SPIN'))

        self.assertEqual(str(cqm), dedent(
            '''
            Objective
              Spin('a') + Spin('b')

            Constraints

            Bounds
            ''').lstrip())
