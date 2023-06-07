# Copyright 2020 D-Wave Systems Inc.
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
import numbers
import shutil
import tempfile
import unittest

import dimod
import numpy as np

from parameterized import parameterized


# will want to migrate this to generators at some point
def gnp_random_dqm(num_variables, num_cases, p, p_case, seed=None):
    rng = np.random.default_rng(seed)

    dqm = dimod.DiscreteQuadraticModel()

    if isinstance(num_cases, numbers.Integral):
        num_cases = np.full(num_variables, fill_value=num_cases)
    else:
        num_cases = np.asarray(num_cases)
        if num_cases.shape[0] != num_variables:
            raise ValueError

    for nc in num_cases:
        v = dqm.add_variable(nc)
        dqm.set_linear(v, rng.uniform(size=dqm.num_cases(v)))

    for u, v in itertools.combinations(range(num_variables), 2):
        if rng.uniform() < p:
            size = (dqm.num_cases(u), dqm.num_cases(v))

            r = rng.uniform(size=size)
            r[rng.binomial(1, 1-p_case, size=size) == 1] = 0

            dqm.set_quadratic(u, v, r)

    return dqm


class TestBug(unittest.TestCase):
    def test_778(self):
        # https://github.com/dwavesystems/dimod/issues/778
        dqm = dimod.DQM()

        variables = np.array([0, 1, 2], dtype=np.int64)

        for variable in variables:
            dqm.add_variable(num_cases=5, label=variable)

        self.assertEqual(dqm.num_cases(variables[0]), 5)

    def test_782(self):
        # https://github.com/dwavesystems/dimod/issues/782
        rng = np.random.default_rng(782)

        dqm1 = dimod.DiscreteQuadraticModel()
        x = {}
        for i in range(10):
            x[i] = dqm1.add_variable(5)
        for c in range(5):
            dqm1.add_linear_equality_constraint(((x[i], c, rng.normal())
                                             for i in range(10)),
                                             lagrange_multiplier=1.0,
                                             constant=-1.0)
        dqm2 = dqm1.copy()

        state = {v: rng.integers(0, dqm1.num_cases(v)) for v in
                 dqm1.variables}

        self.assertEqual(dqm1.energy(state),
                         dqm2.energy(state))


class TestConstruction(unittest.TestCase):
    def test_empty(self):
        dqm = dimod.DQM()

        self.assertEqual(dqm.num_variables(), 0)
        self.assertEqual(dqm.variables, [])
        self.assertEqual(dqm.adj, {})
        self.assertEqual(dqm.num_cases(), 0)
        self.assertEqual(dqm.offset, 0)

    def test_one_variable(self):
        dqm = dimod.DQM()

        v = dqm.add_variable(10)

        self.assertEqual(v, 0)
        self.assertEqual(dqm.num_variables(), 1)
        self.assertEqual(dqm.variables, [0])
        self.assertEqual(dqm.adj, {0: set()})
        self.assertEqual(dqm.num_cases(), 10)
        self.assertEqual(dqm.num_cases(0), 10)

    def test_offset(self):
        dqm = dimod.DQM()
        dqm.add_variable(2)
        dqm.set_linear(0, [1,1])
        initial_energy = dqm.energy([0])

        dqm.offset = 10
        self.assertEqual(dqm.offset, 10)
        self.assertEqual(
            dqm.energy([0]), initial_energy + dqm.offset
        )

    def test_one_labelled(self):
        dqm = dimod.DQM()

        v = dqm.add_variable(10, label='a')

        self.assertEqual(v, 'a')
        self.assertEqual(dqm.num_variables(), 1)
        self.assertEqual(dqm.num_cases(), 10)
        self.assertEqual(dqm.num_cases('a'), 10)
        with self.assertRaises(ValueError):
            dqm.num_cases(0)

    def test_two(self):
        dqm = dimod.DQM()

        u = dqm.add_variable(10)
        v = dqm.add_variable(5)

        self.assertEqual(u, 0)
        self.assertEqual(v, 1)
        self.assertEqual(dqm.num_variables(), 2)
        self.assertEqual(dqm.num_cases(), 15)
        self.assertEqual(dqm.num_cases(u), 10)
        self.assertEqual(dqm.num_cases(v), 5)

        with self.assertRaises(ValueError):
            dqm.num_cases(2)

    def test_second_unlabelled(self):
        dqm = dimod.DQM()

        u = dqm.add_variable(4, label='a')
        v = dqm.add_variable(3)

        self.assertEqual(dqm.variables, ['a', 1])

    def test_second_unlabelled_conflict(self):
        dqm = dimod.DQM()

        u = dqm.add_variable(4, label=1)
        v = dqm.add_variable(3)
        v = dqm.add_variable(3)

        self.assertEqual(dqm.variables, [1, 0, 2])


class TestCopy(unittest.TestCase):
    def test_simple(self):
        dqm = dimod.DQM()
        u = dqm.add_variable(4)
        v = dqm.add_variable(3)
        dqm.set_quadratic(u, v, {(0, 2): -1, (2, 1): 1})
        dqm.set_linear(u, [0, 1, 2, 3])

        new = dqm.copy()

        self.assertIsNot(dqm, new)
        self.assertIsInstance(new, type(dqm))
        np.testing.assert_array_equal(dqm.get_linear(u), new.get_linear(u))
        np.testing.assert_array_equal(dqm.get_linear(v), new.get_linear(v))
        self.assertEqual(dqm.get_quadratic(u, v), new.get_quadratic(u, v))

        new.set_linear(u, [3, 2, 1, 0])
        np.testing.assert_array_equal(dqm.get_linear(u), [0, 1, 2, 3])

        new.add_variable(5)
        self.assertEqual(new.num_variables(), dqm.num_variables() + 1)
        self.assertEqual(new.num_cases(), dqm.num_cases() + 5)


class TestEnergy(unittest.TestCase):
    def test_one_variable(self):
        dqm = dimod.DQM()
        u = dqm.add_variable(4)

        for s in range(4):
            sample = [s]
            self.assertEqual(dqm.energy(sample), 0.0)

        np.testing.assert_array_equal(dqm.energies([[s] for s in range(4)]),
                                      np.zeros(4))

    def test_two_variable(self):
        dqm = dimod.DQM()
        u = dqm.add_variable(3)
        v = dqm.add_variable(4, label='hello')

        dqm.set_linear_case(v, 3, 1.5)
        dqm.set_quadratic(u, v, {(0, 1): 5, (2, 0): 107})

        samples = list(itertools.product(range(3), range(4)))

        energies = dqm.energies((samples, [0, 'hello']))

        np.testing.assert_array_equal([0.0, 5.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.5,
                                       107.0, 0.0, 0.0, 1.5], energies)

    def test_two_variable_labelled(self):
        dqm = dimod.DQM()
        u = dqm.add_variable(10)
        v = dqm.add_variable(5, label='v')

        dqm.set_quadratic(u, v, {(0, 1): 1})
        dqm.set_linear_case(u, 0, 1.5)

        self.assertEqual(dqm.energy({u: 0, v: 1}), 2.5)

    def test_two_variable_labeled_permutations(self):

        dqm = gnp_random_dqm(4, [2, 4, 1, 4], .5, .5, seed=7)
        dqm.relabel_variables({0: 'a', 2: (3, 4)})

        sample = {'a': 1, 1: 3, (3, 4): 0, 3: 3}

        energies = set()
        for order in itertools.permutations(dqm.variables):
            s = ([sample[v] for v in order], order)
            en = dqm.energy(s)
            energies.add(en)

        self.assertEqual(len(energies), 1)


# we use these for paramaterized testing of to_file and from_file
_empty = dimod.DQM()

_twovar = dimod.DQM()
_twovar.add_variable(5)
_twovar.add_variable(7)
_twovar.set_linear_case(0, 3, 1.5)
_twovar.set_quadratic(0, 1, {(0, 1): 1.5, (3, 4): 1})

_random = gnp_random_dqm(5, [4, 5, 2, 1, 10], .5, .5, seed=23)

_kwargs = [('default', dict()),
           ('spool_size=0', dict(spool_size=0)),
           ('spool_size=10', dict(spool_size=10)),
           ('compress=True', dict(compress=True)),
           ('compress=False', dict(compress=False)),
           ('spool_size=0,compress=True', dict(spool_size=0, compress=True))
           ]

file_parameterized = [('_'.join([dname, kname]), dqm, kwargs)
                      for dname, dqm in [('empty', _empty),
                                         ('twovar', _twovar),
                                         ('random', _random)]
                      for kname, kwargs in _kwargs]


class TestFile(unittest.TestCase):

    def assertDQMEqual(self, dqm0, dqm1):
        self.assertEqual(dqm1.num_variables(), dqm0.num_variables())
        self.assertEqual(dqm1.num_cases(), dqm0.num_cases())
        self.assertEqual(dqm1.variables, dqm0.variables)
        for v in dqm0.variables:
            self.assertEqual(dqm1.num_cases(v), dqm0.num_cases(v))
            np.testing.assert_array_equal(dqm1.get_linear(v),
                                          dqm0.get_linear(v))
        self.assertEqual(dqm1.adj, dqm0.adj)
        self.assertEqual(dqm1._cydqm.adj,
                         dqm0._cydqm.adj)  # implementation detail
        for u in dqm0.adj:
            for v in dqm0.adj[u]:
                self.assertEqual(dqm0.get_quadratic(u, v),
                                 dqm1.get_quadratic(u, v))

    def test_bug(self):
        # https://github.com/dwavesystems/dimod/issues/730
        # this is technically an internal attribute, but none-the-less has
        # surprising behavior
        dqm = dimod.DiscreteQuadraticModel()
        with dqm.to_file() as f:
            f._file.read()

    def test_compress(self):
        dqm = gnp_random_dqm(5, [4, 5, 2, 1, 10], .5, .5, seed=23)

        with dqm.to_file(compress=True) as cf:
            with dqm.to_file() as f:
                self.assertLess(len(cf.read()), len(f.read()))

    def test_compressed(self):
        # deprecated
        dqm = gnp_random_dqm(5, [4, 5, 2, 1, 10], .5, .5, seed=23)

        with self.assertWarns(DeprecationWarning):
            with dqm.to_file(compressed=True) as cf:
                with dqm.to_file() as f:
                    self.assertLess(len(cf.read()), len(f.read()))

    @parameterized.expand(file_parameterized)
    def test_functional(self, name, dqm, kwargs):
        with dqm.to_file(**kwargs) as f:
            new = dimod.DQM.from_file(f)
        self.assertDQMEqual(dqm, new)

    @parameterized.expand(file_parameterized)
    def test_functional_buffer(self, name, dqm, kwargs):
        with dqm.to_file(**kwargs) as f:
            new = dimod.DQM.from_file(f.read())
        self.assertDQMEqual(dqm, new)

    @parameterized.expand(file_parameterized)
    def test_functional_tempfile(self, name, dqm, kwargs):
        with tempfile.TemporaryFile() as tf:
            with dqm.to_file() as df:
                shutil.copyfileobj(df, tf)
            tf.seek(0)
            new = dimod.DQM.from_file(tf)
        self.assertDQMEqual(dqm, new)

    def test_readable(self):
        with dimod.DQM().to_file() as f:
            self.assertTrue(f.readable())

    def test_readinto(self):
        dqm = dimod.DQM()
        dqm.add_variable(5)
        dqm.add_variable(7)

        dqm.set_linear_case(0, 3, 1.5)
        dqm.set_quadratic(0, 1, {(0, 1): 1.5, (3, 4): 1})

        with dqm.to_file() as f:
            buff = f.read()

        with dqm.to_file() as f:
            buff2 = bytearray(len(buff))
            f.readinto(buff2)

        self.assertEqual(buff, buff2)

    def test_seekable(self):
        with dimod.DQM().to_file() as f:
            self.assertTrue(f.seekable())

    def test_writeable(self):
        with dimod.DQM().to_file() as f:
            self.assertTrue(f.writable())

    def test_case_label(self):
        cldqm = dimod.CaseLabelDQM()
        cldqm.add_variable('abcde')
        cldqm.add_variable('fghijkl')

        cldqm.set_linear('d', 1.5)
        cldqm.set_quadratic('a', 'g', 1.5)
        cldqm.set_quadratic('d', 'j', 1)

        dqm = dimod.DQM()
        dqm.add_variable(5)
        dqm.add_variable(7)

        dqm.set_linear_case(0, 3, 1.5)
        dqm.set_quadratic(0, 1, {(0, 1): 1.5, (3, 4): 1})

        with cldqm.to_file(ignore_labels=True) as f:
            self.assertDQMEqual(dimod.DQM.from_file(f), dqm)


class TestLinear(unittest.TestCase):
    def test_set_linear_case(self):
        dqm = dimod.DQM()
        dqm.add_variable(5, 'a')

        np.testing.assert_array_equal(dqm.get_linear('a'), [0, 0, 0, 0, 0])

        dqm.set_linear_case('a', 1, 1.5)
        dqm.set_linear_case('a', 2, 4)

        self.assertEqual(dqm.get_linear_case('a', 0), 0)
        self.assertEqual(dqm.get_linear_case('a', 1), 1.5)
        self.assertEqual(dqm.get_linear_case('a', 2), 4)
        np.testing.assert_array_equal(dqm.get_linear('a'), [0, 1.5, 4, 0, 0])

    def test_set_linear(self):
        dqm = dimod.DQM()
        dqm.add_variable(5, 'a')

        np.testing.assert_array_equal(dqm.get_linear('a'), [0, 0, 0, 0, 0])

        with self.assertRaises(ValueError):
            dqm.set_linear('a', [0])

        dqm.set_linear('a', [0, 1.5, 4, 0, 0])

        self.assertEqual(dqm.get_linear_case('a', 0), 0)
        self.assertEqual(dqm.get_linear_case('a', 1), 1.5)
        self.assertEqual(dqm.get_linear_case('a', 2), 4)
        np.testing.assert_array_equal(dqm.get_linear('a'), [0, 1.5, 4, 0, 0])

    @parameterized.expand(
        [(np.dtype(dt).name, dt)
         for dt in [np.int8, np.int16, np.int32, np.int64,
                    np.uint8, np.uint16, np.uint32, np.uint64,
                    np.float32, np.float64]])
    def test_set_linear_offdtype(self, name, dtype):
        dqm = dimod.DQM()
        v = dqm.add_variable(5)
        dqm.set_linear(v, np.zeros(5, dtype=dtype))


class TestQuadratic(unittest.TestCase):
    def test_self_loop(self):
        dqm = dimod.DQM()
        u = dqm.add_variable(5)
        v = dqm.add_variable(5)

        with self.assertRaises(ValueError):
            dqm.set_quadratic(u, u, {})
        with self.assertRaises(ValueError):
            dqm.set_quadratic_case(u, 0, u, 1, .5)

    def test_set_quadratic_case(self):
        dqm = dimod.DQM()
        dqm.add_variable(5, label='a')
        dqm.add_variable(10, label='b')

        with self.assertRaises(ValueError):
            dqm.get_quadratic('a', 'b')

        dqm.set_quadratic('a', 'b', {})

        self.assertEqual(dqm.get_quadratic('a', 'b'), {})
        np.testing.assert_array_equal(dqm.get_quadratic('a', 'b', array=True),
                                      np.zeros((5, 10)))

        for ca in range(5):
            for cb in range(10):
                self.assertEqual(dqm.get_quadratic_case('a', ca, 'b', cb), 0)

        dqm.set_quadratic_case('a', 2, 'b', 8, .5)
        dqm.set_quadratic_case('b', 3, 'a', 4, -7)

        self.assertEqual(dqm.get_quadratic('a', 'b'), {(2, 8): .5, (4, 3): -7})
        self.assertEqual(dqm.get_quadratic('b', 'a'), {(8, 2): .5, (3, 4): -7})

        self.assertEqual(dqm.get_quadratic_case('a', 2, 'b', 8), .5)

        arr_ab = dqm.get_quadratic('a', 'b', array=True)
        arr_ba = dqm.get_quadratic('b', 'a', array=True)

        self.assertEqual(arr_ab.shape, (5, 10))
        self.assertEqual(arr_ba.shape, (10, 5))

        arr = np.zeros((5, 10))
        arr[2, 8] = .5
        arr[4, 3] = -7
        np.testing.assert_array_equal(arr, arr_ab)

        arr = np.zeros((10, 5))
        arr[8, 2] = .5
        arr[3, 4] = -7
        np.testing.assert_array_equal(arr, arr_ba)

    def test_set_quadratic_dense(self):
        dqm = dimod.DQM()
        u = dqm.add_variable(3)
        v = dqm.add_variable(2)

        dqm.set_quadratic(u, v, [[0, 1], [1.5, 6], [2, 0]])

        np.testing.assert_array_equal(dqm.get_quadratic(u, v, array=True),
                                      [[0, 1], [1.5, 6], [2, 0]])

        self.assertEqual(dqm.get_quadratic(u, v),
                         {(0, 1): 1, (1, 0): 1.5, (1, 1): 6, (2, 0): 2})

    @parameterized.expand(
        [(np.dtype(dt).name, dt)
         for dt in [np.int8, np.int16, np.int32, np.int64,
                    np.uint8, np.uint16, np.uint32, np.uint64,
                    np.float32, np.float64]])
    def test_set_quadratic_offdtype(self, name, dtype):
        dqm = dimod.DQM()
        u = dqm.add_variable(3)
        v = dqm.add_variable(2)

        biases = np.zeros((3, 2), dtype=dtype)

        dqm.set_quadratic(u, v, biases)


class TestConstraint(unittest.TestCase):
    def test_simple_constraint(self):
        dqm = dimod.DQM()
        num_variables = 2
        num_cases = 3
        x = {}
        for i in range(num_variables):
            x[i] = dqm.add_variable(num_cases, label='x_{i}'.format(i=i))

        for c in range(num_cases):
            dqm.add_linear_equality_constraint(
                [(x[i], c, 1.0) for i in range(num_variables)],
                lagrange_multiplier=1.0, constant=-1.0)

        for i in x:
            for case in range(num_cases):
                self.assertEqual(dqm.get_linear_case(x[i], case), -1)
            for j in x:
                if j > i:
                    for case in range(num_cases):
                        self.assertEqual(dqm.get_quadratic_case(x[i], case, x[j], case), 2.0)

    def test_more_constraint(self):
        dqm = dimod.DQM()
        x = dqm.add_variable(5, label='x')
        y = dqm.add_variable(3, label='y')
        w = dqm.add_variable(4, label='w')

        expression = [(x, 1, 1.0), (y, 2, 2.0), (w, 3, 1.0)]
        constant = -2.0
        dqm.add_linear_equality_constraint(
            expression,
            lagrange_multiplier=1.0, constant=constant)

        expression_dict = {v: (c, b) for v, c, b in expression}
        for cx, cy, cw in itertools.product(range(5), range(3), range(4)):
            s = constant
            state = {'x': cx, 'y': cy, 'w': cw}
            for v, cv, bias in expression:
                if expression_dict[v][0] == state[v]:
                    s += bias
            self.assertAlmostEqual(dqm.energy(state), s ** 2)

    def test_inequality_constraint_empty(self):
        dqm = dimod.DQM()
        dqm.add_variable(5, label='x')
        dqm.add_variable(3, label='y')
        dqm.add_variable(4, label='w')
        expression = [('x', 1, -7.0), ('y', 2, -2.0), ('w', 3, 4.0)]
        constant = 0
        num_dqm_vars = dqm.num_variables()
        self.assertRaises(ValueError, dqm.add_linear_inequality_constraint,
                          expression,
                          lagrange_multiplier=1.0,
                          constant=constant,
                          label='inequality_0',
                          slack_method="log2",
                          lb=-40,
                          ub=-30)
        self.assertTrue(dqm.num_variables() == num_dqm_vars)

        self.assertWarns(Warning, dqm.add_linear_inequality_constraint,
                         expression,
                         lagrange_multiplier=1.0,
                         constant=5,
                         label='inequality_0',
                         slack_method="log2",
                         lb=-50,
                         ub=30)

        self.assertTrue(dqm.num_variables() == num_dqm_vars)

    def test_inequality_constraint_equality(self):
        dqm = dimod.DQM()

        expr = [('a', 1, 1), ('b', 2, 1),
                ('c', 3, 1)]
        # we want 1 <= sum(expr) <= 1
        for i, j, k in expr:
            if i not in dqm.variables:
                dqm.add_variable(4, i)

        dqm1 = dqm.copy()
        dqm_equal = dqm.copy()

        slack_terms = \
            dqm.add_linear_inequality_constraint(expr,
                                                 constant=0,
                                                 lb=1,
                                                 ub=1,
                                                 lagrange_multiplier=1,
                                                 label="a")
        self.assertEqual(len(slack_terms), 0)

        slack_terms =\
            dqm1.add_linear_inequality_constraint(expr,
                                                  constant=-1,
                                                  lb=0,
                                                  ub=0,
                                                  lagrange_multiplier=1,
                                                  label="a")
        self.assertTrue(len(slack_terms) == 0)

        dqm_equal.add_linear_equality_constraint(expr,
                                                 constant=-1,
                                                 lagrange_multiplier=1)
        for j in [0, 1]:
            with self.assertWarns(DeprecationWarning):
                check = dqm.to_numpy_vectors()[j] - dqm_equal.to_numpy_vectors()[j]
            with self.assertWarns(DeprecationWarning):
                check1 = \
                    dqm1.to_numpy_vectors()[j] - dqm_equal.to_numpy_vectors()[j]
            for i in range(len(check)):
                self.assertAlmostEqual(check[i], 0)
                self.assertAlmostEqual(check1[i], 0)

        for k in range(3):
            with self.assertWarns(DeprecationWarning):
                check = dqm.to_numpy_vectors()[2][k] -\
                         dqm_equal.to_numpy_vectors()[2][k]
            with self.assertWarns(DeprecationWarning):
                check1 = dqm1.to_numpy_vectors()[2][k] -\
                         dqm_equal.to_numpy_vectors()[2][k]
            for i in range(len(check)):
                self.assertAlmostEqual(check[i], 0)
                self.assertAlmostEqual(check1[i], 0)

    def test_inequality_constraint_log2(self):
        dqm = dimod.DQM()
        dqm.add_variable(5, label='x')
        dqm.add_variable(3, label='y')
        dqm.add_variable(4, label='w')
        expression_list = [('x', 1, -7.0), ('y', 2, -2.0), ('w', 3, 4.0)]
        expression_iter = iter(expression_list)
        constant = 0
        ub = -3
        slack_terms = dqm.add_linear_inequality_constraint(
            expression_iter,
            lagrange_multiplier=1.0,
            constant=constant,
            label='inequality_0',
            slack_method="log2",
            ub=ub)

        expression_dict = {v: (c, b) for v, c, b in expression_list +
                           slack_terms}
        for cx, cy, cw, cs1, cs2, cs3 in itertools.product(
                range(5), range(3), range(4), range(2), range(2), range(2)):
            s = -ub-constant
            state = {'x': cx, 'y': cy, 'w': cw,
                     'slack_inequality_0_0': cs1, 'slack_inequality_0_1': cs2,
                     'slack_inequality_0_2': cs3}
            for v, cv, bias in expression_list + slack_terms:
                if expression_dict[v][0] == state[v]:
                    s += bias
            self.assertAlmostEqual(dqm.energy(state), s ** 2)

    def test_inequality_cross_zero(self):
        lbs = [-9, 4]
        ubs = [-4, 14]
        expressions = [-69.0, 69.0]

        for i, method in enumerate(['log2', 'log10']):
            lb = lbs[i]
            ub = ubs[i]
            dqm = dimod.DQM()
            dqm.add_variable(5, label='x')
            expression = [('x', 1, expressions[i])]
            constant = 1
            slack_terms = dqm.add_linear_inequality_constraint(
                expression,
                lagrange_multiplier=1.0,
                constant=constant,
                label='ineq',
                slack_method=method,
                lb=lb,
                ub=ub,
                cross_zero=True)
            self.assertTrue(slack_terms[-1][2] == ub - constant)

    def test_inequality_constraint_log10(self):
        dqm = dimod.DQM()
        dqm.add_variable(5, label='x')
        dqm.add_variable(3, label='y')
        dqm.add_variable(4, label='w')

        expression = [('x', 1, -7.0), ('y', 2, -2.0), ('w', 3, 4.0)]

        constant = 0
        ub = -3
        slack_terms = dqm.add_linear_inequality_constraint(
            expression,
            lagrange_multiplier=1.0,
            constant=constant,
            ub=ub,
            label='ineq_0',
            slack_method="log10")

        expression_dict = {(v, c): (c, b) for v, c, b in expression +
                           slack_terms}
        for cx, cy, cw, cs in itertools.product(range(5), range(3), range(4),
                                                range(7)):
            s = -ub - constant
            state = {'x': cx, 'y': cy, 'w': cw, 'slack_ineq_0_0': cs}
            for v, cv, bias in expression + slack_terms:
                if expression_dict[v, cv][0] == state[v]:
                    s += bias
            self.assertAlmostEqual(dqm.energy(state), s ** 2)

    def test_inequality_constraint_log10_3dqm(self):
        dqm = dimod.DQM()
        dqm.add_variable(5, label='x')
        expression = [('x', 1, -621.0)]
        constant = 3
        ub = 0
        slack_terms = dqm.add_linear_inequality_constraint(
            expression,
            lagrange_multiplier=1.0,
            constant=constant,
            ub=ub,
            label='ineq_0',
            slack_method="log10")

        expression_dict = {(v, c):
                               (c, b) for v, c, b in expression + slack_terms}
        for cx, cs1, cs2, cs3 in itertools.product(
                range(5), range(10), range(10), range(7)):
            s = constant
            state = {'x': cx, 'slack_ineq_0_0': cs1, 'slack_ineq_0_1': cs2,
                     'slack_ineq_0_2': cs3}
            for v, cv, bias in expression + slack_terms:
                if expression_dict[v, cv][0] == state[v]:
                    s += bias
            self.assertAlmostEqual(dqm.energy(state), s ** 2)

    def test_inequality_constraint_linear(self):
        dqm = dimod.DQM()
        dqm.add_variable(2, label='x')
        dqm.add_variable(2, label='y')
        dqm.add_variable(2, label='w')

        expression = [('x', 1, 10.0), ('y', 1, 10.0), ('w', 1, 10.0)]

        constant = 0
        ub = 28
        lb = 3
        slack_terms = dqm.add_linear_inequality_constraint(
            expression,
            lagrange_multiplier=1.0,
            constant=constant,
            ub=ub,
            lb=lb,
            label='c',
            cross_zero=True,
            slack_method="linear")

        for i, (v, c, val) in enumerate(slack_terms):
            if i < 25:
                self.assertEqual(i+1, val)
            else:
                self.assertEqual(val, ub)

    def test_random_constraint(self):
        num_variables = 4
        cases = np.random.randint(3, 6, size=num_variables)
        dqm_0 = gnp_random_dqm(num_variables, cases, 0.5, 0.5, seed=123)
        # copy doesn't work properly, so for now create the same dqm twice
        dqm = gnp_random_dqm(num_variables, cases, 0.5, 0.5, seed=123)
        x = dqm.variables

        expression = [(x[i], np.random.randint(0, cases[i]), np.random.randint(0, 10)) for i in x]
        constant = np.random.randint(1, 10) * num_variables
        lagrange_multiplier = np.random.randint(1, 10)
        dqm.add_linear_equality_constraint(
            expression,
            lagrange_multiplier=lagrange_multiplier, constant=constant)

        expression_dict = {v: (c, b) for v, c, b in expression}
        for case_values in itertools.product(*(range(c) for c in cases)):
            state = {x[i]: case_values[i] for i in x}
            s = constant
            for v, cv, bias in expression:
                if expression_dict[v][0] == state[v]:
                    s += bias

            self.assertAlmostEqual(dqm.energy(state), lagrange_multiplier * s ** 2 + dqm_0.energy(state))

    def test_unknown_variable(self):
        dqm = dimod.DQM()
        with self.assertRaises(ValueError):
            dqm.add_linear_equality_constraint(
                [(0, 0, 0)], lagrange_multiplier=1, constant=-1)

    def test_out_of_range_case(self):
        dqm = dimod.DQM()
        u = dqm.add_variable(5)
        with self.assertRaises(ValueError):
            dqm.add_linear_equality_constraint(
                [(u, 6, 0)], lagrange_multiplier=1, constant=-1)

    def test_self_loop(self):
        dqm = dimod.DQM()
        u = dqm.add_variable(5)
        v = dqm.add_variable(6)

        terms = [(u, 0, .5), (v, 1, .5), (u, 2, .5)]

        dqm.add_linear_equality_constraint(
            terms, lagrange_multiplier=1, constant=1)

        # because two cases within the same variable are mentioned, we can
        # discard that interaction
        self.assertEqual(dqm.num_case_interactions(), 2)

    def test_self_loop_repeat(self):
        num_variables = 4
        cases = [4, 3, 3, 5]
        dqm = dimod.DiscreteQuadraticModel()
        for i in range(num_variables):
            dqm.add_variable(cases[i], i)

        expression1 = [(0, 0, 2), (1, 0, 2), (2, 1, 0), (3, 0, 8)]
        expression2 = [(0, 1, 2), (1, 1, 2), (2, 0, 0), (3, 1, 8)]
        expression = expression1 + expression2
        lagrange_multiplier = 6
        constant = 4
        dqm.add_linear_equality_constraint(
            expression,
            lagrange_multiplier=lagrange_multiplier,
            constant=constant)

        expression_dict1 = {v: (c, b) for v, c, b in expression1}
        expression_dict2 = {v: (c, b) for v, c, b in expression2}

        for case_values in itertools.product(*(range(c) for c in cases)):
            state = {i: case_values[i] for i in range(num_variables)}
            s = constant
            for v, cv, bias in expression1:
                if expression_dict1[v][0] == state[v]:
                    s += bias
            for v, cv, bias in expression2:
                if expression_dict2[v][0] == state[v]:
                    s += bias
            self.assertEqual(dqm.energy(state), lagrange_multiplier * s ** 2)

    def test_self_loop_repeat2(self):
        num_variables = 4
        cases = [4, 3, 3, 5]
        dqm = dimod.DiscreteQuadraticModel()
        for i in range(num_variables):
            dqm.add_variable(cases[i], i)

        expression1 = [(0, 0, 2)]
        expression2 = [(0, 0, 2)]
        expression = expression1 + expression2
        lagrange_multiplier = 6
        constant = 4
        dqm.add_linear_equality_constraint(
            expression,
            lagrange_multiplier=lagrange_multiplier,
            constant=constant)

        expression_dict1 = {v: (c, b) for v, c, b in expression1}
        expression_dict2 = {v: (c, b) for v, c, b in expression2}

        for case_values in itertools.product(*(range(c) for c in cases)):
            state = {i: case_values[i] for i in range(num_variables)}
            s = constant
            for v, cv, bias in expression1:
                if expression_dict1[v][0] == state[v]:
                    s += bias
            for v, cv, bias in expression2:
                if expression_dict2[v][0] == state[v]:
                    s += bias
            self.assertEqual(dqm.energy(state), lagrange_multiplier * s ** 2)

    def test_self_loop3(self):

        dqm1 = dimod.DiscreteQuadraticModel()
        dqm2 = dimod.DiscreteQuadraticModel()

        dqm1.add_variable(5)
        dqm2.add_variable(5)

        lagrange_multiplier = 1
        constant = 0
        dqm1.add_linear_equality_constraint(
            [(0, 0, 1), (0, 0, 2)],
            lagrange_multiplier=lagrange_multiplier,
            constant=constant)

        dqm2.add_linear_equality_constraint(
            [(0, 0, 3)],
            lagrange_multiplier=lagrange_multiplier,
            constant=constant)

        np.testing.assert_array_equal(dqm1.get_linear(0), dqm2.get_linear(0))


class TestNumpyVectors(unittest.TestCase):

    def test_empty_functional(self):
        dqm = dimod.DQM()
        with self.assertWarns(DeprecationWarning):
            new = dimod.DQM.from_numpy_vectors(*dqm.to_numpy_vectors())
        self.assertEqual(new.num_variables(), 0)

    def test_exceptions(self):
        # make a working DQM that we can break in various ways
        starts = [0, 3]
        ldata = [0, 1, 2, 3, 4]
        irow = [0, 0, 1, 2]
        icol = [3, 4, 3, 4]
        qdata = [-1, -2, -3, -4]
        quadratic = (irow, icol, qdata)

        dimod.DQM.from_numpy_vectors(starts, ldata, quadratic)  # smoke

        with self.subTest("badly ordered starts"):
            with self.assertRaises(ValueError):
                dimod.DQM.from_numpy_vectors([3, 0], ldata, quadratic)

        with self.subTest("case_starts inconsistent with linear_biases"):
            with self.assertRaises(ValueError):
                dimod.DQM.from_numpy_vectors([0, 10], ldata, quadratic)

        with self.subTest("inconsistent quadratic"):
            with self.assertRaises(ValueError):
                dimod.DQM.from_numpy_vectors(starts, ldata, ([], [0], [1]))

        with self.subTest("out-of-range icol"):
            with self.assertRaises(ValueError):
                dimod.DQM.from_numpy_vectors(starts, ldata, ([0], [105], [1]))

        with self.subTest("out-of-range irow"):
            with self.assertRaises(ValueError):
                dimod.DQM.from_numpy_vectors(starts, ldata, ([105], [0], [1]))

        with self.subTest("self-loop"):
            with self.assertRaises(ValueError):
                dimod.DQM.from_numpy_vectors(starts, ldata, ([0], [0], [1]))

    def test_selfloop(self):
        # should raise an exception when given a self-loop
        starts = [0, 3, 6]  # degree 3
        ldata = range(9)
        irow = []
        icol = []
        for ci in range(3):
            for cj in range(3, 9):
                irow.append(ci)
                icol.append(cj)
        for ci in range(3, 6):
            for cj in range(3):
                irow.append(ci)
                icol.append(cj)
            for cj in range(6, 9):
                irow.append(ci)
                icol.append(cj)
        for ci in range(6, 9):
            for cj in range(6):
                irow.append(ci)
                icol.append(cj)

        # now add a single self-loop
        irow.append(3)
        icol.append(4)

        qdata = [1]*len(irow)

        with self.assertRaises(ValueError):
            dimod.DQM.from_numpy_vectors(starts, ldata, (irow, icol, qdata))

    def test_two_var_functional(self):
        dqm = dimod.DQM()
        dqm.add_variable(5)
        dqm.add_variable(7)

        dqm.set_linear_case(0, 3, 1.5)
        dqm.set_quadratic(0, 1, {(0, 1): 1.5, (3, 4): 1})

        with self.assertWarns(DeprecationWarning):
            new = dimod.DQM.from_numpy_vectors(*dqm.to_numpy_vectors())

        self.assertEqual(new.num_variables(), dqm.num_variables())
        self.assertEqual(new.num_cases(), dqm.num_cases())
        self.assertEqual(new.variables, dqm.variables)
        for v in dqm.variables:
            self.assertEqual(new.num_cases(v), dqm.num_cases(v))
            np.testing.assert_array_equal(new.get_linear(v),
                                          dqm.get_linear(v))
        self.assertEqual(new.adj, dqm.adj)

    def test_two_var_functional_labelled(self):
        dqm = dimod.DQM()
        dqm.add_variable(5)
        dqm.add_variable(7, 'b')

        dqm.set_linear_case(0, 3, 1.5)
        dqm.set_quadratic(0, 'b', {(0, 1): 1.5, (3, 4): 1})

        with self.assertWarns(DeprecationWarning):
            vectors = dqm.to_numpy_vectors()

            new = dimod.DQM.from_numpy_vectors(*vectors)

            new_vectors = new.to_numpy_vectors()

        np.testing.assert_array_equal(vectors[0], new_vectors[0])
        np.testing.assert_array_equal(vectors[1], new_vectors[1])
        np.testing.assert_array_equal(vectors[2][0], new_vectors[2][0])
        np.testing.assert_array_equal(vectors[2][1], new_vectors[2][1])
        np.testing.assert_array_equal(vectors[2][2], new_vectors[2][2])

        self.assertEqual(new.num_variables(), dqm.num_variables())
        self.assertEqual(new.num_cases(), dqm.num_cases())
        self.assertEqual(new.variables, dqm.variables)
        for v in dqm.variables:
            self.assertEqual(new.num_cases(v), dqm.num_cases(v))
            np.testing.assert_array_equal(new.get_linear(v),
                                          dqm.get_linear(v))
        self.assertEqual(new.adj, dqm.adj)
        self.assertEqual(new._cydqm.adj,
                         dqm._cydqm.adj)  # implementation detail
        for u in dqm.adj:
            for v in dqm.adj[u]:
                self.assertEqual(dqm.get_quadratic(u, v),
                                 new.get_quadratic(u, v))

    def test_random(self):

        dqm = gnp_random_dqm(5, [4, 5, 2, 1, 10], .5, .5, seed=17)

        with self.assertWarns(DeprecationWarning):
            vectors = dqm.to_numpy_vectors()

        new = dimod.DQM.from_numpy_vectors(*vectors)

        with self.assertWarns(DeprecationWarning):
            new_vectors = new.to_numpy_vectors()
        np.testing.assert_array_equal(vectors[0], new_vectors[0])
        np.testing.assert_array_equal(vectors[1], new_vectors[1])
        np.testing.assert_array_equal(vectors[2][0], new_vectors[2][0])
        np.testing.assert_array_equal(vectors[2][1], new_vectors[2][1])
        np.testing.assert_array_equal(vectors[2][2], new_vectors[2][2])

        self.assertEqual(new.num_variables(), dqm.num_variables())
        self.assertEqual(new.num_cases(), dqm.num_cases())
        self.assertEqual(new.variables, dqm.variables)
        for v in dqm.variables:
            self.assertEqual(new.num_cases(v), dqm.num_cases(v))
            np.testing.assert_array_equal(new.get_linear(v),
                                          dqm.get_linear(v))
        self.assertEqual(new.adj, dqm.adj)
        self.assertEqual(new._cydqm.adj,
                         dqm._cydqm.adj)  # implementation detail
        for u in dqm.adj:
            for v in dqm.adj[u]:
                self.assertEqual(dqm.get_quadratic(u, v),
                                 new.get_quadratic(u, v))

    def test_random_shuffled_quadratic(self):

        dqm = gnp_random_dqm(5, [4, 5, 2, 1, 10], .5, .5, seed=17)

        with self.assertWarns(DeprecationWarning):
            vectors = dqm.to_numpy_vectors()

        # suffle the quadratic vectors so they are not ordered anymore
        starts, ldata, (irow, icol, qdata), labels = vectors

        shuffled = (starts, ldata,
                    (np.array(np.flip(icol), copy=True),
                     np.array(np.flip(irow), copy=True),
                     np.array(np.flip(qdata), copy=True)))

        new = dimod.DQM.from_numpy_vectors(*shuffled)

        self.assertEqual(new.num_variables(), dqm.num_variables())
        self.assertEqual(new.num_cases(), dqm.num_cases())
        self.assertEqual(new.variables, dqm.variables)
        for v in dqm.variables:
            self.assertEqual(new.num_cases(v), dqm.num_cases(v))
            np.testing.assert_array_equal(new.get_linear(v),
                                          dqm.get_linear(v))
        self.assertEqual(new.adj, dqm.adj)
        self.assertEqual(new._cydqm.adj,
                         dqm._cydqm.adj)  # implementation detail
        for u in dqm.adj:
            for v in dqm.adj[u]:
                self.assertEqual(dqm.get_quadratic(u, v),
                                 new.get_quadratic(u, v))


class TestRelabel(unittest.TestCase):
    def test_index(self):
        dqm = dimod.DQM()
        dqm.add_variable(5, 'a')
        dqm.add_variable(3, 'b')

        dqm, mapping = dqm.relabel_variables_as_integers()

        self.assertEqual(mapping, {0: 'a', 1: 'b'})

    def test_typical(self):
        dqm = dimod.DQM()
        dqm.add_variable(5, 'a')
        dqm.add_variable(3, 'b')

        new = dqm.relabel_variables({'a': 'b', 'b': 'a'}, inplace=False)

        self.assertEqual(new.variables, ['b', 'a'])


class TestCaseLabelDQM(unittest.TestCase):
    def test_conserved_behavior(self):
        dqm = dimod.CaseLabelDQM()
        u = dqm.add_variable(2)
        v = dqm.add_variable(3)

        # non-unique variable label
        with self.assertRaises(ValueError) as cm:
            dqm.add_variable(2, label=u)

        # no such variable
        with self.assertRaises(ValueError) as cm:
            dqm.set_linear(2, 1)

        # no such variable
        with self.assertRaises(ValueError) as cm:
            dqm.set_quadratic(u, 2, 1)

        dqm.set_linear(u, [0.5, 1])
        dqm.set_linear_case(v, 0, 1.5)
        dqm.set_quadratic(u, v, {(1, 1): -0.5})
        dqm.set_quadratic_case(u, 0, v, 1, -1)

        # illegal variable self interaction
        with self.assertRaises(ValueError) as cm:
            dqm.set_quadratic(u, u, {(0, 1): -0.5})

        # illegal variable self interaction
        with self.assertRaises(ValueError) as cm:
            dqm.set_quadratic_case(u, 0, u, 1, -1)

        self.assertEqual(list(dqm.get_linear(u)), [0.5, 1])
        self.assertEqual(dqm.get_linear_case(v, 0), 1.5)
        self.assertEqual(dqm.get_quadratic(u, v), {(1, 1): -0.5, (0, 1): -1})
        self.assertEqual(dqm.get_quadratic_case(u, 0, v, 1), -1)
        self.assertEqual(dqm.num_variables(), 2)
        self.assertEqual(dqm.num_variable_interactions(), 1)
        self.assertEqual(dqm.num_cases(), 5)
        self.assertEqual(dqm.num_case_interactions(), 2)

    def test_unique_case_labels(self):
        dqm = dimod.CaseLabelDQM()
        dqm.add_variable({'x1', 'x2', 'x3'})
        dqm.add_variable(['y1', 'y2', 'y3'], label='y1')

        # non-unique variable label
        with self.assertRaises(ValueError) as cm:
            dqm.add_variable({'z1', 'z2'}, label='x1')

        # non-unique case label
        with self.assertRaises(ValueError) as cm:
            dqm.add_variable({'x1', 'x4', 'x5'})

        # invalid case labels
        with self.assertRaises(TypeError) as cm:
            dqm.add_variable({{}, []})

        dqm.set_linear('x1', 0.5)
        dqm.set_linear('y1', 1.5)
        dqm.set_quadratic('x2', 'y3', -0.5)
        dqm.set_quadratic('x3', 'y2', -1.5)

        # illegal variable self interaction
        with self.assertRaises(ValueError) as cm:
            dqm.set_quadratic('x1', 'x1', -0.5)

        # no such case
        with self.assertRaises(ValueError) as cm:
            dqm.set_linear('x4', 1)

        # no such case
        with self.assertRaises(ValueError) as cm:
            dqm.set_quadratic('x1', 'x4', 1)

        self.assertEqual(dqm.get_linear('x1'), 0.5)
        self.assertEqual(dqm.get_quadratic('x2', 'y3'), -0.5)
        self.assertEqual(dqm.num_variables(), 2)
        self.assertEqual(dqm.num_variable_interactions(), 1)
        self.assertEqual(dqm.num_cases(), 6)
        self.assertEqual(dqm.num_case_interactions(), 2)

    def test_shared_case_labels(self):
        dqm = dimod.CaseLabelDQM()
        u = dqm.add_variable({'red', 'green', 'blue'}, shared_labels=True)
        v = dqm.add_variable(['blue', 'yellow', 'brown'], label='v', shared_labels=True)
        self.assertEqual(v, 'v')

        # non-unique variable label
        with self.assertRaises(ValueError) as cm:
            dqm.add_variable({'magenta', 'cyan'}, label=u, shared_labels=True)

        # non-unique case label
        with self.assertRaises(ValueError) as cm:
            dqm.add_variable([1, 2, 1], shared_labels=True)

        # invalid case labels
        with self.assertRaises(TypeError) as cm:
            dqm.add_variable({{}, []}, shared_labels=True)

        dqm.set_linear_case(u, 'red', 1)
        dqm.set_linear_case(v, 'yellow', 2)
        dqm.set_quadratic_case(u, 'green', v, 'blue', -0.5)
        dqm.set_quadratic_case(u, 'blue', v, 'brown', -0.5)

        # illegal variable self interaction
        with self.assertRaises(ValueError) as cm:
            dqm.set_quadratic_case(u, 'red', u, 'green', -1)

        # no such case
        with self.assertRaises(ValueError) as cm:
            dqm.set_linear_case(u, 'orange', 1)

        # no such case
        with self.assertRaises(ValueError) as cm:
            dqm.set_quadratic_case(u, 'green', v, 'purple', 1)

        self.assertEqual(dqm.get_linear_case(u, 'red'), 1)
        self.assertEqual(dqm.get_quadratic_case(u, 'green', v, 'blue'), -0.5)
        self.assertEqual(dqm.num_variables(), 2)
        self.assertEqual(dqm.num_variable_interactions(), 1)
        self.assertEqual(dqm.num_cases(), 6)
        self.assertEqual(dqm.num_case_interactions(), 2)

    def test_get_cases(self):
        dqm = dimod.CaseLabelDQM()
        x = dqm.add_variable(['red', 'green', 'blue'], shared_labels=True)
        y = dqm.add_variable(['x1', 'x2', 'x3'])
        z = dqm.add_variable(3)

        self.assertEqual(dqm.get_cases(x), ['red', 'green', 'blue'])
        self.assertEqual(dqm.get_cases(y), ['x1', 'x2', 'x3'])
        self.assertEqual(dqm.get_cases(z), [0, 1, 2])

    def test_non_string_unique_case_labels(self):
        dqm = dimod.CaseLabelDQM()
        x = dqm.add_variable([2, 1, 0])
        y = dqm.add_variable([3, 4, 5])
        z = dqm.add_variable([None, (1,), (2,)])

        cases = dqm.get_cases(x) + dqm.get_cases(y) + dqm.get_cases(z)
        self.assertEqual(len(cases), 9)

        for k, case in enumerate(cases):
            dqm.set_linear(case, 1 + k)

        for k, case in enumerate(cases):
            self.assertEqual(dqm.get_linear(case), 1 + k)

        dqm.set_quadratic(2, None, 1.1)
        dqm.set_quadratic((1,), 3, 2.2)

        self.assertEqual(dqm.num_variable_interactions(), 2)
        self.assertEqual(dqm.num_case_interactions(), 2)

        self.assertEqual(dqm.get_quadratic(None, 2), 1.1)
        self.assertEqual(dqm.get_quadratic(3, (1,)), 2.2)
        self.assertEqual(dqm.get_quadratic(2, 4), 0)

    def test_non_string_shared_case_labels(self):
        dqm = dimod.CaseLabelDQM()
        x = dqm.add_variable([2, 1, 0], shared_labels=True)
        y = dqm.add_variable([2, 1, 3], shared_labels=True)
        z = dqm.add_variable([None, (1,), (2,)], shared_labels=True)

        self.assertEqual(dqm.num_variables(), 3)
        self.assertEqual(dqm.num_cases(), 9)

        for k, var in enumerate((x, y, z)):
            for m, case in enumerate(dqm.get_cases(var)):
                dqm.set_linear_case(var, case, 1 + k * 3 + m)

        for k, var in enumerate((x, y, z)):
            for m, case in enumerate(dqm.get_cases(var)):
                self.assertEqual(dqm.get_linear_case(var, case), 1 + k * 3 + m)

        dqm.set_quadratic_case(x, 2, z, None, 1.1)
        dqm.set_quadratic_case(z, (1,), y, 3, 2.2)

        self.assertEqual(dqm.num_variable_interactions(), 2)
        self.assertEqual(dqm.num_case_interactions(), 2)

        self.assertEqual(dqm.get_quadratic_case(z, None, x, 2), 1.1)
        self.assertEqual(dqm.get_quadratic_case(y, 3, z, (1,)), 2.2)
        self.assertEqual(dqm.get_quadratic_case(x, 2, y, 2), 0)

    def test_mixed_case_label_type_interactions(self):
        dqm = dimod.CaseLabelDQM()
        x = dqm.add_variable(['red', 'green', 'blue'], shared_labels=True)
        y = dqm.add_variable(['x1', 'x2', 'x3'])
        z = dqm.add_variable(3)

        dqm.set_quadratic_case(x, 'red', y, 1, 10)
        dqm.set_quadratic_case(x, 'green', z, 1, -10)
        dqm.set_quadratic_case(y, 0, z, 1, -20)

        self.assertEqual(dqm.num_variable_interactions(), 3)
        self.assertEqual(dqm.num_case_interactions(), 3)

        self.assertEqual(dqm.get_quadratic_case(y, 1, x, 'red'), 10)
        self.assertEqual(dqm.get_quadratic_case(z, 1, x, 'green'), -10)
        self.assertEqual(dqm.get_quadratic_case(z, 1, y, 0), -20)

    def test_map_sample(self):
        dqm = dimod.CaseLabelDQM()
        x = dqm.add_variable(['red', 'green', 'blue'], shared_labels=True)
        y = dqm.add_variable(['x1', 'x2', 'x3'])
        z = dqm.add_variable(3)

        self.assertEqual(dqm.map_sample({x: 0, y: 1, z: 2}),
                         {x: 'red', 'x1': 0, 'x2': 1, 'x3': 0, z: 2})
