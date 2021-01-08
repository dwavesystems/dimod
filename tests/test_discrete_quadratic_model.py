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


class TestConstruction(unittest.TestCase):
    def test_empty(self):
        dqm = dimod.DQM()

        self.assertEqual(dqm.num_variables(), 0)
        self.assertEqual(dqm.variables, [])
        self.assertEqual(dqm.adj, {})
        self.assertEqual(dqm.num_cases(), 0)

    def test_one_variable(self):
        dqm = dimod.DQM()

        v = dqm.add_variable(10)

        self.assertEqual(v, 0)
        self.assertEqual(dqm.num_variables(), 1)
        self.assertEqual(dqm.variables, [0])
        self.assertEqual(dqm.adj, {0: set()})
        self.assertEqual(dqm.num_cases(), 10)
        self.assertEqual(dqm.num_cases(0), 10)

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
        dqm.to_file()._file.read()

    def test_compress(self):
        dqm = gnp_random_dqm(5, [4, 5, 2, 1, 10], .5, .5, seed=23)

        self.assertLess(len(dqm.to_file(compress=True).read()),
                        len(dqm.to_file().read()))

    def test_compressed(self):
        # deprecated
        dqm = gnp_random_dqm(5, [4, 5, 2, 1, 10], .5, .5, seed=23)

        with self.assertWarns(DeprecationWarning):
            self.assertLess(len(dqm.to_file(compressed=True).read()),
                            len(dqm.to_file().read()))

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
        self.assertTrue(dimod.DQM().to_file().readable())

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
        self.assertTrue(dimod.DQM().to_file().seekable())

    def test_writeable(self):
        self.assertTrue(dimod.DQM().to_file().writable())


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
                    np.float, np.double]])
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
                    np.float, np.double]])
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
            x[i] = dqm.add_variable(num_cases, label=f'x_{i}')

        for c in range(num_cases):
            dqm.add_constraint_as_quadratic(
                [(x[i], c, 1.0) for i in range(num_variables)],
                lagrange_multiplier=1.0, constant=-1.0)

        for i in x:
            for case in range(num_cases):
                self.assertEqual(dqm.get_linear_case(x[i], case), -1)
            for j in x:
                if j > i:
                    for case in range(num_cases):
                        self.assertEqual(dqm.get_quadratic_case(x[i], case, x[j], case), 2.0)


class TestNumpyVectors(unittest.TestCase):

    def test_empty_functional(self):
        dqm = dimod.DQM()
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

    def test_random_shuffled_quadratic(self):

        dqm = gnp_random_dqm(5, [4, 5, 2, 1, 10], .5, .5, seed=17)

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
