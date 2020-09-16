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
import unittest

import dimod
import numpy as np


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


class TestQuadratic(unittest.TestCase):
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
