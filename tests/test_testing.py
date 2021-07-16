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

import unittest
import warnings

import dimod


class Test_assert_almost_equal_bqm(unittest.TestCase):
    def test_empty(self):
        bqm0 = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        bqm1 = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        dimod.testing.assert_bqm_almost_equal(bqm0, bqm1)

    def test_self_empty(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        dimod.testing.assert_bqm_almost_equal(bqm, bqm)

    def test_unlike_variables(self):
        bqm0 = dimod.BinaryQuadraticModel.from_ising({'a': -1}, {})
        bqm1 = dimod.BinaryQuadraticModel.from_ising({'b': -1}, {})
        with self.assertRaises(AssertionError):
            dimod.testing.assert_bqm_almost_equal(bqm0, bqm1)

    def test_unlike_offset(self):
        bqm0 = dimod.BinaryQuadraticModel.from_ising({'a': -1}, {}, 1.1)
        bqm1 = dimod.BinaryQuadraticModel.from_ising({'a': -1}, {}, 1.2)
        with self.assertRaises(AssertionError):
            dimod.testing.assert_bqm_almost_equal(bqm0, bqm1)
        dimod.testing.assert_bqm_almost_equal(bqm0, bqm1, places=0)

    def test_unlike_linear(self):
        bqm0 = dimod.BinaryQuadraticModel.from_ising({'a': -1}, {})
        bqm1 = dimod.BinaryQuadraticModel.from_ising({'a': -1.1}, {})
        with self.assertRaises(AssertionError):
            dimod.testing.assert_bqm_almost_equal(bqm0, bqm1)
        dimod.testing.assert_bqm_almost_equal(bqm0, bqm1, places=0)

    def test_unlike_interactions(self):
        bqm0 = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1})
        bqm1 = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1.1})
        with self.assertRaises(AssertionError):
            dimod.testing.assert_bqm_almost_equal(bqm0, bqm1)
        dimod.testing.assert_bqm_almost_equal(bqm0, bqm1, places=0)

    def test_ignore_zero_interactions(self):
        h = {'a': 0, 'b': 0, 'c': 0, 'd': 0}
        J0 = {'ab': 0, 'bc': -1}
        J1 = {'cb': -1, 'cd': 0}

        bqm0 = dimod.BinaryQuadraticModel.from_ising(h, J0)
        bqm1 = dimod.BinaryQuadraticModel.from_ising(h, J1)
        with self.assertRaises(AssertionError):
            dimod.testing.assert_bqm_almost_equal(bqm0, bqm1)
        dimod.testing.assert_bqm_almost_equal(bqm0, bqm1, ignore_zero_interactions=True)
        with self.assertRaises(AssertionError):
            dimod.testing.assert_bqm_almost_equal(bqm1, bqm0)
        dimod.testing.assert_bqm_almost_equal(bqm1, bqm0, ignore_zero_interactions=True)


class TestCreateBQMTests(unittest.TestCase):
    def test_overload_warning(self):

        def my_sampler():
            pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @dimod.testing.load_sampler_bqm_tests(my_sampler)
            class MyTestCase:
                def test_sample_qubo_empty_my_sampler(self):
                    pass

            n = sum(wa.category is dimod.testing.sampler.TestCaseOverloadWarning
                    for wa in w)

            self.assertEqual(n, 1)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            @dimod.testing.load_sampler_bqm_tests(my_sampler,
                                                  suppress_overload_warning=True)
            class MyTestCase:
                def test_sample_qubo_empty_my_sampler(self):
                    pass

            n = sum(wa.category is dimod.testing.sampler.TestCaseOverloadWarning
                    for wa in w)

            self.assertEqual(n, 0)
