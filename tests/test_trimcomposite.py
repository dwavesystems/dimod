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
# ================================================================================================

import unittest

import dimod.testing as dtest
from dimod import BinaryQuadraticModel, HigherOrderComposite
from dimod import TrimComposite, ExactSolver


class TestTrimCompositeFeatures(unittest.TestCase):

    def test_instantiation_smoketest(self):
        sampler = TrimComposite(ExactSolver())

        dtest.assert_sampler_api(sampler)

    def test_none(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': 0}
        quadratic = {('a', 'b'): 3.2}
        offset = 0.0
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=offset)
        sampler = TrimComposite(ExactSolver())
        response = sampler.sample(bqm)

        self.assertEqual(len(response), 8)
        self.assertEqual(response.first.num_occurrences, 1)
        self.assertAlmostEqual(response.first.energy, -4.8)

    def test_aggregate(self):
        linear = {'a': -4, 'b': -4}
        quadratic = {('a', 'b', 'c'): 3.2}
        sampler = TrimComposite(HigherOrderComposite(ExactSolver()))
        response = sampler.sample_ising(linear, quadratic, aggregate=True)

        self.assertEqual(len(response), 8)
        self.assertEqual(response.first.num_occurrences, 4)
        self.assertAlmostEqual(response.first.energy, -11.2)

    def test_trim(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': 0}
        quadratic = {('a', 'b'): 3.2}
        offset = 0.0
        n = 3
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic, offset=offset)
        sampler = TrimComposite(ExactSolver())
        response = sampler.sample(bqm, n=n)

        self.assertEqual(len(response), n)
        self.assertEqual(response.first.num_occurrences, 1)
        self.assertAlmostEqual(response.first.energy, -4.8)

    def test_both(self):
        linear = {'a': -4, 'b': -4}
        quadratic = {('a', 'b', 'c'): 3.2}
        n = 3
        sampler = TrimComposite(HigherOrderComposite(ExactSolver()))
        response = sampler.sample_ising(linear, quadratic, aggregate=True, n=n)

        self.assertEqual(len(response), 3)
        self.assertEqual(response.first.num_occurrences, 4)
        self.assertAlmostEqual(response.first.energy, -11.2)

    def test_negative(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': 0}
        quadratic = {('a', 'b'): 3.2}
        sampler = TrimComposite(ExactSolver())
        with self.assertRaises(ValueError):
            sampler.sample_ising(linear, quadratic, n=-3, aggregate=True)
