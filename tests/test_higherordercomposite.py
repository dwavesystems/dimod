# Copyright 2018 D-Wave Systems Inc.
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

from dimod import HigherOrderComposite, ExactSolver


class TestFixedVariableComposite(unittest.TestCase):

    def test_instantiation_smoketest(self):
        sampler = HigherOrderComposite(ExactSolver())

        dtest.assert_sampler_api(sampler)

    def test_sample(self):
        linear = {0: -0.5, 1: -0.3, 2: -0.8}
        quadratic = {(0, 1, 2): -1.7}

        sampler = HigherOrderComposite(ExactSolver())
        response = sampler.sample_ising(linear, quadratic, penalty_strength=10,
                                        keep_penalty_variables=False)

        self.assertEqual(response.first.sample, {0: 1, 1: 1, 2: 1})
        self.assertAlmostEqual(response.first.energy, -3.3)
        self.assertFalse(response.first.penalty_satisfaction)

    def test_discard(self):
        linear = {0: -0.5, 1: -0.3, 2: -0.8}
        quadratic = {(0, 1, 2): -1.7}

        sampler = HigherOrderComposite(ExactSolver())
        response = sampler.sample_ising(linear, quadratic, penalty_strength=10,
                                        discard_unsatisfied=True)

        self.assertEqual(response.first.sample, {0: 1, 1: 1, 2: 1})
        self.assertAlmostEqual(response.first.energy, -3.3)
        self.assertTrue(response.first.penalty_satisfaction)

    def test_penalty_variables(self):
        linear = {0: -0.5, 1: -0.3, 2: -0.8}
        quadratic = {(0, 1, 2): -1.7}

        sampler = HigherOrderComposite(ExactSolver())
        response = sampler.sample_ising(linear, quadratic, penalty_strength=10,
                                        keep_penalty_variables=True,
                                        discard_unsatisfied=True)

        self.assertEqual(len(response.first.sample),5)
        self.assertAlmostEqual(response.first.energy, -3.3)
        self.assertTrue(response.first.penalty_satisfaction)

    def test_already_qubo(self):
        linear = {0: -0.5, 1: -0.3}
        quadratic = {(0, 1): -1.7}

        sampler = HigherOrderComposite(ExactSolver())
        response = sampler.sample_ising(linear, quadratic,
                                        keep_penalty_variables=True,
                                        discard_unsatisfied=False)

        self.assertEqual(response.first.sample, {0: 1, 1: 1})
        self.assertAlmostEqual(response.first.energy, -2.5)
        self.assertTrue(response.first.penalty_satisfaction)

    def test_already_qubo_2(self):
        linear = {0: -0.5, 1: -0.3}
        quadratic = {(0, 1): -1.7}

        sampler = HigherOrderComposite(ExactSolver())
        response = sampler.sample_ising(linear, quadratic, penalty_strength=10,
                                        keep_penalty_variables=True,
                                        discard_unsatisfied=True)

        self.assertEqual(response.first.sample, {0: 1, 1: 1})
        self.assertAlmostEqual(response.first.energy, -2.5)
        self.assertTrue(response.first.penalty_satisfaction)
