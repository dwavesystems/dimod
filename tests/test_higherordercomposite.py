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

import numpy as np
import unittest

import dimod
import dimod.testing as dtest

from dimod import HigherOrderComposite, ExactSolver


class TestHigherOrderComposite(unittest.TestCase):

    def test_sample(self):
        linear = {0: -0.5, 1: -0.3, 2: -0.8}
        quadratic = {(0, 1, 2): -1.7}

        sampler = HigherOrderComposite(ExactSolver())
        response = sampler.sample_hising(linear, quadratic, penalty_strength=10,
                                         keep_penalty_variables=False,
                                         discard_unsatisfied=False)

        self.assertEqual(response.first.sample, {0: 1, 1: 1, 2: 1})
        self.assertAlmostEqual(response.first.energy, -3.3)
        self.assertFalse(np.prod(response.record.penalty_satisfaction))

    def test_discard(self):
        linear = {0: -0.5, 1: -0.3, 2: -0.8}
        quadratic = {(0, 1, 2): -1.7}

        sampler = HigherOrderComposite(ExactSolver())
        response = sampler.sample_hising(linear, quadratic, penalty_strength=10,
                                         discard_unsatisfied=True)

        self.assertEqual(response.first.sample, {0: 1, 1: 1, 2: 1})
        self.assertAlmostEqual(response.first.energy, -3.3)
        self.assertTrue(np.prod(response.record.penalty_satisfaction))

    def test_penalty_variables(self):
        linear = {0: -0.5, 1: -0.3, 2: -0.8}
        quadratic = {(0, 1, 2): -1.7}

        sampler = HigherOrderComposite(ExactSolver())
        response = sampler.sample_hising(linear, quadratic, penalty_strength=10,
                                         keep_penalty_variables=True,
                                         discard_unsatisfied=True)

        self.assertEqual(len(response.first.sample), 5)
        self.assertAlmostEqual(response.first.energy, -3.3)
        self.assertTrue(response.first.penalty_satisfaction)

    def test_already_qubo(self):
        linear = {0: -0.5, 1: -0.3}
        quadratic = {(0, 1): -1.7}

        sampler = HigherOrderComposite(ExactSolver())
        response = sampler.sample_hising(linear, quadratic,
                                         keep_penalty_variables=True,
                                         discard_unsatisfied=False)

        self.assertEqual(response.first.sample, {0: 1, 1: 1})
        self.assertAlmostEqual(response.first.energy, -2.5)
        self.assertTrue(response.first.penalty_satisfaction)

    def test_already_qubo_2(self):
        linear = {0: -0.5, 1: -0.3}
        quadratic = {(0, 1): -1.7}

        sampler = HigherOrderComposite(ExactSolver())
        response = sampler.sample_hising(linear, quadratic, penalty_strength=10,
                                         keep_penalty_variables=True,
                                         discard_unsatisfied=True)

        self.assertEqual(response.first.sample, {0: 1, 1: 1})
        self.assertAlmostEqual(response.first.energy, -2.5)
        self.assertTrue(response.first.penalty_satisfaction)


class TestInitialState(unittest.TestCase):
    def setUp(self):
        # get a sampler that accepts an initial_state
        base = dimod.NullSampler(parameters=['initial_state'])
        self.tracker = dimod.TrackingComposite(base)

    def test_quadratic(self):

        sampler = HigherOrderComposite(self.tracker)

        Q = {(0, 0): 1, (0, 1): 2}

        sampleset = sampler.sample_hubo(Q, initial_state={0: 1, 1: 1})

        # nothing should change
        self.assertEqual(self.tracker.input['initial_state'], {0: 1, 1: 1})

    def test_higherorder_binary(self):
        sampler = HigherOrderComposite(self.tracker)

        Q = {'abc': -1, 'ab': 1}

        sampleset = sampler.sample_hubo(Q, initial_state={'a': 1, 'b': 1, 'c': 0})

        initial_state = self.tracker.input['initial_state']

        self.assertIn(initial_state, [{'a': 1, 'b': 1, 'c': 0, 'a*b': 1},
                                      {'a': 1, 'b': 1, 'c': 0, 'a*c': 0},
                                      {'a': 1, 'b': 1, 'c': 0, 'b*c': 0},
                                      {'a': 1, 'b': 1, 'c': 0, 'b*a': 1},
                                      {'a': 1, 'b': 1, 'c': 0, 'c*a': 0},
                                      {'a': 1, 'b': 1, 'c': 0, 'c*b': 0},
                                      ])

    def test_higherorder_spin(self):
        sampler = HigherOrderComposite(self.tracker)

        J = {'abc': -1, 'ab': 1}

        sampleset = sampler.sample_hising({}, J,
                                          initial_state={'a': 1, 'b': 1, 'c': -1})

        bqm = self.tracker.input['bqm']
        initial_state = self.tracker.input['initial_state']

        samples = dimod.ExactSolver().sample(bqm).samples()

        # make sure that the initial-state is minimzed over the product/aux
        mask = (samples[:, ['a', 'b', 'c']] == [1, 1, -1]).all(axis=1)
        for v, val in initial_state.items():
            if v in ['a', 'b', 'c']:
                continue
            self.assertTrue(samples[mask, [v]][0, 0], val)
