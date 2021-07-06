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

import unittest

import dimod

from dimod import ClipComposite
from dimod import BinaryQuadraticModel
from dimod import ExactSolver, NullSampler


@dimod.testing.load_sampler_bqm_tests(ClipComposite(dimod.ExactSolver()))
@dimod.testing.load_sampler_bqm_tests(ClipComposite(dimod.NullSampler()))
class TestClipCompositeClass(unittest.TestCase):
    def test_instantiation_smoketest(self):
        sampler = ClipComposite(NullSampler())
        dimod.testing.assert_sampler_api(sampler)

    def test_no_bounds(self):
        bqm = BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                                   {(0, 1): -2.0, (0, 2): -5.0, (0, 3): -2.0,
                                    (0, 4): -2.0, (1, 2): -2.0, (1, 3): -2.0, (1, 4): 4.0,
                                    (2, 3): -3.0, (2, 4): -5.0, (3, 4): -4.0}, 0, dimod.SPIN)
        sampler = ClipComposite(ExactSolver())
        solver = ExactSolver()
        response = sampler.sample(bqm)
        response_exact = solver.sample(bqm)
        self.assertEqual(response.first.sample, response_exact.first.sample)
        self.assertAlmostEqual(response.first.energy, response_exact.first.energy)

    def test_lb_only(self):
        bqm = BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                                   {(0, 1): -2.0, (0, 2): -5.0, (0, 3): -2.0,
                                    (0, 4): -2.0, (1, 2): -2.0, (1, 3): -2.0, (1, 4): 4.0,
                                    (2, 3): -3.0, (2, 4): -5.0, (3, 4): -4.0}, 0, dimod.SPIN)
        sampler = ClipComposite(ExactSolver())
        solver = ExactSolver()
        response = sampler.sample(bqm, lower_bound=-1)
        response_exact = solver.sample(bqm)
        self.assertEqual(response.first.sample, response_exact.first.sample)
        self.assertAlmostEqual(response.first.energy, response_exact.first.energy)

    def test_ub_only(self):
        bqm = BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                                   {(0, 1): -2.0, (0, 2): -5.0, (0, 3): -2.0,
                                    (0, 4): -2.0, (1, 2): -2.0, (1, 3): -2.0, (1, 4): 4.0,
                                    (2, 3): -3.0, (2, 4): -5.0, (3, 4): -4.0}, 0, dimod.SPIN)
        sampler = ClipComposite(ExactSolver())
        solver = ExactSolver()
        response = sampler.sample(bqm, upper_bound=1)
        response_exact = solver.sample(bqm)
        self.assertEqual(response.first.sample, response_exact.first.sample)
        self.assertAlmostEqual(response.first.energy, response_exact.first.energy)

    def test_lb_and_ub(self):
        bqm = BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
                                   {(0, 1): -2.0, (0, 2): -5.0, (0, 3): -2.0,
                                    (0, 4): -2.0, (1, 2): -2.0, (1, 3): -2.0, (1, 4): 4.0,
                                    (2, 3): -3.0, (2, 4): -5.0, (3, 4): -4.0}, 0, dimod.SPIN)
        sampler = ClipComposite(ExactSolver())
        solver = ExactSolver()
        response = sampler.sample(bqm, lower_bound=-1, upper_bound=1)
        response_exact = solver.sample(bqm)
        self.assertEqual(response.first.sample, response_exact.first.sample)
        self.assertAlmostEqual(response.first.energy, response_exact.first.energy)

    def test_with_labels(self):
        bqm = BinaryQuadraticModel({'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0, 'e': 0.0},
                                   {('a', 'b'): -2.0, ('a', 'c'): -5.0, ('a', 'd'): -2.0,
                                    ('a', 'e'): -2.0, ('b', 'c'): -2.0, ('b', 'd'): -2.0, ('b', 'e'): 4.0,
                                    ('c', 'd'): -3.0, ('c', 'e'): -5.0, ('d', 'e'): -4.0}, 0, dimod.SPIN)
        sampler = ClipComposite(ExactSolver())
        solver = ExactSolver()
        response = sampler.sample(bqm, lower_bound=-1, upper_bound=1)
        response_exact = solver.sample(bqm)
        self.assertEqual(response.first.sample, response_exact.first.sample)
        self.assertAlmostEqual(response.first.energy, response_exact.first.energy)

    def test_empty_bqm(self):
        bqm = BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
        sampler = ClipComposite(ExactSolver())
        sampler.sample(bqm, lower_bound=-1, upper_bound=1)

    def test_info_propagation(self):
        bqm = BinaryQuadraticModel.from_ising({}, {})

        class MySampler:
            @staticmethod
            def sample(bqm):
                return dimod.SampleSet.from_samples_bqm([], bqm, info=dict(a=1))

        sampleset = ClipComposite(MySampler).sample(bqm)
        self.assertEqual(sampleset.info, {'a': 1})
