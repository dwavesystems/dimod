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

import dimod.testing
from dimod.vartypes import Vartype

from dimod import BinaryQuadraticModel
from dimod import ConnectedComponentsComposite, ExactSolver, FixedVariableComposite
from dimod import SampleSet


@dimod.testing.load_sampler_bqm_tests(ConnectedComponentsComposite(dimod.ExactSolver()))
@dimod.testing.load_sampler_bqm_tests(ConnectedComponentsComposite(dimod.NullSampler()))
class TestConnectedComponentsComposite(unittest.TestCase):

    def test_instantiation_smoketest(self):
        sampler = ConnectedComponentsComposite(ExactSolver())

        dimod.testing.assert_sampler_api(sampler)

    def test_sample(self):
        bqm = BinaryQuadraticModel({1: -1.3, 4: -0.5},
                                   {(1, 4): -0.6},
                                   0,
                                   vartype=Vartype.SPIN)
        sampler = ConnectedComponentsComposite(ExactSolver())
        response = sampler.sample(bqm)

        self.assertEqual(response.first.sample, {4: 1, 1: 1})
        self.assertAlmostEqual(response.first.energy, -2.4)

    def test_empty_bqm(self):
        bqm = BinaryQuadraticModel({1: -1.3, 4: -0.5},
                                   {(1, 4): -0.6},
                                   0,
                                   vartype=Vartype.SPIN)

        fixed_variables = {1: -1, 4: -1}
        sampler = FixedVariableComposite(ConnectedComponentsComposite(ExactSolver()))
        response = sampler.sample(bqm, fixed_variables=fixed_variables)
        self.assertIsInstance(response, SampleSet)

    def test_sample_two_components(self):
        bqm = BinaryQuadraticModel({0: 0.0, 1: 4.0, 2: -4.0, 3: 0.0}, {(0, 1): -4.0, (2, 3): 4.0}, 0.0, Vartype.BINARY)

        sampler = ConnectedComponentsComposite(ExactSolver())
        response = sampler.sample(bqm)
        self.assertIsInstance(response, SampleSet)
        self.assertEqual(response.first.sample, {0: 0, 1: 0, 2: 1, 3: 0})
        self.assertAlmostEqual(response.first.energy, bqm.energy({0: 0, 1: 0, 2: 1, 3: 0}))

    def test_sample_three_components(self):
        bqm = BinaryQuadraticModel({0: 0.0, 1: 4.0, 2: -4.0, 3: 0.0, 4: 1.0, 5: -1.0},
                                   {(0, 1): -4.0, (2, 3): 4.0, (4, 5): -2.0}, 0.0, Vartype.BINARY)

        sampler = ConnectedComponentsComposite(ExactSolver())
        response = sampler.sample(bqm)
        self.assertIsInstance(response, SampleSet)
        self.assertEqual(response.first.sample, {0: 0, 1: 0, 2: 1, 3: 0, 4: 1, 5: 1})
        self.assertAlmostEqual(response.first.energy, bqm.energy({0: 0, 1: 0, 2: 1, 3: 0, 4: 1, 5: 1}))

    def test_sample_passcomponents(self):
        bqm = BinaryQuadraticModel({0: 0.0, 1: 4.0, 2: -4.0, 3: 0.0}, {(0, 1): -4.0, (2, 3): 4.0}, 0.0, Vartype.BINARY)

        sampler = ConnectedComponentsComposite(ExactSolver())
        response = sampler.sample(bqm, components=[{0, 1}, {2, 3}])
        self.assertIsInstance(response, SampleSet)
        self.assertEqual(response.first.sample, {0: 0, 1: 0, 2: 1, 3: 0})
        self.assertAlmostEqual(response.first.energy, bqm.energy({0: 0, 1: 0, 2: 1, 3: 0}))
