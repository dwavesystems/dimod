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

import dimod.testing as dtest
from dimod.vartypes import Vartype

import dimod

from dimod import BinaryQuadraticModel
from dimod import FixedVariableComposite, ExactSolver, RoofDualityComposite
from dimod import SampleSet

try:
    import dwave.preprocessing as preprocessing
except ImportError:
    preprocessing = False


if preprocessing:
    @dimod.testing.load_sampler_bqm_tests(FixedVariableComposite(ExactSolver()))
    @dimod.testing.load_sampler_bqm_tests(FixedVariableComposite(dimod.NullSampler()))
    class TestFixedVariableComposite(unittest.TestCase):

        def test_instantiation_smoketest(self):
            with self.assertWarns(DeprecationWarning):
                sampler = FixedVariableComposite(ExactSolver())

            dtest.assert_sampler_api(sampler)

        def test_sample(self):
            bqm = BinaryQuadraticModel({1: -1.3, 4: -0.5},
                                       {(1, 4): -0.6},
                                       0,
                                       vartype=Vartype.SPIN)

            fixed_variables = {1: -1}
            with self.assertWarns(DeprecationWarning):
                sampler = FixedVariableComposite(ExactSolver())
            response = sampler.sample(bqm, fixed_variables=fixed_variables)

            self.assertEqual(response.first.sample, {4: -1, 1: -1})
            self.assertAlmostEqual(response.first.energy, 1.2)

        def test_empty_bqm(self):
            bqm = BinaryQuadraticModel({1: -1.3, 4: -0.5},
                                       {(1, 4): -0.6},
                                       0,
                                       vartype=Vartype.SPIN)

            fixed_variables = {1: -1, 4: -1}
            with self.assertWarns(DeprecationWarning):
                sampler = FixedVariableComposite(ExactSolver())
            response = sampler.sample(bqm, fixed_variables=fixed_variables)
            self.assertIsInstance(response, SampleSet)

        def test_empty_fix(self):
            linear = {1: -1.3, 4: -0.5}
            quadratic = {(1, 4): -0.6}

            with self.assertWarns(DeprecationWarning):
                sampler = FixedVariableComposite(ExactSolver())
            with self.assertWarns(UserWarning):
                response = sampler.sample_ising(linear, quadratic)
            self.assertIsInstance(response, SampleSet)

            self.assertEqual(response.first.sample, {4: 1, 1: 1})
            self.assertAlmostEqual(response.first.energy, -2.4)


# @dimod.testing.load_sampler_bqm_tests(RoofDualityComposite(ExactSolver()))
# @dimod.testing.load_sampler_bqm_tests(RoofDualityComposite(dimod.NullSampler()))
@unittest.skipUnless(preprocessing, 'Need dwave-preprocessing')
class TestRoofDualityComposite(unittest.TestCase):
    def test_construction(self):
        with self.assertWarns(DeprecationWarning):
            sampler = RoofDualityComposite(dimod.ExactSolver())
        dtest.assert_sampler_api(sampler)

    def test_3path(self):
        with self.assertWarns(DeprecationWarning):
            sampler = RoofDualityComposite(dimod.ExactSolver())
        sampleset = sampler.sample_ising({'a': 10},  {'ab': -1, 'bc': 1})

        # all should be fixed, so should just see one
        self.assertEqual(len(sampleset), 1)
        self.assertEqual(set(sampleset.variables), set('abc'))

    def test_triangle(self):
        with self.assertWarns(DeprecationWarning):
            sampler = RoofDualityComposite(dimod.ExactSolver())

        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1, 'bc': -1, 'ac': -1})

        # two equally good solutions
        sampleset = sampler.sample(bqm)

        self.assertEqual(set(sampleset.variables), set('abc'))
        dimod.testing.assert_response_energies(sampleset, bqm)

    def test_triangle_sampling_mode_off(self):
        with self.assertWarns(DeprecationWarning):
            sampler = RoofDualityComposite(dimod.ExactSolver())

        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1, 'bc': -1, 'ac': -1})

        # two equally good solutions, but with sampling mode off it will pick one
        sampleset = sampler.sample(bqm, sampling_mode=False)

        self.assertEqual(set(sampleset.variables), set('abc'))
        self.assertEqual(len(sampleset), 1)  # all should be fixed
        dimod.testing.assert_response_energies(sampleset, bqm)
