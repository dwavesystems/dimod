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

import numpy as np

import dimod

from dimod.reference import FixedVariableComposite, ExactSolver
from dimod.binary_quadratic_model import BinaryQuadraticModel

import dimod.testing as dtest

class TestFixedVariableComposite(unittest.TestCase):

    def test_instantiation_smoketest(self):
        sampler = FixedVariableComposite(ExactSolver())

        dtest.assert_sampler_api(sampler)

    def test_sample(self):
        bqm = BinaryQuadraticModel(linear={1: -1.3, 4: -0.5},
                               quadratic={(1, 4): -0.6},
                               offset=0,
                               vartype=dimod.SPIN)

        fixed_variables = {1: -1}
        sampler = FixedVariableComposite(ExactSolver())
        response = sampler.sample(bqm, fixed_variables=fixed_variables)

        self.assertDictEqual(dict(response.first.sample), {4: -1, 1: -1})
        self.assertAlmostEquals(response.first.energy,1.2)

    def test_empty_bqm(self):
        bqm = BinaryQuadraticModel(linear={1: -1.3, 4: -0.5},
                                   quadratic={(1, 4): -0.6},
                                   offset=0,
                                   vartype=dimod.SPIN)

        fixed_variables = {1: -1, 4: -1}
        sampler = FixedVariableComposite(ExactSolver())
        response = sampler.sample(bqm, fixed_variables=fixed_variables)
        self.assertIsInstance(response, dimod.SampleSet)