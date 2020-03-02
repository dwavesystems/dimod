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
import itertools
import random

import dimod
import dimod.testing as dit


# @dimod.testing.load_sampler_bqm_tests(dimod.SpinReversalTransformComposite(dimod.ExactSolver()))
# @dimod.testing.load_sampler_bqm_tests(dimod.SpinReversalTransformComposite(dimod.RandomSampler()))
class TestSpinTransformComposite(unittest.TestCase):
    def test_instantiation(self):
        for factory in [dimod.ExactSolver, dimod.RandomSampler, dimod.SimulatedAnnealingSampler]:

            sampler = dimod.SpinReversalTransformComposite(factory())

            dit.assert_sampler_api(sampler)
            dit.assert_composite_api(sampler)

    def test_typical(self):
        sampler = dimod.SpinReversalTransformComposite(dimod.ExactSolver())
        Q = {('a', 'a'): -1, ('b', 'b'): -1, ('a', 'b'): 2}
        response = sampler.sample_qubo(Q, num_spin_reversal_transforms=100, spin_reversal_variables={'a'})

        dimod.testing.assert_response_energies(response, dimod.BinaryQuadraticModel.from_qubo(Q))
