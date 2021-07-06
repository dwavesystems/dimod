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
import dimod.testing as dtest
from dimod.exceptions import SamplerUnknownArgWarning


@dimod.testing.load_sampler_bqm_tests(dimod.RandomSampler)
class TestRandomSampler(unittest.TestCase):
    def test_initialization(self):
        sampler = dimod.RandomSampler()

        dtest.assert_sampler_api(sampler)

        self.assertEqual(sampler.properties, {})
        self.assertEqual(sampler.parameters, {'num_reads': []})

    def test_energies(self):
        bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0},
                                         {(0, 1): -1.0, (1, 2): 1.0, (0, 2): 1.0},
                                         1.0,
                                         dimod.SPIN)
        sampler = dimod.RandomSampler()
        response = sampler.sample(bqm, num_reads=10)

        dtest.assert_response_energies(response, bqm)

    def test_kwargs(self):
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
        with self.assertWarns(SamplerUnknownArgWarning):
            dimod.RandomSampler().sample(bqm, a=5, b=2)
