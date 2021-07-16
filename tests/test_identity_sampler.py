# Copyright 2020 D-Wave Systems Inc.
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
from dimod.exceptions import SamplerUnknownArgWarning
import numpy as np


# This sampler is an extremely thin wrapper around the
# Initialized.parse_initial_states method, so we rely on that testing for
# the most part, just doing the standard sampler tests and a few
# integration tests
@dimod.testing.load_sampler_bqm_tests(dimod.IdentitySampler)
class TestIdentitySampler(unittest.TestCase):
    def test_passthrough(self):
        samples = [[-1, +1], [+1, -1]]

        sampler = dimod.IdentitySampler()
        sampleset = sampler.sample_ising({}, {(0, 1): 1},
                                         initial_states=samples)

        np.testing.assert_array_equal(sampleset.samples()[:, [0, 1]], samples)

    def test_passthrough(self):
        samples = [[-1, +1], [+1, -1]]

        sampler = dimod.IdentitySampler()
        sampleset = sampler.sample_ising({}, {(0, 1): 1},
                                         initial_states=samples,
                                         initial_states_generator='tile',
                                         num_reads=10)

        self.assertEqual(len(sampleset), 10)
        np.testing.assert_array_equal(sampleset.samples()[:2, [0, 1]], samples)

    def test_kwargs(self):
        sampler = dimod.IdentitySampler()
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)

        with self.assertWarns(SamplerUnknownArgWarning):
            sampleset = sampler.sample(bqm, a=1, b=4)
