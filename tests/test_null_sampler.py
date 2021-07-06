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

import unittest

import dimod
from dimod.exceptions import SamplerUnknownArgWarning


class TestConstruction(unittest.TestCase):
    def test_construction(self):
        sampler = dimod.NullSampler()

        dimod.testing.assert_sampler_api(sampler)

    def test_parameters_iterable(self):
        sampler = dimod.NullSampler(parameters=['a'])
        self.assertEqual(sampler.parameters, {'a': []})

    def test_parameters_dict(self):
        sampler = dimod.NullSampler(parameters={'a': [1.5]})
        self.assertEqual(sampler.parameters, {'a': [1.5]})


@dimod.testing.load_sampler_bqm_tests(dimod.NullSampler)
class TestSample(unittest.TestCase):
    def test_empty_bqm(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        ss = dimod.NullSampler().sample(bqm)

        self.assertEqual(len(ss), 0)
        self.assertEqual(ss.record.sample.shape, (0, 0))

    def test_nonempty_bqm(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': -1}, {'ab': 1})
        ss = dimod.NullSampler().sample(bqm)

        self.assertEqual(len(ss), 0)
        self.assertEqual(ss.record.sample.shape, (0, 2))

    def test_kwargs(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': -1}, {'ab': 1})

        with self.assertWarns(SamplerUnknownArgWarning):
            dimod.NullSampler().sample(bqm, a=5)

        ss = dimod.NullSampler(parameters=['a']).sample(bqm, a=5)
