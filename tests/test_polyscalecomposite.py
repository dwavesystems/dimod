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

import numpy as np

import dimod

from dimod import PolyScaleComposite, HigherOrderComposite, ExactSolver


class RangeLimitedSampler(dimod.PolySampler):
    parameters = None
    properties = None

    def sample_poly(self, poly, num_reads=1):

        if any(bias > 1 for bias in poly.values()):
            raise RuntimeError
        if any(bias < -1 for bias in poly.values()):
            raise RuntimeError

        samples = np.ones((num_reads, len(poly.variables))), list(poly.variables)
        sampleset = dimod.SampleSet.from_samples(samples, vartype=poly.vartype,
                                                 energy=poly.energies(samples))
        return sampleset


class TestConstruction(unittest.TestCase):
    def test_typical(self):
        sampler = PolyScaleComposite(HigherOrderComposite(ExactSolver()))

        self.assertTrue(hasattr(sampler, 'sample_poly'))
        self.assertTrue(hasattr(sampler, 'sample_hising'))
        self.assertTrue(hasattr(sampler, 'sample_hubo'))

    def test_wrap_bqm(self):
        with self.assertRaises(TypeError):
            PolyScaleComposite(ExactSolver())


class TestSampleHising(unittest.TestCase):
    def test_all_zero(self):
        sampler = PolyScaleComposite(RangeLimitedSampler())
        sampler.sample_hising({'a': 0}, {'ab': 0, 'bc': 0, 'abc': 0})

    def test_empty(self):
        sampler = PolyScaleComposite(RangeLimitedSampler())
        samples = sampler.sample_hising({}, {})
        self.assertEqual(len(samples.variables), 0)

    def test_normalizing(self):
        sampler = PolyScaleComposite(RangeLimitedSampler())
        samples = sampler.sample_hising({'a': 4}, {})
        self.assertEqual(samples.first.energy, 4)

    def test_scale(self):
        sampler = PolyScaleComposite(RangeLimitedSampler())
        samples = sampler.sample_hising({'a': 4}, {}, scalar=.25)
        self.assertEqual(samples.first.energy, 4)

    def test_fail_scale(self):
        sampler = PolyScaleComposite(RangeLimitedSampler())
        with self.assertRaises(RuntimeError):
            sampler.sample_hising({'a': 4}, {}, scalar=1)

    def test_all_energies(self):
        sampler = PolyScaleComposite(HigherOrderComposite(ExactSolver()))

        h = {'a': -1, 'b': 4}
        J = {'abc': -1, 'ab': 1, 'aaa': .5}

        sampleset = sampler.sample_hising(h, J, discard_unsatisfied=True)

        for sample, energy in sampleset.data(['sample', 'energy']):
            en = 0
            for v, bias in h.items():
                en += sample[v] * bias
            for term, bias in J.items():
                val = bias
                for v in term:
                    val *= sample[v]
                en += val

            self.assertAlmostEqual(energy, en)


class TestSampleHubo(unittest.TestCase):
    def test_empty(self):
        sampler = PolyScaleComposite(RangeLimitedSampler())
        samples = sampler.sample_hubo({})
        self.assertEqual(len(samples.variables), 0)

        sampler = PolyScaleComposite(RangeLimitedSampler())
        samples = sampler.sample_hubo({'a': 4}, scalar=.25)
        self.assertEqual(samples.first.energy, 4)

    def test_fail_scale(self):
        sampler = PolyScaleComposite(RangeLimitedSampler())
        with self.assertRaises(RuntimeError):
            sampler.sample_hubo({'a': 4}, scalar=1)
