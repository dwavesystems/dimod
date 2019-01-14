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

import dimod.testing as dtest
from dimod import ScaleComposite, ExactSolver, \
    SimulatedAnnealingSampler, HigherOrderComposite, BinaryQuadraticModel


class TestScaleComposite(unittest.TestCase):

    def test_instantiation_smoketest(self):
        sampler = ScaleComposite(ExactSolver())

        dtest.assert_sampler_api(sampler)

    def test_sample_ising(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5

        sampler = ScaleComposite(SimulatedAnnealingSampler())
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        scalar=0.5,
                                        num_reads=100)
        samples = [(s, e) for s, e, in
                   response.aggregate().data(['sample', 'energy'],
                                             sorted_by='num_occurrences',
                                             reverse=True)]
        self.assertEqual({'a': 1, 'b': 1}, samples[0][0])
        self.assertAlmostEqual(samples[0][1], 0.2)

    def test_sample_ising_ignore_variables(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5

        sampler = ScaleComposite(SimulatedAnnealingSampler())
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        scalar=0.5,
                                        ignored_variables=['a'],
                                        ignored_interactions=[('a', 'b')],
                                        num_reads=100)
        samples = [(s, e) for s, e, in
                   response.aggregate().data(['sample', 'energy'],
                                             sorted_by='num_occurrences',
                                             reverse=True)]
        self.assertEqual({'a': 1, 'b': -1}, samples[0][0])
        self.assertAlmostEqual(samples[0][1], 1.8)

    def test_sample_ising_ignore_interaction(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5

        sampler = ScaleComposite(SimulatedAnnealingSampler())
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        scalar=0.5,
                                        ignored_interactions=[('a', 'b')],
                                        num_reads=100)
        samples = [(s, e) for s, e, in
                   response.aggregate().data(['sample', 'energy'],
                                             sorted_by='num_occurrences',
                                             reverse=True)]
        self.assertIn(samples[0][0], [{'a': 1, 'b': -1}, {'a': -1, 'b': 1}])
        self.assertAlmostEqual(samples[0][1], 1.8)

    def test_sample_ising_ignore_offset(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5

        sampler = ScaleComposite(SimulatedAnnealingSampler())
        response = sampler.sample_ising(linear, quadratic, offset=offset,
                                        scalar=0.5,
                                        ignore_offset=True, num_reads=100)
        samples = [(s, e) for s, e, in
                   response.aggregate().data(['sample', 'energy'],
                                             sorted_by='num_occurrences',
                                             reverse=True)]
        self.assertEqual({'a': 1, 'b': 1}, samples[0][0])
        self.assertAlmostEqual(samples[0][1], 0.2)

    def test_sample_hising(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2}
        offset = 5
        sampler = ScaleComposite(HigherOrderComposite(
            SimulatedAnnealingSampler()))

        response = sampler.sample_ising(linear, quadratic,
                                        offset=offset, scalar=0.5,
                                        num_reads=100,
                                        penalty_strength=5.0,
                                        )
        samples = [(s, e) for s, e, in
                   response.aggregate().data(['sample', 'energy'],
                                             sorted_by='num_occurrences',
                                             reverse=True)]
        self.assertEqual({'a': 1, 'b': 1, 'c': 1}, samples[0][0])
        self.assertAlmostEqual(samples[0][1], -3.8)

    def test_sample_hising_ignore_variables(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2}
        offset = 5
        sampler = ScaleComposite(HigherOrderComposite(
            SimulatedAnnealingSampler()))
        response = sampler.sample_ising(linear, quadratic, scalar=0.5,
                                        penalty_strength=5.0,
                                        ignored_interactions=[
                                            ('a', 'b', 'c')],
                                        ignored_variables=['a'], num_reads \
                                            =100)

        samples = [(s, e) for s, e, in
                   response.aggregate().data(['sample', 'energy'],
                                             sorted_by='num_occurrences',
                                             reverse=True)]
        self.assertIn(samples[0][0], [{'a': 1, 'b': -1, 'c': 1},
                                      {'a': 1, 'b': 1, 'c': -1}])
        self.assertAlmostEqual(samples[0][1], -7.2)

    def test_sample_hising_ignore_interaction(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2}
        offset = 5
        sampler = ScaleComposite(HigherOrderComposite(
            SimulatedAnnealingSampler()))
        response = sampler.sample_ising(linear, quadratic, scalar=0.5,
                                        penalty_strength=5.0,
                                        ignored_interactions=[('a', 'b', 'c')],
                                        num_reads \
                                            =100)
        samples = [(s, e) for s, e, in
                   response.aggregate().data(['sample', 'energy'],
                                             sorted_by='num_occurrences',
                                             reverse=True)]
        self.assertIn(samples[0][0], [{'a': -1, 'b': 1, 'c': 1},
                                      {'a': 1, 'b': -1, 'c': 1},
                                      {'a': 1, 'b': 1, 'c': -1}])
        self.assertAlmostEqual(samples[0][1], -7.2)

    def test_sample_hising_ignore_offset(self):
        linear = {'a': -4.0, 'b': -4.0, 'c': -4.0}
        quadratic = {('a', 'b', 'c'): 3.2}
        offset = 5
        sampler = ScaleComposite(HigherOrderComposite(
            SimulatedAnnealingSampler()))

        response = sampler.sample_ising(linear, quadratic,
                                        offset=offset, scalar=0.5,
                                        num_reads=100,
                                        penalty_strength=5.0,
                                        ignore_offset=True,
                                        )
        samples = [(s, e) for s, e, in
                   response.aggregate().data(['sample', 'energy'],
                                             sorted_by='num_occurrences',
                                             reverse=True)]
        self.assertEqual({'a': 1, 'b': 1, 'c': 1}, samples[0][0])
        self.assertAlmostEqual(samples[0][1], -3.8)

    def test_sample(self):
        linear = {'a': -4.0, 'b': -4.0}
        quadratic = {('a', 'b'): 3.2}
        offset = 5
        sampler = ScaleComposite(SimulatedAnnealingSampler())
        bqm = BinaryQuadraticModel.from_ising(linear, quadratic,
                                              offset=offset)
        response = sampler.sample(bqm, scalar=0.5,
                                  ignored_variables=['a'],
                                  ignored_interactions=[('a', 'b')],
                                  num_reads=100)
        samples = [(s, e) for s, e, in
                   response.aggregate().data(['sample', 'energy'],
                                             sorted_by='num_occurrences',
                                             reverse=True)]
        self.assertEqual({'a': 1, 'b': -1}, samples[0][0])
        self.assertAlmostEqual(samples[0][1], 1.8)
