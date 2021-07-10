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
#

import unittest

import dimod

from dimod import ScaleComposite

try:
    import dwave.preprocessing as preprocessing
except ImportError:
    preprocessing = False


if preprocessing:
    @dimod.testing.load_sampler_bqm_tests(ScaleComposite(dimod.ExactSolver()))
    @dimod.testing.load_sampler_bqm_tests(ScaleComposite(dimod.NullSampler()))
    class TestScaleComposite(unittest.TestCase):
        def test_api(self):
            with self.assertWarns(DeprecationWarning):
                sampler = ScaleComposite(dimod.ExactSolver())
            dimod.testing.assert_sampler_api(sampler)

        def test_bias_range(self):
            bqm = dimod.BQM.from_ising({'a': -4.0, 'b': -4.0},
                                       {('a', 'b'): 3.2}, 1.5)

            with self.assertWarns(DeprecationWarning):
                sampler = ScaleComposite(dimod.TrackingComposite(dimod.ExactSolver()))

            sampleset = sampler.sample(bqm, bias_range=[-2, 2])

            # check that everything was restored properly
            dimod.testing.assert_sampleset_energies(sampleset, bqm)

            self.assertEqual(sampler.child.input['bqm'],
                             dimod.BQM.from_ising({'a': -2.0, 'b': -2.0},
                                                  {('a', 'b'): 1.6}, .75))

        def test_bias_ranges(self):
            bqm = dimod.BQM.from_ising({'a': -4.0, 'b': -4.0},
                                       {('a', 'b'): 4}, 1.5)

            with self.assertWarns(DeprecationWarning):
                sampler = ScaleComposite(dimod.TrackingComposite(dimod.ExactSolver()))

            sampleset = sampler.sample(bqm, bias_range=[-3, 3],
                                       quadratic_range=[-2, 2])

            # check that everything was restored properly
            dimod.testing.assert_sampleset_energies(sampleset, bqm)

            self.assertEqual(sampler.child.input['bqm'],
                             dimod.BQM.from_ising({'a': -2.0, 'b': -2.0},
                                                  {('a', 'b'): 2}, .75))

        def test_ignored_interactions(self):
            bqm = dimod.BQM.from_ising({'a': -4.0, 'b': -4.0},
                                       {('a', 'b'): 3.2, ('b', 'c'): 1}, 1.5)

            with self.assertWarns(DeprecationWarning):
                sampler = ScaleComposite(dimod.TrackingComposite(dimod.ExactSolver()))

            sampleset = sampler.sample(bqm, scalar=.5,
                                       ignored_interactions=[('b', 'c')])

            # check that everything was restored properly
            dimod.testing.assert_sampleset_energies(sampleset, bqm)

            self.assertEqual(sampler.child.input['bqm'],
                             dimod.BQM.from_ising({'a': -2.0, 'b': -2.0},
                                                  {'ab': 1.6, 'bc': 1}, .75))

        def test_ignored_offset(self):
            bqm = dimod.BQM.from_ising({'a': -4.0, 'b': -4.0},
                                       {('a', 'b'): 3.2}, 1.5)

            with self.assertWarns(DeprecationWarning):
                sampler = ScaleComposite(dimod.TrackingComposite(dimod.ExactSolver()))

            sampleset = sampler.sample(bqm, scalar=.5, ignore_offset=True)

            # check that everything was restored properly
            dimod.testing.assert_sampleset_energies(sampleset, bqm)

            self.assertEqual(sampler.child.input['bqm'],
                             dimod.BQM.from_ising({'a': -2.0, 'b': -2.0},
                                                  {('a', 'b'): 1.6}, 1.5))

        def test_ignored_variables(self):
            bqm = dimod.BQM.from_ising({'a': -4.0, 'b': -4.0},
                                       {('a', 'b'): 3.2}, 1.5)

            with self.assertWarns(DeprecationWarning):
                sampler = ScaleComposite(dimod.TrackingComposite(dimod.ExactSolver()))

            sampleset = sampler.sample(bqm, scalar=.5, ignored_variables='a')

            # check that everything was restored properly
            dimod.testing.assert_sampleset_energies(sampleset, bqm)

            self.assertEqual(sampler.child.input['bqm'],
                             dimod.BQM.from_ising({'a': -4.0, 'b': -2.0},
                                                  {('a', 'b'): 1.6}, .75))

        def test_scalar(self):
            bqm = dimod.BQM.from_ising({'a': -4.0, 'b': -4.0},
                                       {('a', 'b'): 3.2}, 1.5)

            with self.assertWarns(DeprecationWarning):
                sampler = ScaleComposite(dimod.TrackingComposite(dimod.ExactSolver()))

            sampleset = sampler.sample(bqm, scalar=.5)

            # check that everything was restored properly
            dimod.testing.assert_sampleset_energies(sampleset, bqm)

            self.assertEqual(sampler.child.input['bqm'],
                             dimod.BQM.from_ising({'a': -2.0, 'b': -2.0},
                                                  {('a', 'b'): 1.6}, .75))
