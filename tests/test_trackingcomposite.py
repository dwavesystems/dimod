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


class TestConstruction(unittest.TestCase):
    def test_construction(self):
        sampler = dimod.TrackingComposite(dimod.ExactSolver())

        dimod.testing.assert_sampler_api(sampler)
        dimod.testing.assert_composite_api(sampler)

        self.assertEqual(sampler.inputs, [])
        self.assertEqual(sampler.outputs, [])


@dimod.testing.load_sampler_bqm_tests(dimod.TrackingComposite(dimod.ExactSolver()))
class TestSample(unittest.TestCase):
    def test_clear(self):
        sampler = dimod.TrackingComposite(dimod.ExactSolver())

        h0 = {'a': -1}
        J0 = {('a', 'b'): -1}
        ss0 = sampler.sample_ising(h0, J0)

        h1 = {'b': -1}
        J1 = {('b', 'c'): 2}
        ss1 = sampler.sample_ising(h1, J1)

        sampler.clear()

        self.assertEqual(sampler.inputs, [])
        self.assertEqual(sampler.outputs, [])

    def test_missing_inputs(self):
        sampler = dimod.TrackingComposite(dimod.ExactSolver())

        with self.assertRaises(ValueError):
            sampler.input

        with self.assertRaises(ValueError):
            sampler.output

    def test_sample(self):
        sampler = dimod.TrackingComposite(dimod.ExactSolver())

        bqm = dimod.BinaryQuadraticModel.from_ising({'a': -1}, {})

        ss = sampler.sample(bqm)

        self.assertEqual(sampler.input, dict(bqm=bqm))
        self.assertEqual(sampler.output, ss)

    def test_sample_ising(self):
        sampler = dimod.TrackingComposite(dimod.ExactSolver())

        h0 = {'a': -1}
        J0 = {('a', 'b'): -1}
        ss0 = sampler.sample_ising(h0, J0)

        h1 = {'b': -1}
        J1 = {('b', 'c'): 2}
        ss1 = sampler.sample_ising(h1, J1)

        self.assertEqual(sampler.input, dict(h=h1, J=J1))
        self.assertEqual(sampler.output, ss1)

        self.assertEqual(sampler.inputs, [dict(h=h0, J=J0), dict(h=h1, J=J1)])
        self.assertEqual(sampler.outputs, [ss0, ss1])

    def test_sample_ising_copy_true(self):
        sampler = dimod.TrackingComposite(dimod.ExactSolver(), copy=True)

        h0 = {'a': -1}
        J0 = {('a', 'b'): -1}
        ss0 = sampler.sample_ising(h0, J0)

        self.assertIsNot(sampler.input['h'], h0)
        self.assertIsNot(sampler.output, ss0)

    def test_sample_ising_copy_false(self):
        sampler = dimod.TrackingComposite(dimod.ExactSolver(), copy=False)

        h0 = {'a': -1}
        J0 = {('a', 'b'): -1}
        ss0 = sampler.sample_ising(h0, J0)

        self.assertIs(sampler.input['h'], h0)
        self.assertIs(sampler.output, ss0)

    def test_sample_ising_kwargs(self):
        sampler = dimod.TrackingComposite(dimod.RandomSampler())

        h = {'a': -1}
        J = {('a', 'b'): -1}
        ss = sampler.sample_ising(h, J, num_reads=5)

        self.assertEqual(sampler.input, dict(h=h, J=J, num_reads=5))
        self.assertEqual(list(sampler.input), ["h", "J", "num_reads"])
        self.assertEqual(sampler.output, ss)

    def test_sample_qubo(self):
        sampler = dimod.TrackingComposite(dimod.ExactSolver())

        Q = {('a', 'b'): -1}

        ss = sampler.sample_qubo(Q)
        self.assertEqual(sampler.input, dict(Q=Q))
        self.assertEqual(sampler.output, ss)
