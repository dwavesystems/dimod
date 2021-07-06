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

from dimod import BinaryPolynomial
from dimod import PolyFixedVariableComposite, ExactPolySolver
from dimod import SampleSet


class TestFixedVariableComposite(unittest.TestCase):

    def test_instantiation_smoketest(self):
        sampler = PolyFixedVariableComposite(ExactPolySolver())

        dtest.assert_composite_api(sampler)

    def test_sample(self):
        poly = BinaryPolynomial.from_hising(h={1: -1.3, 2: 1.2, 3: -1.2, 4: -0.5},
                                            J={(1, 2, 3, 4): -0.6, (1, 2, 4): -0.3, (1, 3, 4): -0.8},
                                            offset=0)

        exact_sampler = ExactPolySolver()
        response_exact = exact_sampler.sample_poly(poly)
        gs = response_exact.first.sample
        gse = response_exact.first.energy

        sampler = PolyFixedVariableComposite(ExactPolySolver())
        fixed_variables = {k: v for k, v in gs.items() if k % 2 == 0}
        response = sampler.sample_poly(poly, fixed_variables=fixed_variables)

        self.assertIsInstance(response, SampleSet)
        self.assertEqual(response.first.sample, gs)
        self.assertAlmostEqual(response.first.energy, gse)

    def test_sample_with_labels(self):
        poly = BinaryPolynomial.from_hising(h={'a': -1.3, 'b': 1.2, 'c': -1.2, 'd': -0.5},
                                            J={'abcd': -0.6, 'abd': -0.3, 'acd': -0.8},
                                            offset=0)

        exact_sampler = ExactPolySolver()
        response_exact = exact_sampler.sample_poly(poly)
        gs = response_exact.first.sample
        gse = response_exact.first.energy

        sampler = PolyFixedVariableComposite(ExactPolySolver())
        fixed_variables = {k: v for k, v in gs.items() if k in {'a', 'c'}}
        response = sampler.sample_poly(poly, fixed_variables=fixed_variables)

        self.assertIsInstance(response, SampleSet)
        self.assertEqual(response.first.sample, gs)
        self.assertAlmostEqual(response.first.energy, gse)

    def test_fix_all(self):
        poly = BinaryPolynomial.from_hising(h={1: -1.3, 2: 1.2, 3: -1.2, 4: -0.5},
                                            J={(1, 2, 3, 4): -0.6, (1, 2, 4): -0.3, (1, 3, 4): -0.8},
                                            offset=0)

        exact_sampler = ExactPolySolver()
        response_exact = exact_sampler.sample_poly(poly)
        gs = response_exact.first.sample
        gse = response_exact.first.energy

        sampler = PolyFixedVariableComposite(ExactPolySolver())
        fixed_variables = {k: v for k, v in gs.items()}
        response = sampler.sample_poly(poly, fixed_variables=fixed_variables)

        self.assertIsInstance(response, SampleSet)
        self.assertEqual(response.first.sample, gs)
        self.assertAlmostEqual(response.first.energy, gse)

    def test_empty_poly(self):
        poly = BinaryPolynomial({}, 'SPIN')
        sampler = PolyFixedVariableComposite(ExactPolySolver())
        response = sampler.sample_poly(poly)
        self.assertIsInstance(response, SampleSet)

    def test_empty_fix(self):
        linear = {1: -1.3, 4: -0.5, 3: -2.0, 2: 1.0}
        high_order = {(1, 3, 4): -0.6, (1, 2, 4): +0.6}
        exact_sampler = ExactPolySolver()
        response_exact = exact_sampler.sample_hising(linear, high_order)

        gs = response_exact.first.sample
        gse = response_exact.first.energy
        sampler = PolyFixedVariableComposite(ExactPolySolver())
        response = sampler.sample_hising(linear, high_order)

        self.assertIsInstance(response, SampleSet)
        self.assertEqual(response.first.sample, gs)
        self.assertAlmostEqual(response.first.energy, gse)
