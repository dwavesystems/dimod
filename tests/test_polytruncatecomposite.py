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

import dimod.testing as dtest
from dimod import BinaryQuadraticModel, HigherOrderComposite
from dimod import PolyTruncateComposite, ExactSolver


class TestConstruction(unittest.TestCase):
    def test_10(self):
        sampler = PolyTruncateComposite(HigherOrderComposite(ExactSolver()), 10)
        dtest.assert_composite_api(sampler)

        self.assertEqual(sampler.parameters, sampler.child.parameters)

    def test_0(self):
        with self.assertRaises(ValueError):
            PolyTruncateComposite(HigherOrderComposite(ExactSolver()), 0)


class TestSample(unittest.TestCase):
    def test_sampleset_shorter(self):
        h = {'a': -4.0, 'b': -4.0, 'c': 0}
        J = {('a', 'b'): 3.2}

        sampler = PolyTruncateComposite(HigherOrderComposite(ExactSolver()), 10)
        samples = sampler.sample_hising(h, J)

        # we should see 2**n < 10 rows
        self.assertEqual(len(samples), 8)

    def test_sampleset_trim(self):
        h = {'a': -4.0, 'b': -4.0, 'c': 0}
        J = {('a', 'b'): 3.2}

        sampler = PolyTruncateComposite(HigherOrderComposite(ExactSolver()), 6)
        samples = sampler.sample_hising(h, J)

        self.assertEqual(len(samples), 6)

    def test_with_aggration(self):
        # this is actually just a smoke test, needs better testing in the
        # future...
        h = {'a': -4.0, 'b': -4.0, 'c': 0}
        J = {('a', 'b'): 3.2}

        sampler = PolyTruncateComposite(HigherOrderComposite(ExactSolver()), 6, aggregate=True)
        samples = sampler.sample_hising(h, J)

        self.assertEqual(len(samples), 6)
