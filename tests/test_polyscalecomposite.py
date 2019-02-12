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
# =============================================================================
import unittest

import dimod

from dimod import PolyScaleComposite, HigherOrderComposite, ExactSolver


class TestConstruction(unittest.TestCase):
    def test_typical(self):
        sampler = PolyScaleComposite(HigherOrderComposite(ExactSolver()))

        self.assertTrue(hasattr(sampler, 'sample_poly'))
        self.assertTrue(hasattr(sampler, 'sample_hising'))
        self.assertTrue(hasattr(sampler, 'sample_hubo'))

    def test_wrap_bqm(self):
        with self.assertRaises(TypeError):
            PolyScaleComposite(ExactSolver())

# todo: check all-zero with normalize


class TestSampleHising(unittest.TestCase):
    pass


class TestSampleHubo(unittest.TestCase):
    pass


class TestSamplePoly(unittest.TestCase):
    pass
