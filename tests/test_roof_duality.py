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
import dimod
from dimod import ExtendedRoofDualityComposite

try:
    from dimod import fix_variables
except ImportError:
    cpp = False
else:
    cpp = True


@unittest.skipUnless(cpp, "no cpp extensions built")
class TestFixVariables(unittest.TestCase):
    def test_3path(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 10}, {'ab': -1, 'bc': 1})
        fixed = dimod.fix_variables(bqm)
        self.assertEqual(fixed, {'a': -1, 'b': -1, 'c': 1})


class TestExtendedRoofDualityComposite(unittest.TestCase):
    @unittest.skipIf(cpp, "cpp extensions built")
    def test_nocpp_error(self):
        with self.assertRaises(ImportError):
            ExtendedRoofDualityComposite(dimod.ExactSolver()).sample_ising({}, {})

    @unittest.skipUnless(cpp, "no cpp extensions built")
    def test_construction(self):
        sampler = ExtendedRoofDualityComposite(dimod.ExactSolver())
        dtest.assert_sampler_api(sampler)

    @unittest.skipUnless(cpp, "no cpp extensions built")
    def test_3path(self):
        sampler = ExtendedRoofDualityComposite(dimod.ExactSolver())
        sampleset = sampler.sample_ising({'a': 10},  {'ab': -1, 'bc': 1})

        # all should be fixed, so should just see one solution
        self.assertEqual(len(sampleset), 1)
        self.assertEqual(set(sampleset.variables), set('abc'))

    @unittest.skipUnless(cpp, "no cpp extensions built")
    def test_3path_contracted(self):
        sampler = ExtendedRoofDualityComposite(dimod.ExactSolver())
        sampleset = sampler.sample_ising({},  {'ab': -1, 'bc': 1})

        # all should be contracted, so should see two solutions
        self.assertEqual(len(sampleset), 2)
        self.assertEqual(set(sampleset.variables), set('abc'))

    @unittest.skipUnless(cpp, "no cpp extensions built")
    def test_triangle(self):
        sampler = ExtendedRoofDualityComposite(dimod.ExactSolver())

        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1, 'bc': -1, 'ac': -1})

        # two equally good solutions
        sampleset = sampler.sample(bqm)

        self.assertEqual(set(sampleset.variables), set('abc'))
        dimod.testing.assert_response_energies(sampleset, bqm)

    @unittest.skipUnless(cpp, "no cpp extensions built")
    def test_triangle_sampling_mode_off(self):
        sampler = ExtendedRoofDualityComposite(dimod.ExactSolver())

        bqm = dimod.BinaryQuadraticModel.from_ising({}, {'ab': -1, 'bc': -1, 'ac': -1})

        # two equally good solutions, but with sampling mode off it will pick one
        sampleset = sampler.sample(bqm, sampling_mode=False)

        self.assertEqual(set(sampleset.variables), set('abc'))
        self.assertEqual(len(sampleset), 1)  # all should be fixed
        dimod.testing.assert_response_energies(sampleset, bqm)

    @unittest.skipUnless(cpp, "no cpp extensions built")
    def test_bowtie(self):
        sampler = ExtendedRoofDualityComposite(dimod.ExactSolver())

        J = {(u, v): 1 for (u, v) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]}
        J[(0, 3)] = -10
        bqm = dimod.BinaryQuadraticModel.from_ising({}, J)

        # 18 optimal solutions
        sampleset = sampler.sample(bqm)

        self.assertEqual(set(sampleset.variables), set(range(6)))
        dimod.testing.assert_response_energies(sampleset, bqm)

    @unittest.skipUnless(cpp, "no cpp extensions built")
    def test_bowtie_sampling_mode_off(self):
        sampler = ExtendedRoofDualityComposite(dimod.ExactSolver())

        J = {(u, v): 1 for (u, v) in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]}
        J[(0, 3)] = -10
        bqm = dimod.BinaryQuadraticModel.from_ising({0: 0.1}, J)

        # With sampling mode off, should pick a single solution
        sampleset = sampler.sample(bqm, sampling_mode=False)

        self.assertEqual(set(sampleset.variables), set(range(6)))
        self.assertEqual(len(sampleset), 1)  # all should be fixed
        dimod.testing.assert_response_energies(sampleset, bqm)
