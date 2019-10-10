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
# =============================================================================

import unittest

import dimod.testing as dtest
from dimod.vartypes import Vartype

from dimod import BinaryQuadraticModel
from dimod import CutVertexComposite, ExactSolver, FixedVariableComposite, ConnectedComponentsComposite
from dimod import SampleSet
from dimod.reference.composites.cutvertex import BiconnectedTreeDecomposition
import itertools


class TestCutVerticesComposite(unittest.TestCase):

    def test_instantiation_smoketest(self):
        sampler = CutVertexComposite(ExactSolver())

        dtest.assert_sampler_api(sampler)

    def test_sample(self):
        bqm = BinaryQuadraticModel(linear={0: 0.0, 1: 1.0, 2: -1.0, 3: 0.0, 4: -0.5},
                                   quadratic={(0, 1): 1.5, (0, 2): 0.7, (1, 2): -0.3, (0, 3): 0.9, (0, 4): 1.6,
                                              (3, 4): -0.3},
                                   offset=0.0,
                                   vartype=Vartype.SPIN)
        sampler = CutVertexComposite(ExactSolver())
        response = sampler.sample(bqm)
        self.assertIsInstance(response, SampleSet)

        ground_response = ExactSolver().sample(bqm)
        self.assertEqual(response.first.sample, ground_response.first.sample)
        self.assertAlmostEqual(response.first.energy, ground_response.first.energy)

    def test_empty_bqm(self):
        bqm = BinaryQuadraticModel(linear={1: -1.3, 4: -0.5},
                                   quadratic={(1, 4): -0.6},
                                   offset=0,
                                   vartype=Vartype.SPIN)

        fixed_variables = {1: -1, 4: -1}
        sampler = FixedVariableComposite(CutVertexComposite(ExactSolver()))
        response = sampler.sample(bqm, fixed_variables=fixed_variables)
        self.assertIsInstance(response, SampleSet)

    def test_sample_two_components(self):
        bqm = BinaryQuadraticModel({0: 0.0, 1: 4.0, 2: -4.0, 3: 0.0}, {(0, 1): -4.0, (2, 3): 4.0}, 0.0, Vartype.BINARY)

        sampler = ConnectedComponentsComposite(CutVertexComposite(ExactSolver()))
        response = sampler.sample(bqm)
        self.assertIsInstance(response, SampleSet)
        self.assertEqual(response.first.sample, {0: 0, 1: 0, 2: 1, 3: 0})
        self.assertAlmostEqual(response.first.energy, bqm.energy({0: 0, 1: 0, 2: 1, 3: 0}))

    def test_sample_pass_treedecomp(self):
        bqm = BinaryQuadraticModel(linear={0: 0.0, 1: 1.0, 2: -1.0, 3: 0.0, 4: -0.5},
                                   quadratic={(0, 1): 1.5, (0, 2): 0.7, (1, 2): -0.3, (0, 3): 0.9, (0, 4): 1.6,
                                              (3, 4): -0.3},
                                   offset=0.0,
                                   vartype=Vartype.SPIN)

        sampler = CutVertexComposite(ExactSolver())
        tree_decomp = BiconnectedTreeDecomposition(bqm)
        response = sampler.sample(bqm, tree_decomp=tree_decomp)
        self.assertIsInstance(response, SampleSet)
        ground_response = ExactSolver().sample(bqm)
        self.assertEqual(response.first.sample, ground_response.first.sample)
        self.assertAlmostEqual(response.first.energy, ground_response.first.energy)

    def test_forked_tree_decomp(self):
        comps = [[0, 1, 2], [2, 3, 4], [3, 5, 6], [4, 7, 8]]
        J = {(u, v): -1 for c in comps for (u, v) in itertools.combinations(c, 2)}
        h = {0: 0.1}
        bqm = BinaryQuadraticModel.from_ising(h, J)
        sampler = CutVertexComposite(ExactSolver())
        response = sampler.sample(bqm)

        ground_state = {i: -1 for i in range(9)}
        self.assertEqual(response.first.sample, ground_state)
        self.assertAlmostEqual(response.first.energy, bqm.energy(ground_state))

    def test_simple_tree(self):
        J = {(u, v): -1 for (u, v) in [(0, 3), (1, 3), (2, 3)]}
        h = {3: 5}
        bqm = BinaryQuadraticModel.from_ising(h, J)
        sampler = CutVertexComposite(ExactSolver())
        response = sampler.sample(bqm)

        ground_state = {i: -1 for i in range(4)}
        self.assertEqual(response.first.sample, ground_state)
        self.assertAlmostEqual(response.first.energy, bqm.energy(ground_state))
