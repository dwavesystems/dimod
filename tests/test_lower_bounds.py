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
import itertools
import dimod
from dimod.lower_bounds import lp_lower_bound, sdp_lower_bound


class TestLowerBounds(unittest.TestCase):

    def test_empty(self):
        bqm = dimod.BinaryQuadraticModel({}, {}, -1.0, dimod.SPIN)

        bound, _ = lp_lower_bound(bqm)
        self.assertEqual(bound, -1.0)

        bound, _ = sdp_lower_bound(bqm)
        self.assertEqual(bound, -1.0)

    def test_SPIN_BINARY(self):
        # bound should be the same regardless of whether SPIN or BINARY is passed in
        bqm = dimod.BinaryQuadraticModel({'a': 1}, {'ab': -2, 'bc': 0.5}, -1.0, dimod.BINARY)
        spin_bqm = bqm.change_vartype(dimod.SPIN, inplace=False)

        binary_bound, _ = lp_lower_bound(bqm)
        spin_bound, _ = lp_lower_bound(spin_bqm)
        self.assertEqual(binary_bound, spin_bound)

        binary_bound, _ = sdp_lower_bound(bqm)
        spin_bound, _ = sdp_lower_bound(spin_bqm)
        self.assertAlmostEqual(binary_bound, spin_bound, delta=0.01)

    def test_triangles(self):
        # example with LP bound is not tight, but is tight with cycle cuts
        bqm = dimod.BinaryQuadraticModel({'a': 0, 'b': 1, 'c': 3}, {'ab': 2, 'ac': -2, 'bc': -4}, 0.0, dimod.BINARY)
        ground_state = {'a': 0, 'b': 0, 'c': 0}
        ground_energy = bqm.energy(ground_state)

        bound, _ = lp_lower_bound(bqm)
        self.assertLess(bound, ground_energy)

        bound, _ = lp_lower_bound(bqm, cycle_cuts=4)
        self.assertAlmostEqual(bound, ground_energy)

        bound, _ = lp_lower_bound(bqm, integer_solve=True)
        self.assertAlmostEqual(bound, ground_energy)

        bound, _ = sdp_lower_bound(bqm)
        self.assertLess(bound, ground_energy)

    def test_ferromagnet(self):
        # LP bound should be tight for unfrustrated systems
        bqm = dimod.BinaryQuadraticModel({}, {e: -1 for e in itertools.combinations(range(5), 2)}, 0.0,
                                         dimod.SPIN)
        ground_state = {i: 1 for i in range(5)}
        ground_energy = bqm.energy(ground_state)

        bound, _ = lp_lower_bound(bqm)
        self.assertAlmostEqual(bound, ground_energy)

        bound, _ = lp_lower_bound(bqm, cycle_cuts=4)
        self.assertAlmostEqual(bound, ground_energy)

        bound, _ = lp_lower_bound(bqm, integer_solve=True)
        self.assertAlmostEqual(bound, ground_energy)

        bound, _ = sdp_lower_bound(bqm)
        self.assertAlmostEqual(bound, ground_energy, delta=0.01)

    def test_5cycle(self):
        # example with bounds not not tight, even with LP triangles (but SDP is tighter than LP)
        bqm = dimod.BinaryQuadraticModel({}, {e: 1 for e in [(i, i+1) for i in range(4)]+[(4, 0)]}, 0.0,
                                         dimod.SPIN)
        ground_state = {0: 1, 1: -1, 2: 1, 3: -1, 4: -1}
        ground_energy = bqm.energy(ground_state)

        bound, _ = lp_lower_bound(bqm)
        self.assertLess(bound, ground_energy)

        bound, _ = lp_lower_bound(bqm, cycle_cuts=4)
        self.assertLess(bound, ground_energy)

        bound, _ = sdp_lower_bound(bqm)
        self.assertLess(bound, ground_energy)

        bound, _ = lp_lower_bound(bqm, integer_solve=True)
        self.assertAlmostEqual(bound, ground_energy)
