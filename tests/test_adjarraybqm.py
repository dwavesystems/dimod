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
import itertools
import unittest

import numpy as np

from dimod.bqm.adjarraybqm import AdjArrayBQM


class TestAPI:
    """
    Tests for BQMs like AdjArrayBQM (doesn't try to change the shape)
    """
    def test_empty_shape(self):
        self.assertEqual(self.BQM().shape, (0, 0))
        self.assertEqual(self.BQM(0).shape, (0, 0))

        self.assertEqual(self.BQM().num_variables, 0)
        self.assertEqual(self.BQM(0).num_variables, 0)

        self.assertEqual(self.BQM().num_interactions, 0)
        self.assertEqual(self.BQM(0).num_interactions, 0)

        self.assertEqual(len(self.BQM()), 0)
        self.assertEqual(len(self.BQM(0)), 0)

    def test_disconnected_shape(self):
        bqm = self.BQM(5)

        self.assertEqual(bqm.shape, (5, 0))
        self.assertEqual(bqm.num_variables, 5)
        self.assertEqual(bqm.num_interactions, 0)
        self.assertEqual(len(bqm), 5)

    def test_3x3array_shape(self):
        bqm = self.BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]])

        self.assertEqual(bqm.shape, (3, 2))
        self.assertEqual(bqm.num_variables, 3)
        self.assertEqual(bqm.num_interactions, 2)
        self.assertEqual(len(bqm), 3)

    def test_disconnected_get_linear(self):
        bqm = self.BQM(5)

        for v in range(5):
            self.assertEqual(bqm.get_linear(v), 0)

        with self.assertRaises(ValueError):
            bqm.get_linear(-1)

        with self.assertRaises(ValueError):
            bqm.get_linear(5)

    def test_3x3array_get_quadratic(self):
        bqm = self.BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]])

        self.assertEqual(bqm.get_quadratic(0, 1), 1)
        self.assertEqual(bqm.get_quadratic(1, 0), 1)

        self.assertEqual(bqm.get_quadratic(0, 2), 2)
        self.assertEqual(bqm.get_quadratic(2, 0), 2)

        # todo test non-existant edge


class TestAdjArrayBQMAPI(TestAPI, unittest.TestCase):
    """Runs the generic tests"""
    BQM = AdjArrayBQM


class TestConstruction(unittest.TestCase):
    """Tests for properties and special methods"""

    def test_empty(self):
        bqm = AdjArrayBQM()

        self.assertEqual(len(bqm), 0)
        self.assertEqual(bqm.shape, (0, 0))

        lin, quad = bqm.to_lists()
        self.assertEqual(lin, [])
        self.assertEqual(quad, [])

    def test_integral_nonzero(self):
        bqm = AdjArrayBQM(1000)

        self.assertEqual(len(bqm), 1000)
        self.assertEqual(bqm.shape, (1000, 0))

        lin, quad = bqm.to_lists()
        self.assertEqual(lin, [(0, 0) for _ in range(1000)])
        self.assertEqual(quad, [])

    def test_dense_triu(self):
        bqm = AdjArrayBQM(np.triu(np.ones((5, 5))))

        self.assertEqual(bqm.shape, (5, 10))
        lin, quad = bqm.to_lists()
        self.assertEqual(lin, [(d, 1) for d in range(0, 5*4, 4)])
        self.assertEqual(quad, [(v, 1)
                                for u in range(5)
                                for v in range(5)
                                if u != v])

    def test_dense(self):
        bqm = AdjArrayBQM([[.1, 1, 2], [0, 0, 0], [1, 1, 0]])

        self.assertEqual(bqm.shape, (3, 3))

        lin, quad = bqm.to_lists()
        self.assertEqual(lin, [(0, .1), (2, 0), (4, 0)])
        self.assertEqual(quad, [(1, 1), (2, 3),
                                (0, 1), (2, 1),
                                (0, 3), (1, 1)])


# class TestEnergies(unittest.TestCase):
#     def test_2path(self):
#         bqm = AdjArrayBQM([[.1, -1], [0, -.2]])
#         samples = [[-1, -1],
#                    [-1, +1],
#                    [+1, -1],
#                    [+1, +1]]

#         energies = bqm.energies(np.asarray(samples))

#         np.testing.assert_array_almost_equal(energies, [-.9, .7, 1.3, -1.1])

#     def test_5chain(self):
#         arr = np.tril(np.triu(np.ones((5, 5)), 1), 1)
#         bqm = AdjArrayBQM(arr)
#         samples = [[-1, +1, -1, +1, -1]]

#         energies = bqm.energies(np.asarray(samples))
#         np.testing.assert_array_almost_equal(energies, [-4])

#     def test_random(self):
#         bqm = AdjArrayBQM([[0, -1, 0, 0],
#                            [0, 0, .5, .2],
#                            [0, 0, 0, 1.3],
#                            [0, 0, 0, 0]])

#         J = {(0, 1): -1, (1, 2): .5, (1, 3): .2, (2, 3): 1.3}

#         for sample in itertools.product((-1, 1), repeat=len(bqm)):
#             energy, = bqm.energies(np.atleast_2d(sample))

#             target = sum(b*sample[u]*sample[v] for (u, v), b in J.items())

#             self.assertAlmostEqual(energy, target)
