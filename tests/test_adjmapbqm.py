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

import numpy as np

from dimod.bqm.adjmapbqm import AdjMapBQM
from tests.test_adjarraybqm import TestAPI


class TestAdjMapBQMAPI(TestAPI, unittest.TestCase):
    """Runs the generic tests"""
    BQM = AdjMapBQM


class TestConstruction(unittest.TestCase):
    """Tests for properties and special methods and mutability"""

    def test_empty(self):
        bqm = AdjMapBQM()
        self.assertEqual(bqm.shape, (0, 0))

    def test_integral_zero(self):
        bqm = AdjMapBQM(0)
        self.assertEqual(bqm.shape, (0, 0))

    def test_integral_nonzero(self):
        bqm = AdjMapBQM(5)
        self.assertEqual(bqm.shape, (5, 0))

        adj = bqm.to_lists()
        self.assertEqual(adj, list(([], 0) for _ in range(5)))

    def test_dense(self):
        bqm = AdjMapBQM(np.triu(np.ones((5, 5))))
        adj = bqm.to_lists()
        self.assertEqual(adj, [([(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)], 1.0),
                               ([(0, 1.0), (2, 1.0), (3, 1.0), (4, 1.0)], 1.0),
                               ([(0, 1.0), (1, 1.0), (3, 1.0), (4, 1.0)], 1.0),
                               ([(0, 1.0), (1, 1.0), (2, 1.0), (4, 1.0)], 1.0),
                               ([(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)], 1.0)])


# class TestLinearBase(unittest.TestCase):
#     def test_append(self):
#         bqm = AdjMapBQM()
#         self.assertEqual(bqm.shape, (0, 0))
#         bqm.append_linear(.5)
#         self.assertEqual(bqm.shape, (1, 0))
#         self.assertEqual(bqm.to_lists(), [([], .5)])

#     def test_pop(self):
#         bqm = AdjMapBQM([[.5, 1], [0, 0]])
#         self.assertEqual(bqm.shape, (2, 1))
#         bqm.pop_linear()
#         self.assertEqual(bqm.shape, (1, 0))
#         self.assertEqual(bqm.to_lists(), [([], .5)])


# class TestQuadraticBase(unittest.TestCase):
#     def test_get(self):
#         bqm = AdjMapBQM([[.5, 1], [0, 0]])
#         self.assertEqual(bqm.get_quadratic(0, 1), 1)
#         self.assertEqual(bqm.get_quadratic(1, 0), 1)

#     def test_remove(self):
#         bqm = AdjMapBQM([[.5, 1], [0, 0]])
#         bqm.remove_quadratic(0, 1)
#         self.assertEqual(bqm.shape, (2, 0))
#         bqm.remove_quadratic(0, 1)  # not throw error

#     def test_set(self):
#         bqm = AdjMapBQM(2)
#         bqm.set_quadratic(0, 1, .5)
#         self.assertEqual(bqm.get_quadratic(0, 1), .5)
#         bqm.set_quadratic(0, 1, .7)
#         self.assertEqual(bqm.get_quadratic(0, 1), .7)


# class TestToAdjArray(unittest.TestCase):
#     def test_empty(self):
#         lin, quad = AdjMapBQM().to_adjarray().to_lists()
#         self.assertEqual(lin, [])
#         self.assertEqual(quad, [])

#     def test_linear(self):
#         bqm = AdjMapBQM(3)
#         bqm.append_linear(.5)

#         lin, quad = bqm.to_adjarray().to_lists()
#         self.assertEqual(lin, [(0, 0), (0, 0), (0, 0), (0, .5)])
#         self.assertEqual(quad, [])

#     def test_3path(self):
#         bqm = AdjMapBQM(2)
#         bqm.append_linear(.5)
#         bqm.set_quadratic(1, 0, 1.7)
#         bqm.set_quadratic(2, 1, -3)

#         lin, quad = bqm.to_adjarray().to_lists()
#         self.assertEqual(lin, [(0, 0), (1, 0), (3, .5)])
#         self.assertEqual(quad, [(1, 1.7), (0, 1.7), (2, -3.0), (1, -3.0)])
