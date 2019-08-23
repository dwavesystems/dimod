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


class TestFixedShapeBQMAPI(TestAPI, unittest.TestCase):
    """Runs the tests that run on AdjArrayBQM"""
    BQM = AdjMapBQM


class TestAdjMapBQM(unittest.TestCase):
    """Test the mutation methods"""

    def test_add_variable_exception(self):
        bqm = AdjMapBQM()
        with self.assertRaises(TypeError):
            bqm.add_variable([])

    def test_add_variable_labelled(self):
        bqm = AdjMapBQM()
        bqm.add_variable('a')
        bqm.add_variable(1)
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(list(bqm.iter_variables()), ['a', 1])
        bqm.add_variable()
        self.assertEqual(bqm.shape, (3, 0))
        self.assertEqual(list(bqm.iter_variables()), ['a', 1, 2])

    def test_add_variable_int_labelled(self):
        bqm = AdjMapBQM()
        self.assertEqual(bqm.add_variable(1), 1)
        self.assertEqual(bqm.add_variable(), 0)  # 1 is already taken
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(bqm.add_variable(), 2)
        self.assertEqual(bqm.shape, (3, 0))

    def test_add_variable_unlabelled(self):
        bqm = AdjMapBQM()
        bqm.add_variable()
        bqm.add_variable()
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(list(bqm.iter_variables()), [0, 1])

    def test_pop_variable_labelled(self):
        bqm = AdjMapBQM()
        bqm.add_variable('a')
        bqm.add_variable(1)
        bqm.add_variable(0)
        self.assertEqual(bqm.pop_variable(), 0)
        self.assertEqual(bqm.pop_variable(), 1)
        self.assertEqual(bqm.pop_variable(), 'a')
        with self.assertRaises(ValueError):
            bqm.pop_variable()

    def test_pop_variable_unlabelled(self):
        bqm = AdjMapBQM(2)
        self.assertEqual(bqm.pop_variable(), 1)
        self.assertEqual(bqm.pop_variable(), 0)
        with self.assertRaises(ValueError):
            bqm.pop_variable()

    def test_remove_interaction(self):
        bqm = AdjMapBQM(np.triu(np.ones((3, 3))))
        self.assertTrue(bqm.remove_interaction(0, 1))
        self.assertFalse(bqm.remove_interaction(0, 1))
        self.assertEqual(bqm.shape, (3, 2))
        with self.assertRaises(ValueError):
            bqm.get_quadratic(0, 1)

    def test_set_quadratic_exception(self):
        bqm = AdjMapBQM()
        with self.assertRaises(TypeError):
            bqm.set_quadratic([], 1, .5)
        with self.assertRaises(TypeError):
            bqm.set_quadratic(1, [], .5)


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
