# distutils: language = c++
# cython: language_level=3
# Copyright 2020 D-Wave Systems Inc.
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

# note: for these tests to be discoverable, they must be imported in
# tests/__init__.py

import unittest

import numpy as np

from dimod.bqm.cppbqm cimport AdjArrayBQM as cppAdjArrayBQM
from dimod.bqm.cppbqm cimport AdjMapBQM as cppAdjMapBQM
from dimod.bqm.cppbqm cimport AdjVectorBQM as cppAdjVectorBQM

__all__ = ['TestConstruction',
           'TestPopVariable',
           'TestRemoveInteraction',
           ]

ctypedef cppAdjArrayBQM[size_t, float] cppAdjArrayBQM_t
ctypedef cppAdjMapBQM[size_t, float] cppAdjMapBQM_t
ctypedef cppAdjVectorBQM[size_t, float] cppAdjVectorBQM_t


class TestConstruction(unittest.TestCase):
    def test_adjarraybqm_from_array(self):
        cdef int nv = 9
        narr = np.arange(81, dtype=np.double).reshape((nv, nv))
        narr[4, 6] = narr[6, 4] = 0  # make slightly sparse
        cdef double [:, :] Q = narr

        cdef cppAdjArrayBQM_t bqm = cppAdjArrayBQM_t(&Q[0, 0], nv)

        self.assertEqual(bqm.num_variables(), nv)
        self.assertEqual(bqm.num_interactions(), nv*(nv-1)/2-1)

        cdef int ui, vi
        cdef double bias
        for vi in range(nv):
            self.assertEqual(bqm.get_linear(vi), narr[vi, vi])

        for ui in range(nv):
            for vi in range(ui+1, nv):
                bias = Q[ui, vi] + Q[vi, ui]
                if bias:
                    self.assertTrue(bqm.get_quadratic(ui, vi).second)
                    self.assertTrue(bqm.get_quadratic(ui, vi).second)
                    self.assertEqual(bqm.get_quadratic(ui, vi).first, bias)
                    self.assertEqual(bqm.get_quadratic(vi, ui).first, bias)
                else:
                    self.assertFalse(bqm.get_quadratic(ui, vi).second)
                    self.assertFalse(bqm.get_quadratic(ui, vi).second)

    def test_adjarraybqm_from_array_ignore_diagonal(self):
        cdef int nv = 9
        narr = np.arange(81, dtype=np.double).reshape((nv, nv))
        narr[4, 6] = narr[6, 4] = 0  # make slightly sparse
        cdef double [:, :] Q = narr

        cdef cppAdjArrayBQM_t bqm = cppAdjArrayBQM_t(&Q[0, 0], nv, True)

        self.assertEqual(bqm.num_variables(), nv)
        self.assertEqual(bqm.num_interactions(), nv*(nv-1)/2-1)

        cdef int ui, vi
        cdef double bias
        for vi in range(nv):
            self.assertEqual(bqm.get_linear(vi), 0)

        for ui in range(nv):
            for vi in range(ui+1, nv):
                bias = Q[ui, vi] + Q[vi, ui]
                if bias:
                    self.assertTrue(bqm.get_quadratic(ui, vi).second)
                    self.assertTrue(bqm.get_quadratic(ui, vi).second)
                    self.assertEqual(bqm.get_quadratic(ui, vi).first, bias)
                    self.assertEqual(bqm.get_quadratic(vi, ui).first, bias)
                else:
                    self.assertFalse(bqm.get_quadratic(ui, vi).second)
                    self.assertFalse(bqm.get_quadratic(ui, vi).second)

    def test_adjmapbqm_from_array(self):
        cdef int nv = 9
        narr = np.arange(81, dtype=np.double).reshape((nv, nv))
        narr[4, 6] = narr[6, 4] = 0  # make slightly sparse
        cdef double [:, :] Q = narr

        cdef cppAdjMapBQM_t bqm = cppAdjMapBQM_t(&Q[0, 0], nv)

        self.assertEqual(bqm.num_variables(), nv)
        self.assertEqual(bqm.num_interactions(), nv*(nv-1)/2-1)

        cdef int ui, vi
        cdef double bias
        for vi in range(nv):
            self.assertEqual(bqm.get_linear(vi), narr[vi, vi])

        for ui in range(nv):
            for vi in range(ui+1, nv):
                bias = Q[ui, vi] + Q[vi, ui]
                if bias:
                    self.assertTrue(bqm.get_quadratic(ui, vi).second)
                    self.assertTrue(bqm.get_quadratic(ui, vi).second)
                    self.assertEqual(bqm.get_quadratic(ui, vi).first, bias)
                    self.assertEqual(bqm.get_quadratic(vi, ui).first, bias)
                else:
                    self.assertFalse(bqm.get_quadratic(ui, vi).second)
                    self.assertFalse(bqm.get_quadratic(ui, vi).second)

    def test_adjmapbqm_from_array_ignore_diagonal(self):
        cdef int nv = 9
        narr = np.arange(81, dtype=np.double).reshape((nv, nv))
        narr[4, 6] = narr[6, 4] = 0  # make slightly sparse
        cdef double [:, :] Q = narr

        cdef cppAdjMapBQM_t bqm = cppAdjMapBQM_t(&Q[0, 0], nv, True)

        self.assertEqual(bqm.num_variables(), nv)
        self.assertEqual(bqm.num_interactions(), nv*(nv-1)/2-1)

        cdef int ui, vi
        cdef double bias
        for vi in range(nv):
            self.assertEqual(bqm.get_linear(vi), 0)

        for ui in range(nv):
            for vi in range(ui+1, nv):
                bias = Q[ui, vi] + Q[vi, ui]
                if bias:
                    self.assertTrue(bqm.get_quadratic(ui, vi).second)
                    self.assertTrue(bqm.get_quadratic(ui, vi).second)
                    self.assertEqual(bqm.get_quadratic(ui, vi).first, bias)
                    self.assertEqual(bqm.get_quadratic(vi, ui).first, bias)
                else:
                    self.assertFalse(bqm.get_quadratic(ui, vi).second)
                    self.assertFalse(bqm.get_quadratic(ui, vi).second)


    def test_adjvectorbqm_from_array(self):
        cdef int nv = 9
        narr = np.arange(81, dtype=np.double).reshape((nv, nv))
        narr[4, 6] = narr[6, 4] = 0  # make slightly sparse
        cdef double [:, :] Q = narr

        cdef cppAdjVectorBQM_t bqm = cppAdjVectorBQM_t(&Q[0, 0], nv)

        self.assertEqual(bqm.num_variables(), nv)
        self.assertEqual(bqm.num_interactions(), nv*(nv-1)/2-1)

        cdef int ui, vi
        cdef double bias
        for vi in range(nv):
            self.assertEqual(bqm.get_linear(vi), narr[vi, vi])

        for ui in range(nv):
            for vi in range(ui+1, nv):
                bias = Q[ui, vi] + Q[vi, ui]
                if bias:
                    self.assertTrue(bqm.get_quadratic(ui, vi).second)
                    self.assertTrue(bqm.get_quadratic(ui, vi).second)
                    self.assertEqual(bqm.get_quadratic(ui, vi).first, bias)
                    self.assertEqual(bqm.get_quadratic(vi, ui).first, bias)
                else:
                    self.assertFalse(bqm.get_quadratic(ui, vi).second)
                    self.assertFalse(bqm.get_quadratic(ui, vi).second)

    def test_adjvectorbqm_from_array_ignore_diagonal(self):
        cdef int nv = 9
        narr = np.arange(81, dtype=np.double).reshape((nv, nv))
        narr[4, 6] = narr[6, 4] = 0  # make slightly sparse
        cdef double [:, :] Q = narr

        cdef cppAdjVectorBQM_t bqm = cppAdjVectorBQM_t(&Q[0, 0], nv, True)

        self.assertEqual(bqm.num_variables(), nv)
        self.assertEqual(bqm.num_interactions(), nv*(nv-1)/2-1)

        cdef int ui, vi
        cdef double bias
        for vi in range(nv):
            self.assertEqual(bqm.get_linear(vi), 0)

        for ui in range(nv):
            for vi in range(ui+1, nv):
                bias = Q[ui, vi] + Q[vi, ui]
                if bias:
                    self.assertTrue(bqm.get_quadratic(ui, vi).second)
                    self.assertTrue(bqm.get_quadratic(ui, vi).second)
                    self.assertEqual(bqm.get_quadratic(ui, vi).first, bias)
                    self.assertEqual(bqm.get_quadratic(vi, ui).first, bias)
                else:
                    self.assertFalse(bqm.get_quadratic(ui, vi).second)
                    self.assertFalse(bqm.get_quadratic(ui, vi).second)


class TestPopVariable(unittest.TestCase):
    def test_adjvectorbqm_typical(self):
        cdef cppAdjVectorBQM_t bqm = cppAdjVectorBQM_t()

        for _ in range(3):        
            bqm.add_variable()

        bqm.set_quadratic(0, 1, 1)
        bqm.set_quadratic(0, 2, 1)

        bqm.pop_variable()

        self.assertEqual(bqm.num_variables(), 2)
        self.assertEqual(bqm.num_interactions(), 1)
        self.assertEqual(bqm.adj[0].first.size(), 1)
        self.assertEqual(bqm.adj[1].first.size(), 1)
        self.assertEqual(bqm.adj[0].first[0].first, 1)
        self.assertEqual(bqm.adj[1].first[0].first, 0)


class TestRemoveInteraction(unittest.TestCase):
    def test_adjvectorbqm_typical(self):
        cdef cppAdjVectorBQM_t bqm = cppAdjVectorBQM_t()

        for _ in range(3):        
            bqm.add_variable()

        bqm.set_quadratic(0, 2, .5)
        bqm.set_quadratic(0, 1, 6)
        bqm.set_quadratic(1, 2, 1.6)

        bqm.remove_interaction(0, 1)

        self.assertEqual(bqm.num_variables(), 3)
        self.assertEqual(bqm.num_interactions(), 2)
        self.assertEqual(bqm.adj[0].first.size(), 1)
        self.assertEqual(bqm.adj[1].first.size(), 1)
        self.assertEqual(bqm.adj[2].first.size(), 2)
