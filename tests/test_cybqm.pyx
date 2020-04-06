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

from dimod.bqm import AdjArrayBQM, AdjMapBQM, AdjVectorBQM
from dimod.bqm cimport cyAdjArrayBQM, cyAdjMapBQM, cyAdjVectorBQM


__all__ = ['TestRelabel']


class TestRelabel(unittest.TestCase):
    def test_adjarray_natural_labelling(self):
        
        cdef cyAdjArrayBQM bqm = AdjArrayBQM(np.arange(25).reshape((5, 5)), 'SPIN')
        bqm.relabel_variables({0: 4, 1: 3, 3: 1, 4: 0})

        _, inverse = bqm.relabel_variables_as_integers(inplace=True)

        self.assertEqual(bqm.num_variables, bqm.bqm_.num_variables())
        self.assertEqual(bqm.num_interactions, bqm.bqm_.num_interactions())

        for v in range(3):
            self.assertEqual(bqm.get_linear(v), bqm.bqm_.get_linear(v))

        for u, v in bqm.quadratic:
            self.assertEqual(bqm.get_quadratic(u, v),
                             bqm.bqm_.get_quadratic(u, v).first)

    def test_adjmap_natural_labelling(self):
        
        cdef cyAdjMapBQM bqm = AdjMapBQM(np.arange(25).reshape((5, 5)), 'SPIN')
        bqm.relabel_variables({0: 4, 1: 3, 3: 1, 4: 0})

        _, inverse = bqm.relabel_variables_as_integers(inplace=True)

        self.assertEqual(bqm.num_variables, bqm.bqm_.num_variables())
        self.assertEqual(bqm.num_interactions, bqm.bqm_.num_interactions())

        for v in range(3):
            self.assertEqual(bqm.get_linear(v), bqm.bqm_.get_linear(v))

        for u, v in bqm.quadratic:
            self.assertEqual(bqm.get_quadratic(u, v),
                             bqm.bqm_.get_quadratic(u, v).first)

    def test_adjvector_natural_labelling(self):
        
        cdef cyAdjVectorBQM bqm = AdjVectorBQM(np.arange(25).reshape((5, 5)), 'SPIN')
        bqm.relabel_variables({0: 4, 1: 3, 3: 1, 4: 0})

        _, inverse = bqm.relabel_variables_as_integers(inplace=True)

        self.assertEqual(bqm.num_variables, bqm.bqm_.num_variables())
        self.assertEqual(bqm.num_interactions, bqm.bqm_.num_interactions())

        for v in range(3):
            self.assertEqual(bqm.get_linear(v), bqm.bqm_.get_linear(v))

        for u, v in bqm.quadratic:
            self.assertEqual(bqm.get_quadratic(u, v),
                             bqm.bqm_.get_quadratic(u, v).first)
