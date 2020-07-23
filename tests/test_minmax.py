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

from __future__ import division

import unittest

import dimod
import numpy as np
import time

num_vars = 1000

class TestMinMax(unittest.TestCase):

    def linear_minmax(self, bqm):
        lmin = min(bqm.linear.values())
        self.assertEqual(lmin, bqm.linear.min())

        lmax = max(bqm.linear.values())
        self.assertEqual(lmax, bqm.linear.max())
   
    def quad_minmax(self, bqm):
        qmin = min(bqm.quadratic.values())
        self.assertEqual(qmin, bqm.quadratic.min())
       
        qmax = max(bqm.quadratic.values())
        self.assertEqual(qmax, bqm.quadratic.max())

    def test_adjdict_minmax(self):
        D = np.arange(num_vars*num_vars).reshape((num_vars, num_vars))
        bqm = dimod.AdjDictBQM(D, 'SPIN')

        self.linear_minmax(bqm) 
        self.quad_minmax(bqm)

    def test_adjarray_minmax(self):
        D = np.arange(num_vars*num_vars).reshape((num_vars, num_vars))
        bqm = dimod.AdjArrayBQM(D, 'SPIN') 

        self.linear_minmax(bqm) 
        self.quad_minmax(bqm)

    def test_adjvector_minmax(self):
        D = np.arange(num_vars*num_vars).reshape((num_vars, num_vars))
        bqm = dimod.AdjVectorBQM(D, 'SPIN') 

        self.linear_minmax(bqm) 
        self.quad_minmax(bqm)

    def test_adjmap_minmax(self):
        D = np.arange(num_vars*num_vars).reshape((num_vars, num_vars))
        bqm = dimod.AdjMapBQM(D, 'SPIN') 

        self.linear_minmax(bqm) 
        self.quad_minmax(bqm)
