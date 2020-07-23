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
from itertools import groupby
import itertools

import dimod

from dimod import ising_to_qubo, qubo_to_ising, ising_energy, qubo_energy
from dimod.exceptions import WriteableError
from dimod.utilities import LockableDict

import numpy as np

import time

num_vars = 20000

class TestMinMax(unittest.TestCase):
   
    def helper_lmin(self, bqm, bqm_type):

        print("Testing linear min of ", bqm_type, " with num_vars: ", num_vars)

        # Check that implementation is correct
        lmin = min(bqm.linear.values())
        self.assertEqual(lmin, bqm.linear.min())
        self.assertEqual(lmin, bqm.linear.npmin())
        self.assertEqual(lmin, bqm.linear.cymin())

        # Check time
        t = time.time()
        bqm.linear.min()
        print("python: ", time.time() - t)

        t = time.time()
        bqm.linear.npmin()
        print("np: ", time.time() - t)

        t = time.time()
        bqm.linear.cymin()
        print("cython: ", time.time() - t)

        print("\n lmin is: ", lmin, "\n")

    def helper_lmax(self, bqm, bqm_type):
        print("Testing linear max of ", bqm_type, " with num_vars: ", num_vars)

        # Check that implementation is correct
        lmax = max(bqm.linear.values())
        self.assertEqual(lmax, bqm.linear.max())
        self.assertEqual(lmax, bqm.linear.npmax())
        self.assertEqual(lmax, bqm.linear.cymax())

        # Check time
        t = time.time()
        bqm.linear.max()
        print("python: ", time.time() - t)

        t = time.time()
        bqm.linear.npmax()
        print("np: ", time.time() - t)

        t = time.time()
        bqm.linear.cymax()
        print("cython: ", time.time() - t)

        print("\n lmax is: ", lmax, "\n")

    def helper_qmin(self, bqm, bqm_type):

        print("Testing quadratic min of ", bqm_type, " with num_vars: ", num_vars)

        # Check that implementation is correct
        qmin = min(bqm.quadratic.values())
        self.assertEqual(qmin, bqm.quadratic.min())
        self.assertEqual(qmin, bqm.quadratic.npmin())
        self.assertEqual(qmin, bqm.quadratic.cymin())

        # Check time
        t = time.time()
        bqm.quadratic.min()
        print("python: ", time.time() - t)

        t = time.time()
        bqm.quadratic.npmin()
        print("np: ", time.time() - t)

        t = time.time()
        bqm.quadratic.cymin()
        print("cython: ", time.time() - t)

        print("\n qmin is: ", qmin, "\n")

    def helper_qmax(self, bqm, bqm_type):
        print("Testing quadratic max of ", bqm_type, " with num_vars: ", num_vars)

        # Check that implementation is correct
        qmax = max(bqm.quadratic.values())
        self.assertEqual(qmax, bqm.quadratic.max())
        self.assertEqual(qmax, bqm.quadratic.npmax())
        self.assertEqual(qmax, bqm.quadratic.cymax())

        # Check time
        t = time.time()
        bqm.quadratic.max()
        print("python: ", time.time() - t)

        t = time.time()
        bqm.quadratic.npmax()
        print("np: ", time.time() - t)

        t = time.time()
        bqm.quadratic.cymax()
        print("cython: ", time.time() - t)

        print("\n qmax is: ", qmax, "\n")


    def test_adjdict(self):

        bqm = dimod.AdjDictBQM(num_vars, 'SPIN')

        for (v,h) in bqm.linear.items():
            if v + 1 < num_vars:
                bqm.set_quadratic(v, v+1, 2)

        # D = np.arange(num_vars*num_vars).reshape((num_vars, num_vars))
        # bqm = dimod.AdjDictBQM(D, 'SPIN') 

        bqm_type = "AdjDictBQM"

        self.helper_lmin(bqm, bqm_type)
        self.helper_lmax(bqm, bqm_type)

        self.helper_qmin(bqm, bqm_type)   
        self.helper_qmax(bqm, bqm_type)

    def test_adjarray(self):

        bqm2 = dimod.AdjDictBQM(num_vars, 'SPIN')

        for (v,h) in bqm2.linear.items():
            if v + 1 < num_vars:
                bqm2.set_quadratic(v, v+1, 2)

        bqm = dimod.AdjArrayBQM(bqm2)

        # D = np.arange(num_vars*num_vars).reshape((num_vars, num_vars))
        # bqm = dimod.AdjArrayBQM(D, 'SPIN') 

        bqm_type = "AdjArrayBQM"

        self.helper_lmin(bqm, bqm_type)
        self.helper_lmax(bqm, bqm_type)

        self.helper_qmin(bqm, bqm_type)   
        self.helper_qmax(bqm, bqm_type)

    def test_adjvector(self):

        bqm2 = dimod.AdjDictBQM(num_vars, 'SPIN')

        for (v,h) in bqm2.linear.items():
            if v + 1 < num_vars:
                bqm2.set_quadratic(v, v+1, 2)

        bqm = dimod.AdjVectorBQM(bqm2)

        # D = np.arange(num_vars*num_vars).reshape((num_vars, num_vars))
        # bqm = dimod.AdjVectorBQM(D, 'SPIN') 

        bqm_type = "AdjVectorBQM"

        self.helper_lmin(bqm, bqm_type)
        self.helper_lmax(bqm, bqm_type)

        self.helper_qmin(bqm, bqm_type)   
        self.helper_qmax(bqm, bqm_type)

    def test_adjmap(self):

        bqm2 = dimod.AdjDictBQM(num_vars, 'SPIN')

        for (v,h) in bqm2.linear.items():
            if v + 1 < num_vars:
                bqm2.set_quadratic(v, v+1, 2)

        bqm = dimod.AdjMapBQM(bqm2)

        # D = np.arange(num_vars*num_vars).reshape((num_vars, num_vars))
        # bqm = dimod.AdjMapBQM(D, 'SPIN') 

        bqm_type = "AdjMapBQM"

        self.helper_lmin(bqm, bqm_type)
        self.helper_lmax(bqm, bqm_type)

        self.helper_qmin(bqm, bqm_type)   
        self.helper_qmax(bqm, bqm_type)
