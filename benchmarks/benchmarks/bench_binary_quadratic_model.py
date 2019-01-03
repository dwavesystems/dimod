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
# ================================================================================================
import itertools
import copy

import numpy as np
import dimod


class Benchmark(object):
    goal_time = 0.25


class Construction(Benchmark):
    def setup(self):
        self.h_K100 = {v: 0 for v in range(100)}
        self.J_K100 = {edge: 0 for edge in itertools.combinations(range(100), 2)}

    def time_k100(self):
        dimod.BinaryQuadraticModel(self.h_K100, self.J_K100, 0.0, dimod.SPIN)

    def mem_k100(self):
        return dimod.BinaryQuadraticModel(self.h_K100, self.J_K100, 0.0, dimod.SPIN)

    def time_empty(self):
        dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)

    def mem_empty(self):
        return dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)


class ConstructionVeryLarge(Benchmark):
    def setup(self):
        self.J_15000 = J = {}

        # make a graph of approximately degree 15, 4394 nodes, 32617 edges
        m = n = 13
        t = 13
        J.update((((i, j, 0, k0), (i, j, 1, k1)), 1)
                 for i in range(n)
                 for j in range(m)
                 for k0 in range(t)
                 for k1 in range(t))
        J.update((((i, j, 1, k), (i, j+1, 1, k)), 1)
                 for i in range(m)
                 for j in range(n-1)
                 for k in range(t))
        J.update((((i, j, 0, k), (i+1, j, 0, k)), 1)
                 for i in range(m-1)
                 for j in range(n)
                 for k in range(t))

    def time_J15000(self):
        dimod.BinaryQuadraticModel({}, self.J_15000, 0.0, dimod.SPIN)

    def mem_J15000(self):
        return dimod.BinaryQuadraticModel({}, self.J_15000, 0.0, dimod.SPIN)


class Copy(Benchmark):
    def setup(self):
        h_K100 = {v: 0 for v in range(100)}
        J_K100 = {edge: 0 for edge in itertools.combinations(range(100), 2)}
        self.bqm_k100 = dimod.BinaryQuadraticModel(h_K100, J_K100, 0.0, dimod.SPIN)

        self.bqm_empty = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)

    def time_k100(self):
        self.bqm_k100.copy()

    def time_empty(self):
        self.bqm_empty.copy()


class Deepcopy(Benchmark):
    """How long it takes to do a deep copy"""
    def setup(self):
        h_K100 = {v: 0 for v in range(100)}
        J_K100 = {edge: 0 for edge in itertools.combinations(range(100), 2)}
        self.bqm_k100 = dimod.BinaryQuadraticModel(h_K100, J_K100, 0.0, dimod.SPIN)

        self.bqm_empty = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)

    def time_k100(self):
        copy.deepcopy(self.bqm_k100)

    def time_empty(self):
        copy.deepcopy(self.bqm_empty)


class Energies(Benchmark):
    def setup(self):
        h_K100 = {v: 0 for v in range(100)}
        J_K100 = {edge: 0 for edge in itertools.combinations(range(100), 2)}
        self.bqm_k100 = dimod.BinaryQuadraticModel(h_K100, J_K100, 0.0, dimod.SPIN)

        self.samples_1000x100 = np.ones((1000, 100), dtype=np.int8)

    def time_k100(self):
        self.bqm_k100.energies(self.samples_1000x100)
