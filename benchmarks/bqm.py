# Copyright 2021 D-Wave Systems Inc.
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

import dimod
import numpy as np


class TimeEnergies:
    params = ([10, 150, 250],
              [dimod.AdjVectorBQM, dimod.DictBQM, dimod.Float32BQM, dimod.Float64BQM])

    def setup(self, n, cls):
        self.bqm = cls(np.ones((n, n)), 'BINARY')
        self.samples = np.ones((1000, n))

    def teardown(self, n, cls):
        del self.bqm
        del self.samples

    def time_energies(self, n, cls):
        self.bqm.energies(self.samples)


class TimeGetQuadratic:
    params = ([10, 150, 250, 500],
              [dimod.AdjVectorBQM, dimod.DictBQM, dimod.Float32BQM, dimod.Float64BQM])

    def setup(self, n, cls):
        self.bqm = cls(np.ones((n, n)), 'BINARY')

    def teardown(self, n, cls):
        del self.bqm

    def time_get_quadratic(self, n, cls):
        bqm = self.bqm
        for u in range(n):
            for v in range(u+1, n):
                bqm.get_quadratic(u, v)


class TimeIterNeighborhood:
    params = ([10, 150, 250, 500],
              [dimod.DictBQM, dimod.Float32BQM, dimod.Float64BQM])

    def setup(self, n, cls):
        self.bqm = cls(np.ones((n, n)), 'BINARY')

    def teardown(self, n, cls):
        del self.bqm

    def time_iter_neighborhood(self, n, cls):
        bqm = self.bqm
        for v in range(n):
            for _ in bqm.iter_neighborhood(v):
                pass


class TimeIterQuadratic:
    params = ([10, 150, 250, 500],
              [dimod.AdjVectorBQM, dimod.DictBQM, dimod.Float32BQM, dimod.Float64BQM])

    def setup(self, n, cls):
        self.bqm = cls(np.ones((n, n)), 'BINARY')

    def teardown(self, n, cls):
        del self.bqm

    def time_iter_quadratic(self, n, cls):
        for _ in self.bqm.iter_quadratic():
            pass
