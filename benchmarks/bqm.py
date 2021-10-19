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

import dimod
import numpy as np

from parameterized import parameterized


BQMs = dict(bqm_k150=dimod.BinaryQuadraticModel(np.ones((150, 150)), 'BINARY'),
            adj_k150=dimod.AdjVectorBQM(np.ones((150, 150)), 'BINARY'),
            bqm_k500=dimod.BinaryQuadraticModel(np.ones((500, 500)), 'BINARY'),
            adj_k500=dimod.AdjVectorBQM(np.ones((500, 500)), 'BINARY'),
            )


class BQMSuite:
    @parameterized.expand(BQMs.items())
    def time_iter_quadratic(self, _, bqm):
        for _ in bqm.iter_quadratic():
            pass

    @parameterized.expand((name, bqm, np.ones((1000, bqm.num_variables)))
                          for name, bqm in BQMs.items())
    def time_energies(self, _, bqm, samples):
        bqm.energies(samples)
