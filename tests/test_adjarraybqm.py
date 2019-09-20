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

from dimod.bqm import AdjArrayBQM

from tests.test_bqm import TestBQMAPI


class TestAdjArray(TestBQMAPI, unittest.TestCase):
    BQM = AdjArrayBQM


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
