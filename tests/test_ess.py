# Copyright 2026 D-Wave Quantum Inc.
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

"""Test effective sample size estimation."""
import unittest
from math import isnan

import numpy as np

from dimod.ess import _estimate_replicated_lugsail_batch_means
from dimod.ess import estimate_effective_sample_size as estimate_ess


class TestEffectiveSampleSize(unittest.TestCase):

    def test_estimate_ess(self):
        with self.subTest("ESS estimate should be undefined for constant input"):
            self.assertTrue(isnan(estimate_ess(np.ones((100, 1000)))))

        with self.subTest("Null batch size should raise an error."):
            self.assertRaisesRegex(ValueError, "Batch size should be at least three",
                                   estimate_ess, np.ones((100, 1000)), 0)

        with self.subTest("Batch size larger than chain length should raise an error."):
            self.assertRaisesRegex(ValueError, "Batch size should be no more than ``n``",
                                   estimate_ess, np.ones((100, 1000)), 1001)

        with self.subTest("Inputs that are not 2D should raise an error."):
            self.assertRaisesRegex(ValueError, "The input matrix ``x`` should have shape",
                                   estimate_ess, np.ones((123, 100, 1000)), 234)

        with self.subTest("Single-batch estimates are incorrect."):
            x = np.array([[0, 1, 2],
                          [0, 2, 4]])
            s_squared = (np.var([0, 1, 2], ddof=1) + np.var([0, 2, 4], ddof=1))/2
            tau_squared = _estimate_replicated_lugsail_batch_means(x, 3)
            sigma_squared = 2/3*s_squared + tau_squared/3
            answer = 2*3*sigma_squared/tau_squared
            self.assertAlmostEqual(answer, estimate_ess(x, 3))

        with self.subTest("Two-batch estimatse are incorrect."):
            x = np.array([[999, 0, 1, 2, 3, 6, 7],
                          [999, 0, 2, 4, 4, 6, 7]])
            s_squared = (np.var([999, 0, 1, 2, 3, 6, 7], ddof=1)
                         + np.var([999, 0, 2, 4, 4, 6, 7], ddof=1))/2
            tau_squared = _estimate_replicated_lugsail_batch_means(x, 3)
            sigma_squared = 6/7*s_squared + tau_squared/7
            answer = 2*7*sigma_squared/tau_squared
            self.assertAlmostEqual(answer, estimate_ess(x, 3))
