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
import unittest

import numpy as np

import dimod


class TestSampleSet(unittest.TestCase):
    def test_from_samples(self):

        nv = 5
        sample_sets = [dimod.SampleSet.from_samples(np.ones((nv, nv), dtype='int8'), dimod.SPIN, energy=np.ones(nv)),
                       dimod.SampleSet.from_samples([[1]*nv for _ in range(nv)], dimod.SPIN, energy=np.ones(nv)),
                       dimod.SampleSet.from_samples([{v: 1 for v in range(nv)} for _ in range(nv)], dimod.SPIN,
                                                    energy=np.ones(nv)),
                       dimod.SampleSet.from_samples((np.ones((nv, nv), dtype='int8'), list(range(nv))), dimod.SPIN,
                                                    energy=np.ones(nv)),
                       ]

        # all should be the same
        self.assertEqual(sample_sets[1:], sample_sets[:-1])

    def test_from_samples_str_labels(self):
        nv = 5
        alpha = 'abcde'
        sample_sets = [dimod.SampleSet.from_samples([{v: 1 for v in alpha} for _ in range(nv)], dimod.SPIN,
                                                    energy=np.ones(nv)),
                       dimod.SampleSet.from_samples((np.ones((nv, nv), dtype='int8'), alpha), dimod.SPIN,
                                                    energy=np.ones(nv)),
                       ]

        # all should be the same
        self.assertEqual(sample_sets[1:], sample_sets[:-1])

    # def test_from_samples_single_sample(self):
    #     samples = dimod.SampleSet.from_samples([-1, 1, -1], dimod.SPIN, energy=.5)

    #     print(samples)
