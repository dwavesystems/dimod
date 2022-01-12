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

import unittest

import dimod
import numpy as np

from dimod.serialization.format import Formatter


class TestUnknownType(unittest.TestCase):
    def test_int(self):
        with self.assertRaises(TypeError):
            Formatter().format(5)


class TestSampleSet(unittest.TestCase):
    def test_empty(self):
        empty = dimod.SampleSet.from_samples([], dimod.BINARY, energy=[])

        s = Formatter(width=80).format(empty)

        target = '\n'.join(['Empty SampleSet',
                            "Record Fields: ['sample', 'energy', 'num_occurrences']",
                            "Variables: []",
                            "['BINARY', 0 rows, 0 samples, 0 variables]"])

        self.assertEqual(s, target)

    def test_empty_width_50(self):
        empty = dimod.SampleSet.from_samples([], dimod.BINARY, energy=[])

        s = Formatter(width=50).format(empty)

        target = '\n'.join(['Empty SampleSet',
                            "Record Fields: ['sample', 'energy', ...]",
                            "Variables: []",
                            "['BINARY', 0 rows, 0 samples, 0 variables]"])

        self.assertEqual(s, target)

    def test_empty_with_variables(self):
        samples = dimod.SampleSet.from_samples(([], 'abcdefghijklmnopqrstuvwxyz'),
                                               dimod.SPIN, energy=[])
        s = Formatter(width=49).format(samples)

        target = '\n'.join(['Empty SampleSet',
                            "Record Fields: ['sample', 'energy', ...]",
                            "Variables: ['a', 'b', 'c', 'd', 'e', 'f', 'g', ...]",
                            "['SPIN', 0 rows, 0 samples, 26 variables]"])

        self.assertEqual(s, target)

    def test_triu_binary(self):
        arr = np.triu(np.ones((5, 5)))
        variables = [0, 1, 'a', 'b', 'c']

        samples = dimod.SampleSet.from_samples((arr, variables),
                                               dimod.BINARY, energy=[4., 3, 2, 1, 0])

        s = Formatter(width=79, depth=None).format(samples)

        target = '\n'.join(["   0  1  a  b  c energy num_oc.",
                            "4  0  0  0  0  1    0.0       1",
                            "3  0  0  0  1  1    1.0       1",
                            "2  0  0  1  1  1    2.0       1",
                            "1  0  1  1  1  1    3.0       1",
                            "0  1  1  1  1  1    4.0       1",
                            "['BINARY', 5 rows, 5 samples, 5 variables]"])

        self.assertEqual(s, target)

    def test_triu_spin(self):
        arr = np.triu(np.ones((5, 5)))
        variables = [0, 1, 'a', 'b', 'c']

        samples = dimod.SampleSet.from_samples((2*arr-1, variables),
                                               dimod.SPIN, energy=[4., 3, 2, 1, 0])

        s = Formatter(width=79, depth=None).format(samples)

        target = '\n'.join(["   0  1  a  b  c energy num_oc.",
                            "4 -1 -1 -1 -1 +1    0.0       1",
                            "3 -1 -1 -1 +1 +1    1.0       1",
                            "2 -1 -1 +1 +1 +1    2.0       1",
                            "1 -1 +1 +1 +1 +1    3.0       1",
                            "0 +1 +1 +1 +1 +1    4.0       1",
                            "['SPIN', 5 rows, 5 samples, 5 variables]"])

        self.assertEqual(s, target)

    def test_triu_row_summation(self):
        arr = np.triu(np.ones((5, 5)))
        variables = [0, 1, 'a', 'b', 'c']

        samples = dimod.SampleSet.from_samples((arr, variables),
                                               dimod.BINARY, energy=[4., 3, 2, 1, 0])

        s = Formatter(width=79, depth=4).format(samples)

        target = '\n'.join(["   0  1  a  b  c energy num_oc.",
                            "4  0  0  0  0  1    0.0       1",
                            "3  0  0  0  1  1    1.0       1",
                            "...",
                            "0  1  1  1  1  1    4.0       1",
                            "['BINARY', 5 rows, 5 samples, 5 variables]"])

        self.assertEqual(s, target)

    def test_triu_col_summation(self):
        arr = np.triu(np.ones((5, 5)))
        variables = [0, 1, 'a', 'b', 'c']

        samples = dimod.SampleSet.from_samples((arr, variables),
                                               dimod.BINARY, energy=[4., 3, 2, 1, 0])

        s = Formatter(width=30, depth=None).format(samples)

        # without summation length would be 31

        target = '\n'.join(["   0  1 ...  c energy num_oc.",
                            "4  0  0 ...  1    0.0       1",
                            "3  0  0 ...  1    1.0       1",
                            "2  0  0 ...  1    2.0       1",
                            "1  0  1 ...  1    3.0       1",
                            "0  1  1 ...  1    4.0       1",
                            "['BINARY',",
                            " 5 rows,",
                            " 5 samples,",
                            " 5 variables]"])

        self.assertEqual(s, target)

    def test_additional_fields_summation(self):
        arr = np.ones((2, 5))
        variables = list(range(5))

        samples = dimod.SampleSet.from_samples((arr, variables),
                                               dimod.BINARY, energy=1,
                                               other=[5, 6],
                                               anotherother=[234029348023948234, 3])
        s = Formatter(width=30, depth=None).format(samples)

        target = '\n'.join(["   0 ...  4 energy num_oc. ...",
                            "0  1 ...  1      1       1 ...",
                            "1  1 ...  1      1       1 ...",
                            "['BINARY',",
                            " 2 rows,",
                            " 2 samples,",
                            " 5 variables]"])

        self.assertEqual(target, s)

    def test_additional_fields(self):
        arr = np.ones((2, 5))
        variables = list(range(5))

        samples = dimod.SampleSet.from_samples((arr, variables),
                                               dimod.BINARY, energy=1,
                                               other=[5, 6],
                                               anotherother=[234029348023948234, object()])
        s = Formatter(width=79, depth=None).format(samples)

        target = '\n'.join(["   0  1  2  3  4 energy num_oc. anothe. other",
                            "0  1  1  1  1  1      1       1 2340...     5",
                            "1  1  1  1  1  1      1       1 <obj...     6",
                            "['BINARY', 2 rows, 2 samples, 5 variables]"])

        self.assertEqual(target, s)

    def test_discrete(self):
        ss = dimod.SampleSet.from_samples(([[0, 17, 236], [3, 321, 1]], 'abc'),
                                          'INTEGER', energy=[1, 2])
        s = Formatter(width=79, depth=None).format(ss)
        target = '\n'.join(["  a   b   c energy num_oc.",
                            "0 0  17 236      1       1",
                            "1 3 321   1      2       1",
                            "['INTEGER', 2 rows, 2 samples, 3 variables]"])

        self.assertEqual(target, s)

    def test_depth(self):
        ss = dimod.SampleSet.from_samples(([[0, 17, 236],
                                            [3, 321, 1],
                                            [4444444444, 312, 1],
                                            [4, 3, 3]], 'abc'),
                                          'INTEGER', energy=[1, 2, 3, 4])
        s = Formatter(width=79, depth=2).format(ss)
        target = '\n'.join(["  a   b   c energy num_oc.",
                            "0 0  17 236      1       1",
                            "...",
                            "3 4   3   3      4       1",
                            "['INTEGER', 4 rows, 4 samples, 3 variables]"])

        self.assertEqual(target, s)

    def test_misalignment(self):
        ss = dimod.SampleSet.from_samples([[0, 1], [0, 1], [1, 0]], 'BINARY',
                                          energy=[-1, 1.55, 2])

        s = Formatter().format(ss)

        target = ("   0  1 energy num_oc.\n"
                  "0  0  1   -1.0       1\n"
                  "1  0  1   1.55       1\n"
                  "2  1  0    2.0       1\n"
                  "['BINARY', 3 rows, 3 samples, 2 variables]")

        self.assertEqual(target, s)

    def test_wide(self):
        ss = dimod.SampleSet.from_samples(np.ones(1000), 'BINARY', energy=[0])

        s = Formatter().format(ss)

        self.assertEqual(s, ("   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15"
                             " 16 17 ... 999 energy num_oc.\n0  1  1  1  1  1  "
                             "1  1  1  1  1  1  1  1  1  1  1  1  1 ...   1    "
                             "  0       1\n['BINARY', 1 rows, 1 samples, 1000 v"
                             "ariables]"))
