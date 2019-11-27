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
import tempfile
import sys
import unittest

import numpy as np

import dimod
from dimod.serialization.fileview import FileView, load


class TestFileViewBase:
    BQM = None   # this will be overwritten

    def test_functional(self):
        bqm = self.BQM(np.triu(np.arange(25).reshape((5, 5))), 'SPIN')
        bqm.offset = -1

        with FileView(bqm) as fp:
            new = load(fp)

        self.assertIs(type(new), type(bqm))
        self.assertEqual(bqm, new)

    def test_functional_empty(self):
        bqm = self.BQM('SPIN')

        with FileView(bqm) as fp:
            new = load(fp)

        self.assertIs(type(new), type(bqm))
        self.assertEqual(bqm, new)

    def test_functional_labelled(self):
        bqm = self.BQM(({'a': -1}, {'ab': 1}, 7), 'SPIN')

        with FileView(bqm) as fp:
            new = load(fp)

        self.assertIs(type(new), type(bqm))
        self.assertEqual(bqm, new)

    def test_functional_labelled_shapeable(self):
        if not self.BQM.shapeable():
            raise unittest.SkipTest("test only applies to shapeable bqms")

        bqm = self.BQM(({'a': -1}, {'ab': 1}, 7), 'SPIN')
        bqm.add_variable()

        with FileView(bqm) as fp:
            new = load(fp)

        self.assertIs(type(new), type(bqm))
        self.assertEqual(bqm, new)

    def test_readinto_linear_partial(self):
        bqm = self.BQM(np.triu(np.arange(25).reshape((5, 5))), 'SPIN')

        # the number of bytes we expect to get
        num_bytes = len(bqm)*(bqm.ntype.itemsize+bqm.dtype.itemsize)

        # make a buffer that's a bit bigger
        buff = bytearray(num_bytes+10)

        # we expect that the buffer is filled to num_bytes
        self.assertEqual(bqm.readinto_linear(buff), num_bytes)

        # check that all the sub-bytearrays starting from front match
        for nb in range(1, num_bytes):
            subbuff = bytearray(nb)
            num_read = bqm.readinto_linear(subbuff)
            self.assertGreater(num_read, 0)
            self.assertEqual(subbuff[:num_read], buff[:num_read])

        # and starting from the back
        for pos in range(num_bytes):
            subbuff = bytearray(num_bytes)
            num_read = bqm.readinto_linear(subbuff, pos=pos)
            self.assertGreater(num_read, 0)
            self.assertEqual(subbuff[:num_read], buff[pos:pos+num_read])

        # sliding 1
        for pos in range(num_bytes):
            subbuff = bytearray(1)
            num_read = bqm.readinto_linear(subbuff, pos=pos)
            self.assertEqual(num_read, 1)
            self.assertEqual(subbuff, buff[pos:pos+num_read])

        # sliding 7
        for pos in range(num_bytes):
            subbuff = bytearray(7)
            num_read = bqm.readinto_linear(subbuff, pos=pos)
            self.assertGreater(num_read, 0)
            self.assertEqual(subbuff[:num_read], buff[pos:pos+num_read])

        # sliding 17
        for pos in range(num_bytes):
            subbuff = bytearray(17)
            num_read = bqm.readinto_linear(subbuff, pos=pos)
            self.assertGreater(num_read, 0)
            self.assertEqual(subbuff[:num_read], buff[pos:pos+num_read])

    def test_readinto_quadratic_partial(self):
        bqm = self.BQM(np.triu(np.arange(25).reshape((5, 5))), 'SPIN')

        # the number of bytes we expect to get
        num_bytes = 2*bqm.num_interactions*(bqm.itype.itemsize+bqm.dtype.itemsize)

        # make a buffer that's a bit bigger
        buff = bytearray(num_bytes+10)

        # we expect that the buffer is filled to num_bytes
        self.assertEqual(bqm.readinto_quadratic(buff), num_bytes)

        # check that all the sub-bytearrays starting from front match
        for nb in range(1, num_bytes):
            subbuff = bytearray(nb)
            num_read = bqm.readinto_quadratic(subbuff)
            self.assertGreater(num_read, 0)
            self.assertEqual(subbuff[:num_read], buff[:num_read])

        # and starting from the back
        for pos in range(num_bytes):
            subbuff = bytearray(num_bytes)
            num_read = bqm.readinto_quadratic(subbuff, pos=pos)
            self.assertGreater(num_read, 0)
            self.assertEqual(subbuff[:num_read], buff[pos:pos+num_read])

        # sliding 1
        for pos in range(num_bytes):
            subbuff = bytearray(1)
            num_read = bqm.readinto_quadratic(subbuff, pos=pos)
            self.assertEqual(num_read, 1)
            self.assertEqual(subbuff, buff[pos:pos+num_read])

        # sliding 7
        for pos in range(num_bytes):
            subbuff = bytearray(7)
            num_read = bqm.readinto_quadratic(subbuff, pos=pos)
            self.assertGreater(num_read, 0)
            self.assertEqual(subbuff[:num_read], buff[pos:pos+num_read])

        # sliding 17
        for pos in range(num_bytes):
            subbuff = bytearray(17)
            num_read = bqm.readinto_quadratic(subbuff, pos=pos)
            self.assertGreater(num_read, 0)
            self.assertEqual(subbuff[:num_read], buff[pos:pos+num_read])


class TestFileViewAdjArrayBQM(TestFileViewBase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.version_info.major == 2 or sys.version_info.minor < 5:
            raise unittest.SkipTest("Not supported in Python <= 3.5")

        from dimod.bqm import AdjArrayBQM

        cls.BQM = AdjArrayBQM


# class TestFileViewAdjDictBQM(TestFileViewBase, unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         from dimod.bqm import AdjDictBQM

#         cls.BQM = AdjDictBQM


# class TestFileViewAdjMapBQM(TestFileViewBase, unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         if sys.version_info.major == 2 or sys.version_info.minor < 5:
#             raise unittest.SkipTest("Not supported in Python <= 3.5")

#         from dimod.bqm import AdjMapBQM

#         cls.BQM = AdjMapBQM


# class TestFileViewAdjVectorBQM(TestFileViewBase, unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         if sys.version_info.major == 2 or sys.version_info.minor < 5:
#             raise unittest.SkipTest("Not supported in Python <= 3.5")

#         from dimod.bqm import AdjVectorBQM

#         cls.BQM = AdjVectorBQM
