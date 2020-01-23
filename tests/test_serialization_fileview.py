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
import io
import unittest

import numpy as np

import dimod

from dimod.serialization.fileview import FileView, load

# use relative import to handle running tests from different places
from .test_bqm import BQMTestCase, multitest


class TestFileView(BQMTestCase):

    @multitest
    def test_readall_dense(self, BQM):
        # construct the bytes by hand
        if issubclass(BQM, dimod.AdjDictBQM):
            # not (yet) implemented for non cybqms
            return

        arr = np.triu(np.arange(4).reshape((2, 2)))

        bqm = BQM(arr, 'BINARY')
        bqm.offset = 17

        fv = FileView(bqm)

        b = fv.readall()

        # offset
        offset_bytes = bqm.dtype.type(17).tobytes()

        self.assertEqual(b[fv.offset_start:fv.offset_end], offset_bytes)

        # linear
        ltype = np.dtype([('n', bqm.ntype), ('b', bqm.dtype)], align=False)
        ldata = np.empty(2, dtype=ltype)
        ldata['n'][0] = 0
        ldata['n'][1] = 1
        ldata['b'][0] = 0
        ldata['b'][1] = 3
        linear_bytes = ldata.tobytes()

        self.assertEqual(b[fv.linear_start:fv.linear_end], linear_bytes)

        # quadratic
        qtype = np.dtype([('v', bqm.itype), ('b', bqm.dtype)], align=False)
        qdata = np.empty(2, dtype=qtype)
        qdata['v'][0] = 1
        qdata['v'][1] = 0
        qdata['b'][0] = 1
        qdata['b'][1] = 1
        quadratic_bytes = qdata.tobytes()

        self.assertEqual(b[fv.quadratic_start:fv.quadratic_end], quadratic_bytes)

        # and finally check everything
        self.assertEqual(fv.header + offset_bytes + linear_bytes + quadratic_bytes, b)

    # @multitest
    # def test_functional(self, BQM):
    #     bqm = BQM(np.triu(np.arange(25).reshape((5, 5))), 'SPIN')
    #     bqm.offset = -1

    #     with FileView(bqm) as fp:
    #         new = load(fp)

    #     self.assertIs(type(new), type(bqm))
    #     self.assertEqual(bqm, new)

    # def test_functional_empty(self):
    #     bqm = self.BQM('SPIN')

    #     with FileView(bqm) as fp:
    #         new = load(fp)

    #     self.assertIs(type(new), type(bqm))
    #     self.assertEqual(bqm, new)

    # def test_functional_labelled(self):
    #     bqm = self.BQM({'a': -1}, {'ab': 1}, 7, 'SPIN')

    #     with FileView(bqm) as fp:
    #         new = load(fp)

    #     self.assertIs(type(new), type(bqm))
    #     self.assertEqual(bqm, new)

    # def test_functional_labelled_shapeable(self):
    #     if not self.BQM.shapeable():
    #         raise unittest.SkipTest("test only applies to shapeable bqms")

    #     bqm = self.BQM({'a': -1}, {'ab': 1}, 7, 'SPIN')
    #     bqm.add_variable()

    #     with FileView(bqm) as fp:
    #         new = load(fp)

    #     self.assertIs(type(new), type(bqm))
    #     self.assertEqual(bqm, new)

    @multitest
    def test_readinto(self, BQM):
        if issubclass(BQM, dimod.AdjDictBQM):
            # not (yet) implemented for non cybqms
            return

        bqm = BQM(np.triu(np.arange(25).reshape((5, 5))), 'BINARY')
        bqm.offset = 14

        fv = FileView(bqm)

        num_bytes = fv.quadratic_end

        # make a buffer that's a bit bigger
        buff = bytearray(num_bytes + 10)

        self.assertEqual(fv.readinto(buff), num_bytes)

        # reset and make sure that a total read it the same
        fv.seek(0)
        self.assertEqual(fv.read(), buff[:num_bytes])

    @multitest
    def test_readinto_partial_back_to_front(self, BQM):
        if issubclass(BQM, dimod.AdjDictBQM):
            # not (yet) implemented for non cybqms
            return

        bqm = BQM(np.triu(np.arange(25).reshape((5, 5))), 'BINARY')
        bqm.offset = 14

        fv = FileView(bqm)

        buff = fv.readall()

        for pos in range(1, fv.quadratic_end):
            fv.seek(-pos, io.SEEK_END)

            subbuff = bytearray(pos)  # length pos

            self.assertEqual(fv.readinto(subbuff), len(subbuff))
            self.assertEqual(buff[-pos:], subbuff)

    @multitest
    def test_readinto_partial_front_to_back(self, BQM):
        if issubclass(BQM, dimod.AdjDictBQM):
            # not (yet) implemented for non cybqms
            return

        bqm = BQM(np.triu(np.arange(9).reshape((3, 3))), 'BINARY')
        bqm.offset = 14

        fv = FileView(bqm)

        buff = fv.readall()

        for nb in range(fv.quadratic_end):
            self.assertEqual(fv.seek(0), 0)
            subbuff = bytearray(nb)
            num_read = fv.readinto(subbuff)
            self.assertEqual(num_read, nb)
            self.assertEqual(subbuff[:num_read], buff[:num_read])

    @multitest
    def test_readinto_partial_sliding1(self, BQM):
        if issubclass(BQM, dimod.AdjDictBQM):
            # not (yet) implemented for non cybqms
            return

        bqm = BQM(np.tril(np.arange(25).reshape((5, 5))), 'BINARY')
        bqm.offset = -6

        fv = FileView(bqm)

        buff = fv.readall()

        for pos in range(fv.quadratic_end):
            self.assertEqual(pos, fv.seek(pos))
            subbuff = bytearray(1)
            num_read = fv.readinto(subbuff)
            self.assertEqual(num_read, 1)
            self.assertEqual(subbuff, buff[pos:pos+num_read])

    @multitest
    def test_readinto_partial_sliding17(self, BQM):
        if issubclass(BQM, dimod.AdjDictBQM):
            # not (yet) implemented for non cybqms
            return

        bqm = BQM(np.tril(np.arange(25).reshape((5, 5))), 'BINARY')
        bqm.offset = -6

        fv = FileView(bqm)

        buff = fv.readall()

        for pos in range(fv.quadratic_end):
            self.assertEqual(pos, fv.seek(pos))
            subbuff = bytearray(17)
            num_read = fv.readinto(subbuff)
            self.assertGreater(num_read, 0)
            self.assertEqual(subbuff[:num_read], buff[pos:pos+num_read])
