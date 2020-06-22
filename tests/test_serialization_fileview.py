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
import os
import sys
import unittest

import numpy as np

from parameterized import parameterized, parameterized_class

import dimod

from dimod.serialization.fileview import FileView, load, SUPPORTED_VERSIONS

BQM_SUBCLASSES = [dimod.AdjArrayBQM,
                  # dimod.AdjDictBQM,  # not supported yet
                  dimod.AdjMapBQM,
                  dimod.AdjVectorBQM,
                  # dimod.BinaryQuadraticModel,  # not supported yet
                  ]


BQM_x_VERSION = [('v{}.{}_{}'.format(version[0], version[1], cls.__name__),
                  cls, version)
                 for cls in BQM_SUBCLASSES
                 for version in SUPPORTED_VERSIONS]

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'fileview')


class TestFileView(unittest.TestCase):
    @parameterized.expand([(cls.__name__, cls) for cls in BQM_SUBCLASSES])
    def test_dense_v1(self, name, BQM):
        arr = np.triu(np.arange(4).reshape((2, 2)))

        bqm = BQM(arr, 'BINARY')
        bqm.offset = 17

        fv = FileView(bqm, version=1)

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

    @parameterized.expand(BQM_x_VERSION)
    def test_readinto(self, name, BQM, version):

        bqm = BQM(np.triu(np.arange(25).reshape((5, 5))), 'BINARY')
        bqm.offset = 14

        fv = FileView(bqm, version=version)

        num_bytes = fv.quadratic_end

        # make a buffer that's a bit bigger
        buff = bytearray(num_bytes + 10)

        self.assertEqual(fv.readinto(buff), num_bytes)

        # reset and make sure that a total read it the same
        fv.seek(0)
        self.assertEqual(fv.read(), buff[:num_bytes])

    @parameterized.expand(BQM_x_VERSION)
    def test_readinto_partial_back_to_front(self, name, BQM, version):

        bqm = BQM(np.triu(np.arange(25).reshape((5, 5))), 'BINARY')
        bqm.offset = 14

        fv = FileView(bqm, version=version)

        buff = fv.readall()

        for pos in range(1, fv.quadratic_end):
            fv.seek(-pos, io.SEEK_END)

            subbuff = bytearray(pos)  # length pos

            self.assertEqual(fv.readinto(subbuff), len(subbuff))
            self.assertEqual(buff[-pos:], subbuff)

    @parameterized.expand(BQM_x_VERSION)
    def test_readinto_partial_front_to_back(self, name, BQM, version):

        bqm = BQM(np.triu(np.arange(9).reshape((3, 3))), 'BINARY')
        bqm.offset = 14

        fv = FileView(bqm, version=version)

        buff = fv.readall()

        for nb in range(fv.quadratic_end):
            self.assertEqual(fv.seek(0), 0)
            subbuff = bytearray(nb)
            num_read = fv.readinto(subbuff)
            self.assertEqual(num_read, nb)
            self.assertEqual(subbuff[:num_read], buff[:num_read])

    @parameterized.expand(BQM_x_VERSION)
    def test_readinto_partial_sliding1(self, name, BQM, version):

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

    @parameterized.expand(BQM_x_VERSION)
    def test_readinto_partial_sliding17(self, name, BQM, version):

        bqm = BQM(np.tril(np.arange(25).reshape((5, 5))), 'BINARY')
        bqm.offset = -6

        fv = FileView(bqm, version=version)

        buff = fv.readall()

        for pos in range(fv.quadratic_end):
            self.assertEqual(pos, fv.seek(pos))
            subbuff = bytearray(17)
            num_read = fv.readinto(subbuff)
            self.assertGreater(num_read, 0)
            self.assertEqual(subbuff[:num_read], buff[pos:pos+num_read])

    # Ocean only supports 64bit python
    @unittest.skipUnless(sys.maxsize > 2**32, "test is for 64 bit")
    def test_saved_adjvector_5x5_v1(self):
        bqm = dimod.AdjVectorBQM(np.triu(np.arange(25).reshape((5, 5))),
                                 'BINARY')

        filename = os.path.join(TEST_DATA_DIR, '5x5_v1.bqm')

        # with open(filename, 'wb') as fp:
        #     with FileView(bqm, version=1) as fv:
        #         fp.write(fv.readall())

        with open(filename, 'rb') as fp:
            buff = fp.read()

        # test that we still encode the same way
        with FileView(bqm, version=1) as fv:
            self.assertEqual(buff, fv.read())

        # and that loading gives the same
        new = load(buff)
        self.assertEqual(new, bqm)


class TestFunctional(unittest.TestCase):
    @parameterized.expand(BQM_x_VERSION)
    def test_empty(self, name, BQM, version):
        bqm = BQM('SPIN')
        with FileView(bqm, version=version) as fp:
            new = load(fp)

        self.assertIs(type(new), type(bqm))
        self.assertEqual(bqm, new)

    @parameterized.expand(BQM_x_VERSION)
    def test_labelled(self, name, BQM, version):

        bqm = BQM({'a': -1}, {'ab': 1}, 7, 'SPIN')

        with FileView(bqm, version=version) as fp:
            new = load(fp)

        self.assertIs(type(new), type(bqm))
        self.assertEqual(bqm, new)

    @parameterized.expand(BQM_x_VERSION)
    def test_labelled_shapeable(self, name, BQM, version):
        if not BQM.shapeable():
            raise unittest.SkipTest("test only applies to shapeable bqms")

        bqm = BQM({'a': -1}, {'ab': 1}, 7, 'SPIN')
        bqm.add_variable()

        with FileView(bqm, version=version) as fp:
            new = load(fp)

        self.assertIs(type(new), type(bqm))
        self.assertEqual(bqm, new)

    @parameterized.expand(BQM_x_VERSION)
    def test_typical(self, name, BQM, version):
        bqm = BQM(np.triu(np.arange(25).reshape((5, 5))), 'SPIN')
        bqm.offset = -1

        with FileView(bqm, version=version) as fp:
            new = load(fp)

        self.assertIs(type(new), type(bqm))
        self.assertEqual(bqm, new)

    @parameterized.expand(BQM_x_VERSION)
    def test_unhashable_variables(self, name, BQM, version):
        bqm = BQM({(0, 1): 1}, {}, 'SPIN')

        with FileView(bqm, version=version) as fv:
            new = load(fv)

        self.assertEqual(new, bqm)
