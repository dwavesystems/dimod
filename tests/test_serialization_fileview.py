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

import io
import os
import sys
import unittest

import numpy as np

from parameterized import parameterized, parameterized_class

import dimod

from dimod.binary import BinaryQuadraticModel, DictBQM, Float32BQM, Float64BQM
from dimod.serialization.fileview import FileView, load, register

SUPPORTED_VERSIONS = [(1, 0), (2, 0)]

BQMs = dict(BinaryQuadraticModel=BinaryQuadraticModel,
            DictBQM=DictBQM,
            Float32BQM=Float32BQM,
            Float64BQM=Float64BQM,
            )


BQM_x_VERSION = [('v{}.{}_{}'.format(version[0], version[1], cls.__name__),
                  cls, version)
                 for cls in BQMs.values()
                 for version in SUPPORTED_VERSIONS]

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'fileview')


class TestFileView(unittest.TestCase):
    @unittest.skipUnless(sys.maxsize > 2**32, "test is for 64 bit")
    def test_saved_adjvector_5x5_v1(self):
        bqm = BinaryQuadraticModel(
            np.triu(np.arange(25).reshape((5, 5))), 'BINARY')

        filename = os.path.join(TEST_DATA_DIR, '5x5_v1.bqm')

        # with open(filename, 'wb') as fp:
        #     with FileView(bqm, version=1) as fv:
        #         fp.write(fv.read())

        with open(filename, 'rb') as fp:
            buff = fp.read()

        # and that loading gives the same
        new = load(buff)
        self.assertEqual(new, bqm)


class TestFunctional(unittest.TestCase):
    @parameterized.expand(BQM_x_VERSION)
    def test_empty(self, name, BQM, version):
        bqm = BQM('SPIN')
        with self.assertWarns(DeprecationWarning):
            with FileView(bqm, version=version) as fp:
                new = load(fp)

        self.assertEqual(bqm, new)
        if bqm.dtype != np.dtype('O'):
            self.assertEqual(bqm.dtype, new.dtype)

    @parameterized.expand(BQM_x_VERSION)
    def test_empty_bytes(self, name, BQM, version):
        bqm = BQM('SPIN')
        with self.assertWarns(DeprecationWarning):
            with FileView(bqm, version=version) as fp:
                new = load(fp.read())

        self.assertEqual(bqm, new)
        if bqm.dtype != np.dtype('O'):
            self.assertEqual(bqm.dtype, new.dtype)

    @parameterized.expand(BQM_x_VERSION)
    def test_ignore_labels(self, name, BQM, version):
        bqm = BQM(np.triu(np.arange(25).reshape((5, 5))), 'SPIN')

        labeled_bqm = bqm.relabel_variables(dict(enumerate('abcde')),
                                            inplace=False)

        with self.assertWarns(DeprecationWarning):
            with FileView(labeled_bqm, version=version, ignore_labels=True) as fv:
                new = load(fv)

        self.assertEqual(new, bqm)
        if bqm.dtype != np.dtype('O'):
            self.assertEqual(bqm.dtype, new.dtype)

    @parameterized.expand(BQM_x_VERSION)
    def test_labelled(self, name, BQM, version):
        bqm = BQM({'a': -1}, {'ab': 1}, 7, 'SPIN')

        with self.assertWarns(DeprecationWarning):
            with FileView(bqm, version=version) as fp:
                new = load(fp)

        self.assertEqual(bqm, new)
        if bqm.dtype != np.dtype('O'):
            self.assertEqual(bqm.dtype, new.dtype)

    @parameterized.expand(BQM_x_VERSION)
    def test_labelled_bytes(self, name, BQM, version):
        bqm = BQM({'a': -1}, {'ab': 1}, 7, 'SPIN')

        with self.assertWarns(DeprecationWarning):
            with FileView(bqm, version=version) as fp:
                new = load(fp.read())

        if bqm.dtype != np.dtype('O'):
            self.assertEqual(bqm.dtype, new.dtype)
        self.assertEqual(bqm, new)

    @parameterized.expand(BQM_x_VERSION)
    def test_labelled_shapeable(self, name, BQM, version):
        bqm = BQM({'a': -1}, {'ab': 1}, 7, 'SPIN')
        bqm.add_variable()

        with self.assertWarns(DeprecationWarning):
            with FileView(bqm, version=version) as fp:
                new = load(fp)

        if bqm.dtype != np.dtype('O'):
            self.assertEqual(bqm.dtype, new.dtype)
        self.assertEqual(bqm, new)

    @parameterized.expand(BQM_x_VERSION)
    def test_labelled_shapeable_bytes(self, name, BQM, version):
        bqm = BQM({'a': -1}, {'ab': 1}, 7, 'SPIN')
        bqm.add_variable()

        with self.assertWarns(DeprecationWarning):
            with FileView(bqm, version=version) as fp:
                new = load(fp.read())

        if bqm.dtype != np.dtype('O'):
            self.assertEqual(bqm.dtype, new.dtype)
        self.assertEqual(bqm, new)

    @parameterized.expand(BQM_x_VERSION)
    def test_typical(self, name, BQM, version):
        bqm = BQM(np.triu(np.arange(25).reshape((5, 5))), 'SPIN')
        bqm.offset = -1

        with self.assertWarns(DeprecationWarning):
            with FileView(bqm, version=version) as fp:
                new = load(fp)

        if bqm.dtype != np.dtype('O'):
            self.assertEqual(bqm.dtype, new.dtype)
        self.assertEqual(bqm, new)

    @parameterized.expand(BQM_x_VERSION)
    def test_unhashable_variables(self, name, BQM, version):
        bqm = BQM({(0, 1): 1}, {}, 'SPIN')

        with self.assertWarns(DeprecationWarning):
            with FileView(bqm, version=version) as fv:
                new = load(fv)

        self.assertEqual(new, bqm)


class TestLoad(unittest.TestCase):
    def test_bqm(self):
        bqm = BinaryQuadraticModel({'a': -1}, {'ab': 1}, 7, 'SPIN')
        with bqm.to_file() as f:
            self.assertEqual(bqm, load(f))

    def test_cqm(self):
        cqm = dimod.CQM()

        bqm = BinaryQuadraticModel({'a': -1}, {'ab': 1}, 1.5, 'SPIN')
        cqm.add_constraint(bqm, '<=')
        cqm.add_constraint(bqm, '>=')  # add it again

        with cqm.to_file() as f:
            new = load(f)

        self.assertTrue(new.objective.variables >= cqm.objective.variables)
        for v, bias in cqm.objective.iter_linear():
            self.assertEqual(new.objective.get_linear(v), bias)
        for u, v, bias in cqm.objective.iter_quadratic():
            self.assertEqual(new.objective.get_quadratic(u, v), bias)
        self.assertEqual(new.objective.offset, cqm.objective.offset)

        self.assertEqual(set(cqm.constraints), set(new.constraints))
        for label, constraint in cqm.constraints.items():
            self.assertTrue(constraint.lhs.is_equal(new.constraints[label].lhs))
            self.assertEqual(constraint.rhs, new.constraints[label].rhs)
            self.assertEqual(constraint.sense, new.constraints[label].sense)

    def test_dqm(self):
        dqm = dimod.DiscreteQuadraticModel()
        dqm.add_variable(5, 'a')
        dqm.add_variable(6, 'b')
        dqm.set_quadratic_case('a', 0, 'b', 5, 1.5)

        with dqm.to_file() as f:
            new = load(f)

        self.assertEqual(dqm.num_variables(), new.num_variables())
        self.assertEqual(dqm.num_cases(), new.num_cases())
        self.assertEqual(dqm.get_quadratic_case('a', 0, 'b', 5),
                         new.get_quadratic_case('a', 0, 'b', 5))

    def test_exception_propagation(self):
        # register a fake one

        def keyerror(fp):
            raise KeyError("kaboom")

        def valueerror(fp):
            raise ValueError("kaboom2")

        register(b'KEYERROR_TEST', keyerror)
        register(b'VALUERROR_TEST', valueerror)

        with self.assertRaises(KeyError) as err:
            load(b'KEYERROR_TEST SOME DATA')
        self.assertEqual(err.exception.args[0], 'kaboom')

        with self.assertRaises(ValueError) as err:
            load(b'VALUERROR_TEST SOME DATA')
        self.assertEqual(err.exception.args[0], 'kaboom2')
