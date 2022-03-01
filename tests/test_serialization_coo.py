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

import unittest
import tempfile
import shutil
import os

import dimod
import dimod.serialization.coo as coo


class TestCOO(unittest.TestCase):
    def test_dumps_empty_BINARY(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
        s = coo.dumps(bqm)
        self.assertEqual(s, '')

    def test_dumps_empty_SPIN(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
        s = coo.dumps(bqm)
        self.assertEqual(s, '')

    def test_dumps_sortable_SPIN(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({0: 1.}, {(0, 1): 2, (2, 3): .4})
        s = coo.dumps(bqm)
        contents = "0 0 1.000000\n0 1 2.000000\n2 3 0.400000"
        self.assertEqual(s, contents)

    def test_dumps_sortable_SPIN_with_header(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({0: 1.}, {(0, 1): 2, (2, 3): .4})
        s = coo.dumps(bqm, vartype_header=True)
        contents = "# vartype=SPIN\n0 0 1.000000\n0 1 2.000000\n2 3 0.400000"
        self.assertEqual(s, contents)

    def test_dumps_sortable_BINARY_with_header(self):
        bqm = dimod.BinaryQuadraticModel.from_qubo({(0, 0): 1., (0, 1): 2, (2, 3): .4})
        s = coo.dumps(bqm, vartype_header=True)
        contents = "# vartype=BINARY\n0 0 1.000000\n0 1 2.000000\n2 3 0.400000"
        self.assertEqual(s, contents)

    def test_load(self):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'coo_qubo.qubo')

        with open(filepath, 'r') as fp:
            bqm = coo.load(fp, vartype=dimod.BINARY)

        self.assertEqual(bqm, dimod.BinaryQuadraticModel.from_qubo({(0, 0): -1, (1, 1): -1, (2, 2): -1, (3, 3): -1}))

    def test_loads(self):
        contents = "0 0 1.000000\n0 1 2.000000\n2 3 0.400000"
        bqm = coo.loads(contents, vartype=dimod.SPIN)
        self.assertEqual(bqm, dimod.BinaryQuadraticModel.from_ising({0: 1.}, {(0, 1): 2, (2, 3): .4}))

    def test_functional_file_empty_BINARY(self):

        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        tmpdir = tempfile.mkdtemp()
        filename = os.path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            coo.dump(bqm, fp=file)

        with open(filename, 'r') as file:
            new_bqm = coo.load(file, vartype=dimod.BINARY)

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    def test_functional_file_empty_SPIN(self):

        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        tmpdir = tempfile.mkdtemp()
        filename = os.path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            coo.dump(bqm, fp=file)

        with open(filename, 'r') as file:
            new_bqm = coo.load(file, vartype=dimod.SPIN)

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    def test_functional_file_BINARY(self):

        bqm = dimod.BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.BINARY)

        tmpdir = tempfile.mkdtemp()
        filename = os.path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            coo.dump(bqm, fp=file)

        with open(filename, 'r') as file:
            new_bqm = coo.load(file, vartype=dimod.BINARY)

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    def test_functional_file_SPIN(self):

        bqm = dimod.BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.SPIN)

        tmpdir = tempfile.mkdtemp()
        filename = os.path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            coo.dump(bqm, fp=file)

        with open(filename, 'r') as file:
            new_bqm = coo.load(file, vartype=dimod.SPIN)

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    def test_functional_string_empty_BINARY(self):

        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        s = coo.dumps(bqm)
        new_bqm = coo.loads(s, vartype=dimod.BINARY)

        self.assertEqual(bqm, new_bqm)

    def test_functional_string_empty_SPIN(self):

        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        s = coo.dumps(bqm)
        new_bqm = coo.loads(s, vartype=dimod.SPIN)

        self.assertEqual(bqm, new_bqm)

    def test_functional_string_BINARY(self):

        bqm = dimod.BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.BINARY)

        s = coo.dumps(bqm)
        new_bqm = coo.loads(s, vartype=dimod.BINARY)

        self.assertEqual(bqm, new_bqm)

    def test_functional_string_SPIN(self):

        bqm = dimod.BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.SPIN)

        s = coo.dumps(bqm)
        new_bqm = coo.loads(s, vartype=dimod.SPIN)

        self.assertEqual(bqm, new_bqm)

    def test_functional_SPIN_vartypeheader(self):
        bqm = dimod.BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.SPIN)

        s = coo.dumps(bqm, vartype_header=True)
        new_bqm = coo.loads(s)

        self.assertEqual(bqm, new_bqm)

    def test_no_vartype(self):
        bqm = dimod.BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.SPIN)

        s = coo.dumps(bqm)
        with self.assertRaises(ValueError):
            coo.loads(s)

    def test_conflicting_vartype(self):
        bqm = dimod.BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.SPIN)

        s = coo.dumps(bqm, vartype_header=True)
        with self.assertRaises(ValueError):
            coo.loads(s, vartype=dimod.BINARY)

    def test_deprecation_load(self):
        with self.assertWarns(DeprecationWarning):
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'coo_qubo.qubo')

            with open(filepath, 'r') as fp:
                bqm = coo.load(fp, cls=dimod.BinaryQuadraticModel, vartype=dimod.Vartype.SPIN)

    def test_deprecation_loads(self):
        with self.assertWarns(DeprecationWarning):
            contents = "0 0 1.000000\n0 1 2.000000\n2 3 0.400000"

            bqm = coo.loads(contents, cls=dimod.BinaryQuadraticModel, vartype=dimod.Vartype.BINARY)
