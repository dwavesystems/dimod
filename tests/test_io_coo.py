import unittest
import tempfile
import shutil
import os

import dimod
import dimod.io.coo as coo


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

    def test_load(self):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'coo_qubo.qubo')

        with open(filepath, 'r') as fp:
            bqm = coo.load(fp, dimod.BinaryQuadraticModel.empty(dimod.BINARY))

        self.assertEqual(bqm, dimod.BinaryQuadraticModel.from_qubo({(0, 0): -1, (1, 1): -1, (2, 2): -1, (3, 3): -1}))

    def test_loads(self):
        contents = "0 0 1.000000\n0 1 2.000000\n2 3 0.400000"
        bqm = coo.loads(contents, dimod.BinaryQuadraticModel.empty(dimod.SPIN))
        self.assertEqual(bqm, dimod.BinaryQuadraticModel.from_ising({0: 1.}, {(0, 1): 2, (2, 3): .4}))

    def test_functional_file_empty_BINARY(self):

        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        tmpdir = tempfile.mkdtemp()
        filename = os.path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            coo.dump(bqm, fp=file)

        with open(filename, 'r') as file:
            new_bqm = coo.load(file, dimod.BinaryQuadraticModel.empty(dimod.BINARY))

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    def test_functional_file_empty_SPIN(self):

        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        tmpdir = tempfile.mkdtemp()
        filename = os.path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            coo.dump(bqm, fp=file)

        with open(filename, 'r') as file:
            new_bqm = coo.load(file, dimod.BinaryQuadraticModel.empty(dimod.SPIN))

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    def test_functional_file_BINARY(self):

        bqm = dimod.BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.BINARY)

        tmpdir = tempfile.mkdtemp()
        filename = os.path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            coo.dump(bqm, fp=file)

        with open(filename, 'r') as file:
            new_bqm = coo.load(file, dimod.BinaryQuadraticModel.empty(dimod.BINARY))

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    def test_functional_file_SPIN(self):

        bqm = dimod.BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.SPIN)

        tmpdir = tempfile.mkdtemp()
        filename = os.path.join(tmpdir, 'test.qubo')

        with open(filename, 'w') as file:
            coo.dump(bqm, fp=file)

        with open(filename, 'r') as file:
            new_bqm = coo.load(file, dimod.BinaryQuadraticModel.empty(dimod.SPIN))

        shutil.rmtree(tmpdir)

        self.assertEqual(bqm, new_bqm)

    def test_functional_string_empty_BINARY(self):

        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        s = coo.dumps(bqm)
        new_bqm = coo.loads(s, dimod.BinaryQuadraticModel.empty(dimod.BINARY))

        self.assertEqual(bqm, new_bqm)

    def test_functional_string_empty_SPIN(self):

        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        s = coo.dumps(bqm)
        new_bqm = coo.loads(s, dimod.BinaryQuadraticModel.empty(dimod.SPIN))

        self.assertEqual(bqm, new_bqm)

    def test_functional_string_BINARY(self):

        bqm = dimod.BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.BINARY)

        s = coo.dumps(bqm)
        new_bqm = coo.loads(s, dimod.BinaryQuadraticModel.empty(dimod.BINARY))

        self.assertEqual(bqm, new_bqm)

    def test_functional_string_SPIN(self):

        bqm = dimod.BinaryQuadraticModel({0: 1.}, {(0, 1): 2, (2, 3): .4}, 0.0, dimod.SPIN)

        s = coo.dumps(bqm)
        new_bqm = coo.loads(s, dimod.BinaryQuadraticModel.empty(dimod.SPIN))

        self.assertEqual(bqm, new_bqm)
