import itertools
import unittest

import dimod
import numpy as np

try:
    import dimod.bqm._utils
except ImportError:
    cext = False
else:
    cext = True


class TestVectorBQM(unittest.TestCase):
    @staticmethod
    def check_consistent_vectorbqm(vbqm):
        num_variables = len(vbqm)

        for attr in ['vartype', 'dtype', 'index_dtype', 'offset', 'ldata', 'irow', 'icol', 'qdata', 'iadj']:
            assert hasattr(vbqm, attr)

        assert num_variables == len(vbqm.ldata)

        assert len(vbqm.iadj) == len(vbqm.ldata)
        assert all(v in vbqm.iadj for v in range(num_variables))

        assert len(vbqm.irow) == len(vbqm.icol)
        assert len(vbqm.icol) == len(vbqm.qdata)

        for idx, (i, j) in enumerate(zip(vbqm.irow, vbqm.icol)):
            assert vbqm.iadj[i][j] == idx

        assert vbqm.ldata.dtype is vbqm.dtype
        assert vbqm.qdata.dtype is vbqm.dtype
        assert isinstance(vbqm.offset, vbqm.dtype.type)  # cover object case
        assert vbqm.irow.dtype is vbqm.index_dtype
        assert vbqm.icol.dtype is vbqm.index_dtype

    def test_construction_empty(self):
        vbqm = dimod.VectorBQM([], [], 0.0, dimod.SPIN)

        self.check_consistent_vectorbqm(vbqm)

        np.testing.assert_array_equal(vbqm.ldata, [])
        np.testing.assert_array_equal(vbqm.qdata, [])
        self.assertEqual(vbqm.offset, 0.0)

    def test_construction_dense(self):
        vbqm = dimod.VectorBQM([-1, 1], [[0, 1], [.5, 0]], 1.5, dimod.BINARY)
        self.check_consistent_vectorbqm(vbqm)

        np.testing.assert_array_equal(vbqm.ldata, [-1, 1])
        np.testing.assert_array_equal(vbqm.qdata, [1.5])  # should aggregate
        self.assertEqual(vbqm.offset, 1.5)

    def test_construction_sparse(self):
        lin = np.zeros(6)
        quad = ([5, 4, 3, 2, 1, 0], [4, 3, 2, 1, 0, 1], [-1, .5, .5, 1.3, 1.2, -1])
        vbqm = dimod.VectorBQM(lin, quad, .2, dimod.SPIN)

        self.check_consistent_vectorbqm(vbqm)
        self.assertAlmostEqual(vbqm.qdata[vbqm.iadj[0][1]], .2)

    def test_multiple_constructions(self):
        lins = [[-.5, 0.0],
                np.array([-.5, 0.0])]

        quads = [[[0, -1], [0, 0]],
                 [[0, 0], [-1, 0]],
                 [[0, -.5], [-.5, 0]],
                 np.asarray([[0, -.5], [-.5, 0]]),
                 np.asarray([[0, -1], [0, 0]]),
                 ([0], [1], [-1])]

        bqms = [dimod.VectorBQM(l, q, 1.2, dimod.SPIN) for l in lins for q in quads]

        for b0, b1 in itertools.combinations(bqms, 2):
            self.assertEqual(b0, b1)

    def test_energy(self):
        bqm = dimod.VectorBQM([-1, +1], ([0], [1], [-1]), 1.5, dimod.SPIN)

        self.assertEqual(bqm.energy([-1, -1]), .5)
        self.assertEqual(bqm.energy([-1, +1]), 4.5)
        self.assertEqual(bqm.energy([+1, -1]), .5)
        self.assertEqual(bqm.energy([+1, +1]), .5)

    def test_energies(self):
        bqm = dimod.VectorBQM([-1, +1], ([0], [1], [-1]), 1.5, dimod.SPIN)

        samples = [[-1, -1], [-1, +1], [+1, -1], [+1, +1]]

        np.testing.assert_array_equal(bqm.energies(samples), [.5, 4.5, .5, .5])

    def test_energies_order_linear(self):

        samples = list(itertools.product((0, 1), repeat=3))

        # only linear + offset
        bqm = dimod.VectorBQM([1, 2, 3], [], 1.5, dimod.BINARY)
        rbqm = dimod.VectorBQM([3, 2, 1], [], 1.5, dimod.BINARY)

        renergies = rbqm.energies(samples)
        energies = bqm.energies(samples, order=[2, 1, 0])

        np.testing.assert_array_almost_equal(renergies, energies)

    def test_energies_order_quadratic(self):

        samples = list(itertools.product((0, 1), repeat=3))

        # only quadratic + offset
        bqm = dimod.VectorBQM([0, 0, 0], ([0, 1], [1, 2], [1, 2]), 1.5, dimod.BINARY)
        rbqm = dimod.VectorBQM([0, 0, 0], ([2, 1], [1, 0], [1, 2]), 1.5, dimod.BINARY)

        renergies = rbqm.energies(samples)
        energies = bqm.energies(samples, order=[2, 1, 0])

        np.testing.assert_array_almost_equal(renergies, energies)

    def test_energies_order(self):
        samples = list(itertools.product((0, 1), repeat=3))

        # all
        bqm = dimod.VectorBQM([1, 2, 3], ([0, 1], [1, 2], [1, 2]), 1.5, dimod.BINARY)
        rbqm = dimod.VectorBQM([3, 2, 1], ([2, 1], [1, 0], [1, 2]), 1.5, dimod.BINARY)

        renergies = rbqm.energies(samples)
        energies = bqm.energies(samples, order=[2, 1, 0])

        np.testing.assert_array_almost_equal(renergies, energies)

    @unittest.skipUnless(cext, "No c extension built")
    def test_energies_cpp(self):
        num_variables = 1000
        p = .1
        num_samples = 100

        linear = np.random.rand(num_variables)
        row, col = zip(*(pair for pair in itertools.combinations(range(num_variables), 2) if np.random.rand() < p))
        quad = np.random.rand(len(row))

        vbqm = dimod.VectorBQM(linear, (row, col, quad), np.random.rand(), dimod.BINARY)

        samples = np.random.randint(2, size=(num_samples, num_variables))

        energies_c = vbqm.energies(samples, _use_cpp_ext=True)
        energies_np = vbqm.energies(samples, _use_cpp_ext=False)

        np.testing.assert_array_almost_equal(energies_np, energies_c)

    def test_to_binary_2var(self):
        spin_bqm = dimod.VectorBQM([.5, 0], ([0], [1], [-1]), 1.3, dimod.SPIN)
        binary_bqm = spin_bqm.to_binary()

        for binary_sample in itertools.product((0, 1), repeat=len(binary_bqm)):
            spin_sample = [2*x-1 for x in binary_sample]
            self.assertAlmostEqual(binary_bqm.energy(binary_sample), spin_bqm.energy(spin_sample))

    def test_to_spin_2var(self):
        binary_bqm = dimod.VectorBQM([.5, 0], ([0], [1], [-1]), -1.4, dimod.BINARY)
        spin_bqm = binary_bqm.to_spin()

        for binary_sample in itertools.product((0, 1), repeat=len(binary_bqm)):
            spin_sample = [2*x-1 for x in binary_sample]
            self.assertAlmostEqual(binary_bqm.energy(binary_sample), spin_bqm.energy(spin_sample))

    def test_to_binary_to_spin(self):
        bqm = dimod.VectorBQM([0, 1, 2], ([0, 1], [1, 2], [-1, -2]), .5, dimod.SPIN)

        other = bqm.to_binary().to_spin()

        self.assertEqual(bqm, other)
