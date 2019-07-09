import itertools
import unittest

import numpy as np

from dimod.bqm.cybqm import AdjArrayBQM


class TestConstruction(unittest.TestCase):
    """Tests for properties and special methods"""

    def test_empty(self):
        bqm = AdjArrayBQM()

        self.assertEqual(len(bqm), 0)
        self.assertEqual(bqm.shape, (0, 0))

        lin, quad = bqm.to_lists()
        self.assertEqual(lin, [])
        self.assertEqual(quad, [])

    def test_integral_nonzero(self):
        bqm = AdjArrayBQM(1000)

        self.assertEqual(len(bqm), 1000)
        self.assertEqual(bqm.shape, (1000, 0))

        lin, quad = bqm.to_lists()
        self.assertEqual(lin, [(0, 0) for _ in range(1000)])
        self.assertEqual(quad, [])

    def test_dense_triu(self):
        bqm = AdjArrayBQM(np.triu(np.ones((5, 5))))

        self.assertEqual(bqm.shape, (5, 10))
        lin, quad = bqm.to_lists()
        self.assertEqual(lin, [(d, 1) for d in range(0, 5*4, 4)])
        self.assertEqual(quad, [(v, 1)
                                for u in range(5)
                                for v in range(5)
                                if u != v])

    def test_dense(self):
        bqm = AdjArrayBQM([[.1, 1, 2], [0, 0, 0], [1, 1, 0]])

        self.assertEqual(bqm.shape, (3, 3))

        lin, quad = bqm.to_lists()
        self.assertEqual(lin, [(0, .1), (2, 0), (4, 0)])
        self.assertEqual(quad, [(1, 1), (2, 3),
                                (0, 1), (2, 1),
                                (0, 3), (1, 1)])


class TestEnergies(unittest.TestCase):
    def test_2path(self):
        bqm = AdjArrayBQM([[.1, -1], [0, -.2]])
        samples = [[-1, -1],
                   [-1, +1],
                   [+1, -1],
                   [+1, +1]]

        energies = bqm.energies(np.asarray(samples))

        np.testing.assert_array_almost_equal(energies, [-.9, .7, 1.3, -1.1])

    def test_5chain(self):
        arr = np.tril(np.triu(np.ones((5, 5)), 1), 1)
        bqm = AdjArrayBQM(arr)
        samples = [[-1, +1, -1, +1, -1]]

        energies = bqm.energies(np.asarray(samples))
        np.testing.assert_array_almost_equal(energies, [-4])

    def test_random(self):
        bqm = AdjArrayBQM([[0, -1, 0, 0],
                           [0, 0, .5, .2],
                           [0, 0, 0, 1.3],
                           [0, 0, 0, 0]])

        J = {(0, 1): -1, (1, 2): .5, (1, 3): .2, (2, 3): 1.3}

        for sample in itertools.product((-1, 1), repeat=len(bqm)):
            energy, = bqm.energies(np.atleast_2d(sample))

            target = sum(b*sample[u]*sample[v] for (u, v), b in J.items())

            self.assertAlmostEqual(energy, target)
