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
