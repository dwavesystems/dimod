import unittest

from dimod.bqm.cybqm import AdjArrayBQM


class TestConstruction(unittest.TestCase):
    """Tests for properties and special methods"""

    def test_empty(self):
        bqm = AdjArrayBQM(0, 0)

        self.assertEqual(len(bqm), 0)
        self.assertEqual(bqm.shape, (0, 0))

    def test_empty_large_interactions(self):
        # in this case we've allocated 10000 but because nothing has been set
        # we still count as having 0 biases
        bqm = AdjArrayBQM(0, 10000)

        self.assertEqual(len(bqm), 0)
        self.assertEqual(bqm.shape, (0, 0))

    def test_linear_only_large_interactions(self):
        bqm = AdjArrayBQM(1000, 10000)

        self.assertEqual(len(bqm), 1000)
        self.assertEqual(bqm.shape, (1000, 0))
