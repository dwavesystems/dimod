import unittest

from dimod.bqm.cymutablebqm import AdjVectorBQM


class TestConstruction(unittest.TestCase):
    """Tests for properties and special methods and mutability"""

    def test_empty(self):
        bqm = AdjVectorBQM()
        self.assertEqual(bqm.shape, (0, 0))

    def test_append_variable(self):
        bqm = AdjVectorBQM()
        bqm.append_variable()
        self.assertEqual(bqm.get_linear(0), 0)
        bqm.append_variable(.5)
        self.assertEqual(bqm.get_linear(1), .5)
        with self.assertRaises(ValueError):
            bqm.get_linear(2)


class TestAddGetInteraction(unittest.TestCase):

    def test_one_interaction_simple(self):
        bqm = AdjVectorBQM(2)

        bqm.add_interaction(0, 1, .5)

        self.assertEqual(bqm.shape, (2, 1))
        self.assertEqual(bqm.get_quadratic(0, 1), .5)
        self.assertEqual(bqm.get_quadratic(1, 0), .5)

    def test_one_interaction_remove(self):
        bqm = AdjVectorBQM(2)

        bqm.add_interaction(0, 1, .5)
        bqm.remove_interaction(0, 1)

        self.assertEqual(bqm.shape, (2, 0))
        with self.assertRaises(ValueError):
            bqm.get_quadratic(0, 1)
        with self.assertRaises(ValueError):
            bqm.get_quadratic(1, 0)

    def test_one_interaction_multiremove(self):
        bqm = AdjVectorBQM(2)

        bqm.add_interaction(0, 1, .5)
        bqm.remove_interaction(0, 1)
        bqm.remove_interaction(0, 1)
        bqm.remove_interaction(0, 1)

        self.assertEqual(bqm.shape, (2, 0))
        with self.assertRaises(ValueError):
            bqm.get_quadratic(0, 1)
        with self.assertRaises(ValueError):
            bqm.get_quadratic(1, 0)

    def test_several_interaction_multiremove(self):
        bqm = AdjVectorBQM(3)

        bqm.add_interaction(0, 1, .5)
        bqm.add_interaction(0, 2, 1)
        bqm.remove_interaction(0, 2)
        bqm.remove_interaction(0, 2)

        self.assertEqual(bqm.shape, (3, 1))
        self.assertEqual(bqm.get_quadratic(0, 1), .5)
        self.assertEqual(bqm.get_quadratic(1, 0), .5)
        with self.assertRaises(ValueError):
            bqm.get_quadratic(0, 2)
        with self.assertRaises(ValueError):
            bqm.get_quadratic(2, 0)


class TestToAdjArray(unittest.TestCase):
    def test_empty(self):
        lin, quad = AdjVectorBQM().to_adjarray().to_lists()
        self.assertEqual(lin, [])
        self.assertEqual(quad, [])

    def test_linear(self):
        bqm = AdjVectorBQM(3)
        bqm.append_variable(.5)

        lin, quad = bqm.to_adjarray().to_lists()
        self.assertEqual(lin, [(0, 0), (0, 0), (0, 0), (0, .5)])
        self.assertEqual(quad, [])

    def test_3path(self):
        bqm = AdjVectorBQM(2)
        bqm.append_variable(.5)
        bqm.add_interaction(0, 1, 1.7)
        bqm.add_interaction(2, 1, -3)

        lin, quad = bqm.to_adjarray().to_lists()
        self.assertEqual(lin, [(0, 0), (1, 0), (3, .5)])
        self.assertEqual(quad, [(1, 1.7), (0, 1.7), (2, -3.0), (1, -3.0)])
