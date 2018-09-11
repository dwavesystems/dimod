import unittest

import dimod


class TestFixVariables(unittest.TestCase):
    def test_3path(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 10}, {'ab': -1, 'bc': 1})
        fixed = dimod.fix_variables(bqm)
        self.assertEqual(fixed, {'a': -1, 'b': -1, 'c': 1})
