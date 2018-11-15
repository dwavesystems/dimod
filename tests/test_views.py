import unittest

from dimod.views import Variables


class TestVariableView(unittest.TestCase):
    def test_duplicates(self):
        # should have no duplicates
        variables = Variables(['a', 'b', 'c', 'b'])
        self.assertEqual(list(variables), ['a', 'b', 'c'])

    def test_iterable(self):
        variables = Variables('abcdef')
        self.assertEqual(list(variables), list('abcdef'))

    def test_index(self):
        variables = Variables(range(5))
        self.assertEqual(variables.index(4), 4)

    def test_count(self):
        variables = Variables([1, 1, 1, 4, 5])
        self.assertEqual(list(variables), [1, 4, 5])
        for v in range(10):
            if v in variables:
                self.assertEqual(variables.count(v), 1)
            else:
                self.assertEqual(variables.count(v), 0)

    def test_len(self):
        variables = Variables(range(5))
        self.assertEqual(len(variables), 5)
        variables = Variables('aaaaa')
        self.assertEqual(len(variables), 1)
