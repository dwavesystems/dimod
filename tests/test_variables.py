import unittest

from dimod.variables import Variables, MutableVariables


class TestVariables(unittest.TestCase):
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


class TestMutableVariables(unittest.TestCase):
    def assertConsistentVariables(self, variables):
        for i, v in enumerate(variables):
            self.assertEqual(variables.index(v), i)

    def test_construction(self):
        li = [-1, 1, 2, 'a', 'b', 'c']

        variables = MutableVariables(li)

        self.assertEqual(variables, li)

    def test__setitem__(self):
        variables = MutableVariables('abc')

        variables[0] = -1
        self.assertEqual(variables, [-1, 'b', 'c'])
        self.assertConsistentVariables(variables)

        # set to self
        variables[0] = -1
        self.assertEqual(variables, [-1, 'b', 'c'])
        self.assertConsistentVariables(variables)

        with self.assertRaises(IndexError):
            variables[100] = 'a'
        self.assertConsistentVariables(variables)

        with self.assertRaises(ValueError):
            variables[0] = 'b'
        self.assertConsistentVariables(variables)

    def test__delitem__(self):
        variables = MutableVariables('abc')

        del variables[0]
        self.assertNotIn('a', variables)
        self.assertEqual(variables, ['b', 'c'])
        self.assertConsistentVariables(variables)

        with self.assertRaises(IndexError):
            del variables[100]
        self.assertConsistentVariables(variables)

    def test_insert(self):
        variables = MutableVariables('abc')

        variables.insert(1, 5)
        self.assertEqual(variables, ['a', 5, 'b', 'c'])
        self.assertConsistentVariables(variables)

        variables.insert(4, 'q')
        self.assertEqual(variables, ['a', 5, 'b', 'c', 'q'])
        self.assertConsistentVariables(variables)

        variables.insert(1000, 't')
        self.assertEqual(variables, ['a', 5, 'b', 'c', 'q', 't'])
        self.assertConsistentVariables(variables)

        with self.assertRaises(ValueError):
            variables.insert(0, 'a')
        self.assertConsistentVariables(variables)
