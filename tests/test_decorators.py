import unittest
from operator import itemgetter

from dimod.vartypes import SPIN, BINARY
from dimod.decorators import vartype_argument


class TestVartypeArgument(unittest.TestCase):
    def test_single_explicit_arg(self):
        @vartype_argument('x')
        def f(x):
            return x

        self.assertEqual(f(SPIN), SPIN)
        self.assertEqual(f(BINARY), BINARY)
        self.assertEqual(f('SPIN'), SPIN)
        self.assertEqual(f(x='SPIN'), SPIN)
        self.assertRaises(TypeError, f, 'x')
        self.assertRaises(TypeError, f, x='x')
        self.assertRaises(RuntimeError, f)

    def test_multiple_explicit_args(self):
        @vartype_argument('x', 'y')
        def f(x, y):
            return x, y

        self.assertEqual(f(SPIN, BINARY), (SPIN, BINARY))
        self.assertEqual(f('SPIN', 'BINARY'), (SPIN, BINARY))
        self.assertEqual(f(y='BINARY', x='SPIN'), (SPIN, BINARY))
        self.assertRaises(TypeError, f, 'x', 'y')
        self.assertRaises(TypeError, f, x='x', y='SPIN')
        self.assertRaises(RuntimeError, f)

    def test_kwargs(self):
        @vartype_argument('x', 'y')
        def f(**kwargs):
            return itemgetter('x', 'y')(kwargs)

        self.assertEqual(f(x=SPIN, y=BINARY), (SPIN, BINARY))
        self.assertRaises(RuntimeError, f)
        self.assertRaises(RuntimeError, f, x=SPIN)
        self.assertRaises(TypeError, f, x='x', y='SPIN')

    def test_explicit_with_kwargs(self):
        @vartype_argument('x', 'y')
        def f(x, **kwargs):
            return x, kwargs.pop('y')

        self.assertEqual(f('SPIN', y='BINARY'), (SPIN, BINARY))
        self.assertEqual(f(y='BINARY', x='SPIN'), (SPIN, BINARY))
        self.assertRaises(RuntimeError, f)
        self.assertRaises(RuntimeError, f, x=SPIN)
        self.assertRaises(TypeError, f, x='x', y='SPIN')

    def test_default_argname(self):
        @vartype_argument()
        def f(vartype):
            return vartype

        self.assertEqual(f('SPIN'), SPIN)
        self.assertEqual(f(vartype='SPIN'), SPIN)
        self.assertRaises(TypeError, f, 'invalid')
        self.assertRaises(TypeError, f, None)

    def test_argname_mismatch(self):
        @vartype_argument()
        def f(x=None):
            return x

        self.assertRaises(RuntimeError, f)
        self.assertRaises(RuntimeError, f, x=SPIN)

        @vartype_argument('vartype')
        def g(x=None):
            return x

        self.assertRaises(RuntimeError, g)
        self.assertRaises(RuntimeError, g, x=SPIN)
        self.assertRaises(TypeError, g, vartype=SPIN)
