import unittest

from dimodsampler.decorators import *


class TestAPIDecorators(unittest.TestCase):
    def test_qubo_exceptions(self):

        @qubo(0)
        def qubo_func(Q):
            pass

        # not dict should raise a TypeError
        for q in [[], (), 0, 0.]:
            with self.assertRaises(TypeError):
                qubo_func(q)

        # bad edges should raise a typeerror
        Q = {7: 8}
        with self.assertRaises(TypeError):
            qubo_func(Q)

        # bad edge length should raise a valueerror
        Q = {(0, 1, 2): 8}
        with self.assertRaises(ValueError):
            qubo_func(Q)

    def test_ising_exceptions(self):

        @ising(0, 1)
        def ising_func(h, J):
            pass

        # not dict should raise a TypeError
        h = {}
        for J in [[], (), 0, 0.]:
            with self.assertRaises(TypeError):
                ising_func(h, J)

        J = {}
        for h in [0, 0.]:
            with self.assertRaises(TypeError):
                ising_func(h, J)

        # bad edges should raise a typeerror
        h = {}
        J = {7: 8}
        with self.assertRaises(TypeError):
            ising_func(h, J)

        # bad edge length should raise a valueerror
        h = {}
        J = {(0, 1, 2): 8}
        with self.assertRaises(ValueError):
            ising_func(h, J)

    def test_ising_h_change(self):

        @ising(0, 1)
        def ising_func(h, J):
            self.assertTrue(isinstance(h, dict))

            for u, v in J:
                self.assertIn(u, h)
                self.assertIn(v, h)

        ising_func([], {(0, 1): 1})
