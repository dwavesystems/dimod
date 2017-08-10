import unittest

from dimod.decorators import *


class dummyResponse:
    def relabel_samples(self, relabel, copy=True):
        return None


class TestIndexRelabelling(unittest.TestCase):
    def test_qubo_unorderable(self):

        # create a dummy function that checks that the nodes
        @qubo_index_labels(0)
        def qubo_func(Q):
            labels = set().union(*Q)

            for idx in range(len(labels)):
                self.assertIn(idx, labels)

            # assume that relabel is applied correctly
            return dummyResponse()

        # variables with the same type of labels
        Q = {('a', 'b'): 0, ('b', 'c'): 1}
        qubo_func(Q)

        # variables with multiple types of label which are unorderable
        Q = {('a', 3): 0, ('b', 'c'): 1}
        qubo_func(Q)

    def test_ising_unorderable(self):

        # create dummy function that checks the nodes
        @ising_index_labels(0, 1)
        def ising_func(h, J):
            labels = set().union(*J) | set(h)

            for idx in range(len(labels)):
                self.assertIn(idx, labels)

            # assume that relabel is applied correctly
            return dummyResponse()

        h = {'a': 0, 'b': 1}
        J = {('a', 'c'): .5}
        ising_func(h, J)

        # unorderable labels
        h = {(0, 1): 0, 'b': 0}
        J = {((0, 1), 'b'): -1}

        ising_func(h, J)


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
