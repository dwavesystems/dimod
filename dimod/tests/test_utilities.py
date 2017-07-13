import unittest
from itertools import groupby

from dimod import ising_to_qubo, qubo_to_ising, ising_energy, qubo_energy


class TestIsingEnergy(unittest.TestCase):
    def test_trivial(self):
        en = ising_energy({}, {}, {})
        self.assertEqual(en, 0)

    def test_typical(self):
        # AND gate
        h = {0: -.5, 1: 0, 2: 1, 3: -.5}
        J = {(0, 2): -1, (1, 2): -1, (0, 3): .5, (1, 3): -1}

        en0 = min(ising_energy(h, J, {0: -1, 1: -1, 2: -1, 3: -1}),
                  ising_energy(h, J, {0: -1, 1: -1, 2: -1, 3: 1}))
        en1 = min(ising_energy(h, J, {0: 1, 1: -1, 2: -1, 3: -1}),
                  ising_energy(h, J, {0: 1, 1: -1, 2: -1, 3: 1}))
        en2 = min(ising_energy(h, J, {0: -1, 1: 1, 2: -1, 3: -1}),
                  ising_energy(h, J, {0: -1, 1: 1, 2: -1, 3: 1}))
        en3 = min(ising_energy(h, J, {0: 1, 1: 1, 2: 1, 3: -1}),
                  ising_energy(h, J, {0: 1, 1: 1, 2: 1, 3: 1}))

        self.assertEqual(en0, en1)
        self.assertEqual(en0, en2)
        self.assertEqual(en0, en3)


class TestQuboEnergy(unittest.TestCase):
    def test_trivial(self):
        en = qubo_energy({}, {})
        self.assertEqual(en, 0)

    def test_typical(self):
        # AND gate
        h = {0: -.5, 1: 0, 2: 1, 3: -.5}
        J = {(0, 2): -1, (1, 2): -1, (0, 3): .5, (1, 3): -1}
        Q, __ = ising_to_qubo(h, J)

        en0 = min(qubo_energy(Q, {0: 0, 1: 0, 2: 0, 3: 0}),
                  qubo_energy(Q, {0: 0, 1: 0, 2: 0, 3: 1}))
        en1 = min(qubo_energy(Q, {0: 1, 1: 0, 2: 0, 3: 0}),
                  qubo_energy(Q, {0: 1, 1: 0, 2: 0, 3: 1}))
        en2 = min(qubo_energy(Q, {0: 0, 1: 1, 2: 0, 3: 0}),
                  qubo_energy(Q, {0: 0, 1: 1, 2: 0, 3: 1}))
        en3 = min(qubo_energy(Q, {0: 1, 1: 1, 2: 1, 3: 0}),
                  qubo_energy(Q, {0: 1, 1: 1, 2: 1, 3: 1}))

        self.assertEqual(en0, en1)
        self.assertEqual(en0, en2)
        self.assertEqual(en0, en3)


class TestIsingToQubo(unittest.TestCase):
    def test_trivial(self):
        q, offset = ising_to_qubo({}, {})
        self.assertEqual(q, {})
        self.assertEqual(offset, 0)

    def test_no_zeros(self):
        q, offset = ising_to_qubo({0: 0, 0: 0, 0: 0}, {(0, 0): 0, (4, 5): 0})
        self.assertEqual(q, {})
        self.assertEqual(offset, 0)

    def test_j_diag(self):
        q, offset = ising_to_qubo({}, {(0, 0): 1, (300, 300): 99})
        self.assertEqual(q, {})
        self.assertEqual(offset, 100)

    def test_typical(self):
        h = {i: v for i, v in enumerate([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])}
        j = {(0, 0): -5, (0, 5): 2, (0, 8): 4, (1, 4): -5, (1, 7): 1, (2, 0): 5,
             (2, 1): 4, (3, 0): -1, (3, 6): -3, (3, 8): 3, (4, 0): 2, (4, 7): 3,
             (4, 9): 3, (5, 1): 3, (6, 5): -4, (6, 6): -5, (6, 7): -4, (7, 1): -4,
             (7, 8): 3, (8, 2): -4, (8, 3): -3, (8, 6): -5, (8, 7): -4, (9, 0): 4,
             (9, 1): -1, (9, 4): -5, (9, 7): 3}
        q, offset = ising_to_qubo(h, j)
        norm_q = normalized_matrix(q)
        self.assertEqual(norm_q, {(0, 0): -42, (0, 2): 20, (0, 3): -4, (0, 4): 8,
                                  (0, 5): 8, (0, 8): 16, (0, 9): 16, (1, 1): -4,
                                  (1, 2): 16, (1, 4): -20, (1, 5): 12, (1, 7): -12,
                                  (1, 9): -4, (2, 2): -16, (2, 8): -16, (3, 3): 4,
                                  (3, 6): -12, (4, 4): 2, (4, 7): 12, (4, 9): -8,
                                  (5, 5): -2, (5, 6): -16, (6, 6): 34, (6, 7): -16,
                                  (6, 8): -20, (7, 7): 8, (7, 8): -4, (7, 9): 12,
                                  (8, 8): 18})
        self.assertEqual(offset, -8)


class TestQuboToIsing(unittest.TestCase):
    def test_trivial(self):
        h, j, offset = qubo_to_ising({})
        self.assertEqual(h, {})
        self.assertEqual(j, {})
        self.assertEqual(offset, 0)

    def test_no_zeros(self):
        h, j, offset = qubo_to_ising({(0, 0): 0, (4, 5): 0})
        self.assertEqual(h, {0: 0, 4: 0, 5: 0})
        self.assertEqual(j, {})
        self.assertEqual(offset, 0)

    def test_typical(self):
        q = {(0, 0): 4, (0, 3): 5, (0, 5): 4, (1, 1): 5, (1, 6): 1, (1, 7): -2,
             (1, 9): -3, (3, 0): -2, (3, 1): 2, (4, 5): 4, (4, 8): 2, (4, 9): -1,
             (5, 1): 2, (5, 6): -5, (5, 8): -4, (6, 0): 1, (6, 5): 2, (6, 6): -4,
             (6, 7): -2, (7, 0): -2, (7, 5): -3, (7, 6): -5, (7, 7): -3, (7, 8): 1,
             (8, 0): 2, (8, 5): 1, (9, 7): -3}
        h, j, offset = qubo_to_ising(q)
        self.assertEqual(h, {0: 4.0, 1: 2.5, 3: 1.25, 4: 1.25, 5: 0.25,
                             6: -4.0, 7: -5.5, 8: 0.5, 9: -1.75})
        norm_j = normalized_matrix(j)
        self.assertEqual(norm_j, {(0, 3): 0.75, (0, 5): 1, (0, 6): 0.25, (0, 7): -0.5,
                                  (0, 8): 0.5, (1, 3): 0.5, (1, 5): 0.5, (1, 6): 0.25,
                                  (1, 7): -0.5, (1, 9): -0.75, (4, 5): 1, (4, 8): 0.5,
                                  (4, 9): -0.25, (5, 6): -0.75, (5, 7): -0.75,
                                  (5, 8): -0.75, (6, 7): -1.75, (7, 8): 0.25,
                                  (7, 9): -0.75})
        self.assertEqual(offset, -0.25)


def normalized_matrix(mat):
    def key_fn(x):
        return x[0]

    smat = sorted(((sorted(k), v) for k, v in mat.items()), key=key_fn)
    return dict((tuple(k), s) for k, g in groupby(smat, key=key_fn) for s in
                [sum(v for _, v in g)] if s != 0)