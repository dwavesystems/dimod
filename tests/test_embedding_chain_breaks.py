import unittest

import numpy as np

import dimod


class TestBrokenChains(unittest.TestCase):
    def test_broken_chains_typical(self):
        S = np.array([[-1, 1, -1, 1],
                      [1, 1, -1, -1],
                      [-1, 1, -1, -1]])
        chains = [[0, 1], [2, 3]]

        broken = dimod.embedding.broken_chains(S, chains)

        np.testing.assert_array_equal([[1, 1], [0, 0], [1, 0]], broken)

    def test_broken_chains_chains_length_0(self):
        S = np.array([[-1, 1, -1, 1],
                      [1, 1, -1, -1],
                      [-1, 1, -1, -1]])
        chains = [[0, 1], [], [2, 3]]

        broken = dimod.embedding.broken_chains(S, chains)

        np.testing.assert_array_equal([[1, 0, 1], [0, 0, 0], [1, 0, 0]], broken)

    def test_broken_chains_single_sample(self):
        S = [-1, 1, 1, 1]
        chains = [[0, 1], [2, 3]]
        with self.assertRaises(ValueError):
            dimod.embedding.broken_chains(S, chains)

    def test_matrix(self):
        samples_matrix = np.matrix([[-1, +1, -1, +1],
                                    [+1, +1, +1, +1],
                                    [-1, -1, +1, -1],
                                    [-1, -1, +1, +1]], dtype='int8')
        chain_list = [(0, 1), (2, 3)]

        broken = dimod.embedding.broken_chains(samples_matrix, chain_list)


class TestDiscardMatrix(unittest.TestCase):
    def test_discard_no_breaks_all_ones_identity_embedding(self):

        samples_matrix = np.matrix(np.ones((100, 50)), dtype='int8')
        chain_list = [[idx] for idx in range(50)]

        new_matrix, idxs = dimod.embedding.discard(samples_matrix, chain_list)

        np.testing.assert_equal(new_matrix, samples_matrix)

    def test_discard_no_breaks_all_ones_one_var_embedding(self):

        samples_matrix = np.matrix(np.ones((100, 50)), dtype='int8')
        chain_list = [[idx for idx in range(50)]]

        new_matrix, idxs = dimod.embedding.discard(samples_matrix, chain_list)

        self.assertEqual(new_matrix.shape, (100, 1))

    def test_discard_typical(self):

        samples_matrix = np.matrix([[-1, +1, -1, +1],
                                    [+1, +1, +1, +1],
                                    [-1, -1, +1, -1],
                                    [-1, -1, +1, +1]], dtype='int8')
        chain_list = [(0, 1), (2, 3)]

        new_matrix, idxs = dimod.embedding.discard(samples_matrix, chain_list)

        np.testing.assert_equal(new_matrix, [[+1, +1],
                                             [-1, +1]])


class TestMajorityVote(unittest.TestCase):
    def test_typical_spin(self):
        S = np.array([[-1, +1, -1, +1],
                      [+1, +1, -1, +1],
                      [-1, +1, -1, -1]])
        chains = [[0, 1, 2], [3]]

        samples, idx = dimod.embedding.majority_vote(S, chains)

        np.testing.assert_equal(samples, [[-1, +1],
                                          [+1, +1],
                                          [-1, -1]])

    def test_typical_binary(self):
        S = np.array([[0, 1, 0, 1],
                      [1, 1, 0, 1],
                      [0, 1, 0, 0]])
        chains = [[0, 1, 2], [3]]

        samples, idx = dimod.embedding.majority_vote(S, chains)

        np.testing.assert_equal(samples, [[0, 1],
                                          [1, 1],
                                          [0, 0]])

    def test_four_chains(self):
        S = [[-1, -1, -1, -1],
             [+1, -1, -1, -1],
             [+1, +1, -1, -1],
             [-1, +1, -1, -1],
             [-1, +1, +1, -1],
             [+1, +1, +1, -1],
             [+1, -1, +1, -1],
             [-1, -1, +1, -1],
             [-1, -1, +1, +1],
             [+1, -1, +1, +1],
             [+1, +1, +1, +1],
             [-1, +1, +1, +1],
             [-1, +1, -1, +1],
             [+1, +1, -1, +1],
             [+1, -1, -1, +1],
             [-1, -1, -1, +1]]
        chains = [[0], [1], [2, 3]]

        samples, idx = dimod.embedding.majority_vote(S, chains)

        self.assertEqual(samples.shape, (16, 3))
        self.assertEqual(set().union(*samples), {-1, 1})  # should be spin-valued
