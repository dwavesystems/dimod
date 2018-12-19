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
        samples_matrix = np.array([[-1, +1, -1, +1],
                                   [+1, +1, +1, +1],
                                   [-1, -1, +1, -1],
                                   [-1, -1, +1, +1]], dtype='int8')
        chain_list = [(0, 1), (2, 3)]

        broken = dimod.embedding.broken_chains(samples_matrix, chain_list)


class TestDiscard(unittest.TestCase):
    def test_discard_no_breaks_all_ones_identity_embedding(self):

        samples_matrix = np.array(np.ones((100, 50)), dtype='int8')
        chain_list = [[idx] for idx in range(50)]

        new_matrix, idxs = dimod.embedding.discard(samples_matrix, chain_list)

        np.testing.assert_equal(new_matrix, samples_matrix)

    def test_discard_no_breaks_all_ones_one_var_embedding(self):

        samples_matrix = np.array(np.ones((100, 50)), dtype='int8')
        chain_list = [[idx for idx in range(50)]]

        new_matrix, idxs = dimod.embedding.discard(samples_matrix, chain_list)

        self.assertEqual(new_matrix.shape, (100, 1))

    def test_discard_typical(self):

        samples_matrix = np.array([[-1, +1, -1, +1],
                                   [+1, +1, +1, +1],
                                   [-1, -1, +1, -1],
                                   [-1, -1, +1, +1]], dtype='int8')
        chain_list = [(0, 1), (2, 3)]

        new_matrix, idxs = dimod.embedding.discard(samples_matrix, chain_list)

        np.testing.assert_equal(new_matrix, [[+1, +1],
                                             [-1, +1]])

    def test_mixed_chain_types(self):
        chains = [(0, 1), [2, 3], {4, 5}]
        samples = [[1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 1]]
        unembedded, idx = dimod.embedding.discard(samples, chains)

        np.testing.assert_array_equal(unembedded, [[1, 1, 1]])
        np.testing.assert_array_equal(idx, [0])


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


class TestMinimizeEnergy(unittest.TestCase):
    def test_minimize_energy(self):
        embedding = {0: (0, 5), 1: (1, 6), 2: (2, 7), 3: (3, 8), 4: (4, 10)}
        h = []
        j = {(0, 1): -1, (0, 2): 2, (0, 3): 2, (0, 4): -1,
             (2, 1): -1, (1, 3): 2, (3, 1): -1, (1, 4): -1,
             (2, 3): 1, (4, 2): -1, (2, 4): -1, (3, 4): 1}

        bqm = dimod.BinaryQuadraticModel.from_ising(h, j)

        solutions = [
            [-1, -1, -1, -1, -1, -1, +1, +1, +1, 3, +1],
            [+1, +1, +1, +1, +1, -1, +1, -1, -1, 3, -1],
            [+1, +1, -1, +1, -1, -1, -1, -1, -1, 3, -1]
        ]
        expected = [
            [-1, -1, +1, +1, -1],
            [+1, +1, +1, -1, +1],
            [-1, -1, -1, +1, -1]
        ]

        cbm = dimod.embedding.MinimizeEnergy(bqm, embedding)

        unembedded, idx = cbm(solutions, [embedding[v] for v in range(5)])

        np.testing.assert_array_equal(expected, unembedded)

    def test_minimize_energy_easy(self):
        chains = ({0, 1}, [2], (4, 5, 6))
        embedding = {v: chain for v, chain in enumerate(chains)}
        h = [-1, 0, 0]
        j = {}
        bqm = dimod.BinaryQuadraticModel.from_ising(h, j)
        solutions = [
            [-1, -1, +1, 3, -1, -1, -1],
            [-1, +1, -1, 3, +1, +1, +1]
        ]
        expected = [
            [-1, +1, -1],
            [+1, -1, +1]
        ]
        cbm = dimod.embedding.MinimizeEnergy(bqm, embedding)

        unembedded, idx = cbm(solutions, chains)

        np.testing.assert_array_equal(expected, unembedded)

    def test_empty_matrix(self):
        chains = []
        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
        solutions = [[]]
        embedding = {}
        cbm = dimod.embedding.MinimizeEnergy(bqm, embedding)
        unembedded, idx = cbm(solutions, chains)

        np.testing.assert_array_equal([[]], unembedded)
        np.testing.assert_array_equal(idx, [0])

    def test_empty_chains(self):
        embedding = {}
        h = []
        j = {(0, 1): -1, (0, 2): 2, (0, 3): 2, (0, 4): -1,
             (2, 1): -1, (1, 3): 2, (3, 1): -1, (1, 4): -1,
             (2, 3): 1, (4, 2): -1, (2, 4): -1, (3, 4): 1}

        bqm = dimod.BinaryQuadraticModel.from_ising(h, j)

        solutions = [
            [-1, -1, -1, -1, -1, -1, +1, +1, +1, 3, +1],
            [+1, +1, +1, +1, +1, -1, +1, -1, -1, 3, -1],
            [+1, +1, -1, +1, -1, -1, -1, -1, -1, 3, -1]
        ]
        expected = [
            [-1, -1, +1, +1, -1],
            [+1, +1, +1, -1, +1],
            [-1, -1, -1, +1, -1]
        ]

        cbm = dimod.embedding.MinimizeEnergy(bqm, embedding)

        unembedded, idx = cbm(solutions, [])

        np.testing.assert_array_equal(unembedded, [[], [], []])
