# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================

import unittest
import itertools
from collections import Mapping
import random

import numpy as np
import numpy.testing as npt

import dimod

try:
    import networkx as nx
    _networkx = True
except ImportError:
    _networkx = False


class TestUtils(unittest.TestCase):
    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_target_to_source_identity_embedding(self):
        """a 1-to-1 embedding should not change the adjacency"""
        target_adj = nx.karate_club_graph()

        embedding = {v: {v} for v in target_adj}

        source_adj = dimod.embedding.target_to_source(target_adj, embedding)

        # test the adjacencies are equal (source_adj is a dict and target_adj is a networkx graph)
        for v in target_adj:
            self.assertIn(v, source_adj)
            for u in target_adj[v]:
                self.assertIn(u, source_adj[v])

        for v in source_adj:
            self.assertIn(v, target_adj)
            for u in source_adj[v]:
                self.assertIn(u, target_adj[v])

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_target_to_source_embedding_to_one_node(self):
        """an embedding that maps everything to one node should result in a singleton graph"""
        target_adj = nx.barbell_graph(16, 7)
        embedding = {'a': set(target_adj)}  # all map to 'a'

        source_adj = dimod.embedding.target_to_source(target_adj, embedding)
        self.assertEqual(source_adj, {'a': set()})

        embedding = {'a': {0, 1}}  # not every node is assigned to a chain
        source_adj = dimod.embedding.target_to_source(target_adj, embedding)
        self.assertEqual(source_adj, {'a': set()})

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_target_to_source_embedding_overlap(self):
        """overlapping embeddings should raise an error"""
        target_adj = nx.complete_graph(5)
        embedding = {'a': {0, 1}, 'b': {1, 2}}  # overlap

        with self.assertRaises(ValueError):
            source_adj = dimod.embedding.target_to_source(target_adj, embedding)

    def test_target_to_source_square_to_triangle(self):
        target_adjacency = {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {0, 2}}  # a square graph
        embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        source_adjacency = dimod.embedding.target_to_source(target_adjacency, embedding)
        self.assertEqual(source_adjacency, {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}})

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_edgelist_to_adjacency_typical(self):
        graph = nx.barbell_graph(17, 8)

        edgelist = set(graph.edges())

        adj = dimod.embedding.utils.edgelist_to_adjacency(edgelist)

        # test that they're equal
        for u, v in edgelist:
            self.assertIn(u, adj)
            self.assertIn(v, adj)
            self.assertIn(u, adj[v])
            self.assertIn(v, adj[u])

        for u in adj:
            for v in adj[u]:
                self.assertTrue((u, v) in edgelist or (v, u) in edgelist)
                self.assertFalse((u, v) in edgelist and (v, u) in edgelist)

    def test_chain_to_quadratic_K5(self):
        """Test that when given a chain, the returned Jc uses all
        available edges."""
        chain_variables = set(range(5))

        # fully connected
        adjacency = {u: set(chain_variables) for u in chain_variables}
        for v, neighbors in adjacency.items():
            neighbors.remove(v)

        Jc = dimod.embedding.chain_to_quadratic(chain_variables, adjacency, 1.0)

        for u, v in itertools.combinations(chain_variables, 2):
            self.assertFalse((u, v) in Jc and (v, u) in Jc)
            self.assertTrue((u, v) in Jc or (v, u) in Jc)
        for u in chain_variables:
            self.assertFalse((u, u) in Jc)

    def test_chain_to_quadratic_5_cycle(self):
        chain_variables = set(range(5))

        # now try a cycle
        adjacency = {v: {(v + 1) % 5, (v - 1) % 5} for v in chain_variables}

        Jc = dimod.embedding.chain_to_quadratic(chain_variables, adjacency, 1.0)

        for u in adjacency:
            for v in adjacency[u]:
                self.assertFalse((u, v) in Jc and (v, u) in Jc)
                self.assertTrue((u, v) in Jc or (v, u) in Jc)

    def test_chain_to_quadratic_disconnected(self):
        chain_variables = {0, 2}

        adjacency = {0: {1}, 1: {0, 2}, 2: {1}}

        with self.assertRaises(ValueError):
            dimod.embedding.chain_to_quadratic(chain_variables, adjacency, 1.0)

    def test_chain_break_frequency_matrix_all_ones(self):
        """should have no breaks"""

        samples = np.ones((10, 5))

        embedding = {'a': {2, 4}, 'b': {1, 3}}

        freq = dimod.embedding.chain_break_frequency(samples, embedding)

        self.assertEqual(freq, {'a': 0, 'b': 0})

    def test_chain_break_frequency_matrix_all_zeros(self):
        """should have no breaks"""

        samples = np.zeros((10, 5))

        embedding = {'a': {2, 4}, 'b': {1, 3}}

        freq = dimod.embedding.chain_break_frequency(samples, embedding)

        self.assertEqual(freq, {'a': 0, 'b': 0})

    def test_chain_break_frequency_matrix_mix(self):
        samples = np.matrix([[-1, 1], [1, 1]])

        embedding = {'a': {0, 1}}

        freq = dimod.embedding.chain_break_frequency(samples, embedding)

        self.assertEqual(freq, {'a': .5})

    def test_chain_break_frequency_response_mix_string_labels(self):
        response = dimod.Response.from_dicts([{'a': 1, 'b': 0}, {'a': 0, 'b': 0}], {'energy': [1, 0]})
        embedding = {0: {'a', 'b'}}
        freq = dimod.embedding.chain_break_frequency(response, embedding)

        self.assertEqual(freq, {0: .5})


class TestEmbedBQM(unittest.TestCase):
    def test_embed_bqm_empty(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        embedded_bqm = dimod.embed_bqm(bqm, {}, {})

        self.assertIsInstance(embedded_bqm, dimod.BinaryQuadraticModel)
        self.assertFalse(embedded_bqm)  # should be empty

    def test_embed_bqm_subclass_propagation(self):

        class MyBQM(dimod.BinaryQuadraticModel):
            pass

        bqm = MyBQM.empty(dimod.BINARY)

        embedded_bqm = dimod.embed_bqm(bqm, {}, {})

        self.assertIsInstance(embedded_bqm, dimod.BinaryQuadraticModel)
        self.assertIsInstance(embedded_bqm, MyBQM)
        self.assertFalse(embedded_bqm)  # should be empty

    def test_embed_bqm_only_offset(self):
        bqm = dimod.BinaryQuadraticModel({}, {}, 1.0, dimod.SPIN)

        embedded_bqm = dimod.embed_bqm(bqm, {}, {})

        self.assertEqual(bqm, embedded_bqm)

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_embed_bqm_identity(self):

        bqm = dimod.BinaryQuadraticModel({'a': -1}, {(0, 1): .5, (1, 'a'): -1.}, 1.0, dimod.BINARY)

        embedding = {v: {v} for v in bqm.linear}  # identity embedding
        target_adj = bqm.to_networkx_graph()  # identity target graph

        embedded_bqm = dimod.embed_bqm(bqm, embedding, target_adj)

        self.assertEqual(bqm, embedded_bqm)

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_embed_bqm_NAE3SAT_to_square(self):

        h = {'a': 0, 'b': 0, 'c': 0}
        J = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}

        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

        embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

        embedded_bqm = dimod.embed_bqm(bqm, embedding, nx.cycle_graph(4), chain_strength=1)

        self.assertEqual(embedded_bqm,
                         dimod.BinaryQuadraticModel({0: 0, 1: 0, 2: 0, 3: 0},
                                                    {(0, 1): 1, (1, 2): 1, (2, 3): -1, (0, 3): 1},
                                                    1.0,  # offset the energy from satisfying chains
                                                    dimod.SPIN))

        # check that the energy has been preserved
        for config in itertools.product((-1, 1), repeat=3):
            sample = dict(zip(('a', 'b', 'c'), config))
            target_sample = {u: sample[v] for v, chain in embedding.items() for u in chain}  # no chains broken
            self.assertAlmostEqual(bqm.energy(sample), embedded_bqm.energy(target_sample))

    def test_embed_ising_components_empty(self):
        h = {}
        J = {}
        embedding = {}
        adjacency = {}

        he, Je = dimod.embedding.embed_ising(h, J, embedding, adjacency)

        self.assertFalse(he)
        self.assertFalse(Je)
        self.assertIsInstance(he, dict)
        self.assertIsInstance(Je, dict)

    def test_embed_ising_bad_chain(self):
        h = {}
        j = {(0, 1): 1}
        embeddings = {0: {0, 2}, 1: {1}}  # (0, 2) not connected
        adj = {0: {1}, 1: {0, 2}, 2: {1}}

        with self.assertRaises(ValueError):
            dimod.embedding.embed_ising(h, j, embeddings, adj)

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_embed_ising_nonadj(self):
        """chain in embedding has non-adjacenct nodes"""
        h = {}
        j = {(0, 1): 1}
        embeddings = {0: {0, 1}, 1: {3, 4}}

        adj = nx.Graph()
        adj.add_edges_from({(0, 1), (1, 2), (2, 3), (3, 4)})

        with self.assertRaises(ValueError):
            dimod.embedding.embed_ising(h, j, embeddings, adj)

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_embed_ising_h_embedding_mismatch(self):
        """there is not a mapping in embedding for every var in h"""
        h = dict(zip(range(3), [1, 2, 3]))
        j = {}
        embedding = {0: (0, 1), 1: (2,)}
        adj = nx.Graph()
        adj.add_edges_from({(0, 1), (1, 2)})

        with self.assertRaises(ValueError):
            dimod.embedding.embed_ising(h, j, embedding, adj)

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_embed_ising_j_index_too_large(self):
        """j references a variable not mentioned in embedding"""
        h = {}
        j = {(0, 1): -1, (0, 2): 1}
        embedding = {0: (0, 1), 1: (2,)}
        adj = nx.Graph()
        adj.add_edges_from({(0, 1), (1, 2)})

        with self.assertRaises(ValueError):
            dimod.embedding.embed_ising(h, j, embedding, adj)

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_embed_ising_typical(self):
        h = {0: 1, 1: 10}
        j = {(0, 1): 15, (2, 1): -8, (0, 2): 5, (2, 0): -2}
        embeddings = {0: [1], 1: [2, 3], 2: [0]}

        adj = nx.Graph()
        adj.add_edges_from({(0, 1), (1, 2), (2, 3), (3, 0), (2, 0)})

        expected_h0 = {0: 0, 1: 1, 2: 5, 3: 5}
        expected_j0 = {(0, 1): 3, (0, 2): -4, (0, 3): -4, (1, 2): 15, (2, 3): -1}

        h0, j0 = dimod.embedding.embed_ising(h, j, embeddings, adj)
        self.assertEqual(h0, expected_h0)

        # check j0
        for (u, v), bias in j0.items():
            self.assertTrue((u, v) in expected_j0 or (v, u) in expected_j0)
            self.assertFalse((u, v) in expected_j0 and (v, u) in expected_j0)

            if (u, v) in expected_j0:
                self.assertEqual(expected_j0[(u, v)], bias)
            else:
                self.assertEqual(expected_j0[(v, u)], bias)

    def test_embed_ising_embedding_not_in_adj(self):
        """embedding refers to a variable not in the adjacency"""
        h = {0: 0, 1: 0}
        J = {(0, 1): 1}

        embedding = {0: [0, 1], 1: [2]}

        adjacency = {0: {1}, 1: {0}}

        with self.assertRaises(ValueError):
            dimod.embedding.embed_ising(h, J, embedding, adjacency)

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_embed_bqm_BINARY(self):
        Q = {('a', 'a'): 0, ('a', 'b'): -1, ('b', 'b'): 0, ('c', 'c'): 0, ('b', 'c'): -1}
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

        embedded_bqm = dimod.embed_bqm(bqm, embedding, nx.cycle_graph(4), chain_strength=1)

        # check that the energy has been preserved
        for config in itertools.product((0, 1), repeat=3):
            sample = dict(zip(('a', 'b', 'c'), config))
            target_sample = {u: sample[v] for v, chain in embedding.items() for u in chain}  # no chains broken
            self.assertAlmostEqual(bqm.energy(sample), embedded_bqm.energy(target_sample))

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_embed_qubo(self):
        Q = {('a', 'a'): 0, ('a', 'b'): -1, ('b', 'b'): 0, ('c', 'c'): 0, ('b', 'c'): -1}
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

        embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

        target_Q = dimod.embed_qubo(Q, embedding, nx.cycle_graph(4), chain_strength=1)

        self.assertEqual(target_Q, {(0, 0): 0.0, (1, 1): 0.0, (2, 2): 2.0, (3, 3): 2.0,
                                    (0, 1): -1.0, (1, 2): -1.0, (2, 3): -4.0})

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_embedding_with_extra_chains(self):
        embedding = {0: [0, 1], 1: [2], 2: [3]}
        G = nx.cycle_graph(4)

        bqm = dimod.BinaryQuadraticModel.from_qubo({(0, 0): 1})

        target_bqm = dimod.embed_bqm(bqm, embedding, G)

        for v in itertools.chain(*embedding.values()):
            self.assertIn(v, target_bqm)


class TestIterUnembed(unittest.TestCase):

    def test_majority_vote(self):
        """should return the most common value in the chain"""

        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        embedding = {'a': {0, 1, 2}}

        # specify that majority vote should be used
        source_samples = list(dimod.iter_unembed(samples, embedding, chain_break_method=dimod.embedding.majority_vote))

        self.assertEqual(source_samples, [{'a': -1}, {'a': +1}])

    def test_majority_vote_with_response(self):
        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        # now set up an embedding that works for one sample and not the other
        embedding = {'a': {0, 1, 2}}

        # load the samples into a dimod response
        response = dimod.Response.from_dicts(samples, {'energy': [0 for __ in samples]})

        # specify that majority vote should be used
        source_samples = list(dimod.iter_unembed(samples, embedding, chain_break_method=dimod.embedding.majority_vote))

        self.assertEqual(len(source_samples), 2)
        self.assertEqual(source_samples, [{'a': -1}, {'a': +1}])

    def test_discard(self):
        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        embedding = {'a': {0, 1, 2}}

        # specify that majority vote should be used
        source_samples = list(dimod.iter_unembed(samples, embedding, chain_break_method=dimod.embedding.discard))

        # no samples should be returned because they are all broken
        self.assertEqual(len(source_samples), 0)

        # now set up an embedding that works for one sample and not the other
        embedding = {'a': {0, 1}, 'b': {2}}

        # specify that discard should be used
        source_samples = list(dimod.iter_unembed(samples, embedding, chain_break_method=dimod.embedding.discard))

        # only the first sample should be returned
        self.assertEqual(len(source_samples), 1)
        self.assertEqual(source_samples, [{'a': -1, 'b': +1}])

    def test_discard_with_response(self):
        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        # now set up an embedding that works for one sample and not the other
        embedding = {'a': {0, 1}, 'b': {2}}

        # load the samples into a dimod response
        response = dimod.Response.from_dicts(samples, {'energy': [0 for __ in samples]})

        # specify that discard should be used
        source_samples = list(dimod.iter_unembed(samples, embedding, chain_break_method=dimod.embedding.discard))

        # only the first sample should be returned
        self.assertEqual(len(source_samples), 1)
        self.assertEqual(source_samples, [{'a': -1, 'b': +1}])

    def test_energy_minimization(self):
        sample0 = {0: -1, 1: -1, 2: +1, 3: +1}
        sample1 = {0: +1, 1: -1, 2: +1, 3: -1}
        samples = [sample0, sample1]

        embedding = {'a': {0, 1}, 'b': {2}, 'c': {3}}

        linear = {'a': -1, 'b': 0, 'c': 0}
        quadratic = {}

        chain_break_method = dimod.embedding.MinimizeEnergy(linear=linear, quadratic=quadratic)
        source_samples = list(dimod.iter_unembed(samples, embedding, chain_break_method=chain_break_method))

        source0, source1 = source_samples

        # no broken chains
        self.assertEqual(source0, {'a': -1, 'b': +1, 'c': +1})

        # in this case 'a' being spin-up minimizes the energy
        self.assertEqual(source1, {'a': +1, 'b': +1, 'c': -1})

        linear = {'a': 1, 'b': 0, 'c': 0}
        quadratic = {('a', 'b'): -5}

        chain_break_method = dimod.embedding.MinimizeEnergy(linear=linear, quadratic=quadratic)
        source_samples = list(dimod.iter_unembed(samples, embedding, chain_break_method=chain_break_method))

        source0, source1 = source_samples

        # no broken chains
        self.assertEqual(source0, {'a': -1, 'b': +1, 'c': +1})

        # in this case 'a' being spin-up minimizes the energy due to the quadratic bias
        self.assertEqual(source1, {'a': +1, 'b': +1, 'c': -1})

        # now we need two broken chains
        sample = {0: +1, 1: -1, 2: +1, 3: -1, 4: +1}
        samples = [sample]

        embedding = {'a': {0, 1}, 'b': {2, 3}, 'c': {4}}

        quadratic = {('a', 'b'): -1, ('b', 'c'): 1}

        chain_break_method = dimod.embedding.MinimizeEnergy(quadratic=quadratic)
        source_samples = list(dimod.iter_unembed(samples, embedding, chain_break_method=chain_break_method))

        source, = source_samples

        self.assertEqual(source, {'b': -1, 'c': +1, 'a': -1})


class TestUnembedResponse(unittest.TestCase):
    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_energies_functional(self):
        h = {'a': .1, 'b': 0, 'c': 0}
        J = {('a', 'b'): 1, ('b', 'c'): 1.3, ('a', 'c'): -1}
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J, offset=1.3)

        embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

        embedded_bqm = dimod.embed_bqm(bqm, embedding, nx.cycle_graph(4), chain_strength=1)

        embedded_response = dimod.ExactSolver().sample(embedded_bqm)

        response = dimod.unembed_response(embedded_response, embedding, bqm)

        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(bqm.energy(sample), energy)

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_energies_discard(self):
        h = {'a': .1, 'b': 0, 'c': 0}
        J = {('a', 'b'): 1, ('b', 'c'): 1.3, ('a', 'c'): -1}
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J, offset=1.3)

        embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

        embedded_bqm = dimod.embed_bqm(bqm, embedding, nx.cycle_graph(4), chain_strength=1)

        embedded_response = dimod.ExactSolver().sample(embedded_bqm)

        chain_break_method = dimod.embedding.discard
        response = dimod.unembed_response(embedded_response, embedding, bqm,
                                          chain_break_method=chain_break_method)

        self.assertEqual(len(embedded_response) / 2, len(response))  # half chains should be broken

        for sample, energy in response.data(['sample', 'energy']):
            self.assertEqual(bqm.energy(sample), energy)


class TestEmbeddingChainBreaks(unittest.TestCase):

    def test__most_common(self):
        from dimod.embedding.chain_breaks import _most_common

        self.assertEqual(_most_common([-1, +1, +1]), +1)
        self.assertEqual(_most_common([+1, -1, -1]), -1)
        self.assertEqual(_most_common([0, 1, 1]), 1)
        self.assertEqual(_most_common([1, 0, 0]), 0)

        with self.assertRaises(ValueError):
            _most_common([])

    def test_all_equal(self):
        from dimod.embedding.chain_breaks import _all_equal

        self.assertTrue(_all_equal([1, 1]))
        self.assertTrue(_all_equal([-1, -1]))
        self.assertTrue(_all_equal([0, 0]))
        self.assertFalse(_all_equal([+1, -1]))
        self.assertFalse(_all_equal([1, 0]))
        self.assertTrue(_all_equal([]))
