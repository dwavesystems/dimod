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
        samples = np.array([[-1, 1], [1, 1]])

        embedding = {'a': {0, 1}}

        freq = dimod.embedding.chain_break_frequency(samples, embedding)

        self.assertEqual(freq, {'a': .5})

    def test_chain_break_frequency_response_mix_string_labels(self):
        response = dimod.Response.from_dicts([{'a': 1, 'b': 0}, {'a': 0, 'b': 0}], {'energy': [1, 0]}, dimod.BINARY)
        embedding = {0: {'a', 'b'}}
        freq = dimod.embedding.chain_break_frequency(response, embedding)

        self.assertEqual(freq, {0: .5})
