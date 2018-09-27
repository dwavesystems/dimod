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


class TestUnembedResponse(unittest.TestCase):

    def test_majority_vote(self):
        """should return the most common value in the chain"""

        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        embedding = {'a': {0, 1, 2}}

        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {})

        resp = dimod.Response.from_samples(samples, {'energy': [-1, 1]}, {}, dimod.SPIN)

        resp = dimod.unembed_response(resp, embedding, bqm, chain_break_method=dimod.embedding.majority_vote)

        # specify that majority vote should be used
        source_samples = list(resp)

        self.assertEqual(source_samples, [{'a': -1}, {'a': +1}])

    def test_majority_vote_with_response(self):
        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        # now set up an embedding that works for one sample and not the other
        embedding = {'a': {0, 1, 2}}

        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {})

        resp = dimod.Response.from_samples(samples, {'energy': [-1, 1]}, {}, dimod.SPIN)

        resp = dimod.unembed_response(resp, embedding, bqm, chain_break_method=dimod.embedding.majority_vote)

        # specify that majority vote should be used
        source_samples = list(resp)

        self.assertEqual(len(source_samples), 2)
        self.assertEqual(source_samples, [{'a': -1}, {'a': +1}])

    def test_discard(self):
        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        embedding = {'a': {0, 1, 2}}

        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {})

        resp = dimod.Response.from_samples(samples, {'energy': [-1, 1]}, {}, dimod.SPIN)

        resp = dimod.unembed_response(resp, embedding, bqm, chain_break_method=dimod.embedding.discard)

        source_samples = list(resp)

        # no samples should be returned because they are all broken
        self.assertEqual(len(source_samples), 0)

        # now set up an embedding that works for one sample and not the other
        embedding = {'a': {0, 1}, 'b': {2}}

        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {('a', 'b'): -1})

        resp = dimod.Response.from_samples(samples, {'energy': [-1, 1]}, {}, dimod.SPIN)

        resp = dimod.unembed_response(resp, embedding, bqm, chain_break_method=dimod.embedding.discard)

        source_samples = list(resp)
        # only the first sample should be returned
        self.assertEqual(len(source_samples), 1)
        self.assertEqual(source_samples, [{'a': -1, 'b': +1}])

    def test_discard_with_response(self):
        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        # now set up an embedding that works for one sample and not the other
        embedding = {'a': {0, 1}, 'b': {2}}

        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {('a', 'b'): -1})

        resp = dimod.Response.from_samples(samples, {'energy': [-1, 1]}, {}, dimod.SPIN)

        resp = dimod.unembed_response(resp, embedding, bqm, chain_break_method=dimod.embedding.discard)

        source_samples = list(resp)

        # only the first sample should be returned
        self.assertEqual(len(source_samples), 1)
        self.assertEqual(source_samples, [{'a': -1, 'b': +1}])

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

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_unembed_response_with_discard_matrix_typical(self):
        h = {'a': .1, 'b': 0, 'c': 0}
        J = {('a', 'b'): 1, ('b', 'c'): 1.3, ('a', 'c'): -1}
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J, offset=1.3)

        embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

        embedded_bqm = dimod.embed_bqm(bqm, embedding, nx.cycle_graph(4), chain_strength=1)

        embedded_response = dimod.ExactSolver().sample(embedded_bqm)

        chain_break_method = dimod.embedding.discard
        response = dimod.unembed_response(embedded_response, embedding, bqm,
                                          chain_break_method=dimod.embedding.discard)

        self.assertEqual(len(embedded_response) / 2, len(response))  # half chains should be broken

        for sample, energy in response.data(['sample', 'energy']):
            self.assertEqual(bqm.energy(sample), energy)

    def test_embedding_superset(self):
        # source graph in the embedding is a superset of the bqm
        response = dimod.Response(np.rec.array([([-1,  1, -1,  1, -1,  1, -1,  1], -1.4, 1),
                                                ([-1,  1, -1, -1, -1,  1, -1, -1], -1.4, 1),
                                                ([+1, -1, -1, -1,  1, -1, -1, -1], -1.6, 1),
                                                ([+1, -1, -1, -1,  1, -1, -1, -1], -1.6, 1)],
                                  dtype=[('sample', 'i1', (8,)), ('energy', '<f8'), ('num_occurrences', '<i8')]),
                                  [0, 1, 2, 3, 4, 5, 6, 7], {}, 'SPIN')
        embedding = {0: {0, 4}, 1: {1, 5}, 2: {2, 6}, 3: {3, 7}}
        bqm = dimod.OrderedBinaryQuadraticModel.from_ising([.1, .2], {(0, 1): 1.5}, 0.0)

        unembedded = dimod.unembed_response(response, embedding, source_bqm=bqm)

        arr = np.rec.array([([-1,  1], -1.4, 1), ([-1,  1], -1.4, 1), ([+1, -1], -1.6, 1), ([+1, -1], -1.6, 1)],
                           dtype=[('sample', 'i1', (2,)), ('energy', '<f8'), ('num_occurrences', '<i8')])

        np.testing.assert_array_equal(arr, unembedded.record)

    def test_chain_break_statistics_discard(self):
        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        # now set up an embedding that works for one sample and not the other
        embedding = {'a': {0, 1}, 'b': {2}}

        bqm = dimod.BinaryQuadraticModel.from_ising({'a': 1}, {('a', 'b'): -1})

        resp = dimod.Response.from_samples(samples, {'energy': [-1, 1]}, {}, dimod.SPIN)

        resp = dimod.unembed_response(resp, embedding, bqm, chain_break_method=dimod.embedding.discard,
                                      chain_break_fraction=True)

        source_samples = list(resp)

        # only the first sample should be returned
        self.assertEqual(len(source_samples), 1)
        self.assertEqual(source_samples, [{'a': -1, 'b': +1}])

        self.assertEqual(resp.record.chain_break_fraction.sum(), 0)


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

    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_energy_range_embedding(self):
        # start with an Ising
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {(0, 1): 1, (1, 2): 1, (0, 2): 1})

        # convert to BINARY
        bqm.change_vartype(dimod.BINARY, inplace=True)

        # embedding
        embedding = {0: ['a', 'b'], 1: ['c'], 2: ['d']}
        graph = nx.cycle_graph(['a', 'b', 'c', 'd'])
        graph.add_edge('a', 'c')

        # embed
        embedded = dimod.embed_bqm(bqm, embedding, graph, chain_strength=1)

        preferred = dimod.BinaryQuadraticModel({'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0},
                                               {('a', 'c'): 0.5, ('b', 'c'): 0.5, ('c', 'd'): 1.0,
                                                ('a', 'd'): 1.0, ('a', 'b'): -1.0}, 1.0, dimod.SPIN)

        self.assertEqual(embedded, preferred)
