import unittest
import itertools
import random

import networkx as nx
import dwave_networkx as dnx
import dimod

import dwave_embedding_utilities as eutil
from dwave_embedding_utilities import _embedding_to_chain


class TestTargetToSource(unittest.TestCase):
    def test_self_embedding(self):
        """a 1-to-1 embedding should not change the adjacency"""
        target_adj = dnx.chimera_graph(4)
        embedding = {v: {v} for v in target_adj}

        source_adj = eutil.target_to_source(target_adj, embedding)

        # print(source_adj)

        # test the adjacencies are equal (source_adj is a dict and target_adj is a networkx graph)
        for v in target_adj:
            self.assertIn(v, source_adj)
            for u in target_adj[v]:
                self.assertIn(u, source_adj[v])

        for v in source_adj:
            self.assertIn(v, target_adj)
            for u in source_adj[v]:
                self.assertIn(u, target_adj[v])

    def test_embedding_to_one_node(self):
        """an embedding that maps everything to one node should result in a singleton graph"""
        target_adj = nx.barbell_graph(16, 7)
        embedding = {'a': set(target_adj)}  # all map to 'a'

        source_adj = eutil.target_to_source(target_adj, embedding)
        self.assertEqual(source_adj, {'a': set()})

        embedding = {'a': {0, 1}}  # not every node is assigned to a chain
        source_adj = eutil.target_to_source(target_adj, embedding)
        self.assertEqual(source_adj, {'a': set()})

    def test_embedding_overlap(self):
        """overlapping embeddings should raise an error"""
        target_adj = nx.complete_graph(5)
        embedding = {'a': {0, 1}, 'b': {1, 2}}  # overlap

        with self.assertRaises(ValueError):
            source_adj = eutil.target_to_source(target_adj, embedding)


class TestApplyEmbedding(unittest.TestCase):
    def test_embed_ising_components_empty(self):
        h = {}
        J = {}
        embedding = {}
        adjacency = {}

        he, Je, Jc = eutil.embed_ising(h, J, embedding, adjacency)

        self.assertFalse(he)
        self.assertFalse(Je)
        self.assertFalse(Jc)
        self.assertIsInstance(he, dict)
        self.assertIsInstance(Je, dict)
        self.assertIsInstance(Jc, dict)

    def test_embed_bad_chain(self):
        h = {}
        j = {(0, 1): 1}
        embeddings = {0: {0, 2}, 1: {1}}  # (0, 2) not connected
        adj = {0: {1}, 1: {0, 2}, 2: {1}}

        with self.assertRaises(ValueError):
            eutil.embed_ising(h, j, embeddings, adj)

    def test__embedding_to_chain(self):
        """Test that when given a chain, the returned Jc uses all
        available edges."""
        chain_variables = set(range(5))

        # fully connected
        adjacency = {u: set(chain_variables) for u in chain_variables}
        for v, neighbors in adjacency.items():
            neighbors.remove(v)

        Jc = _embedding_to_chain(chain_variables, adjacency, 1.0)

        for u, v in itertools.combinations(chain_variables, 2):
            self.assertFalse((u, v) in Jc and (v, u) in Jc)
            self.assertTrue((u, v) in Jc or (v, u) in Jc)
        for u in chain_variables:
            self.assertFalse((u, u) in Jc)

        # now try a cycle
        adjacency = {v: {(v + 1) % 5, (v - 1) % 5} for v in chain_variables}

        Jc = _embedding_to_chain(chain_variables, adjacency, 1.0)

        for u in adjacency:
            for v in adjacency[u]:
                self.assertFalse((u, v) in Jc and (v, u) in Jc)
                self.assertTrue((u, v) in Jc or (v, u) in Jc)

    def test_embed_nonadj(self):
        """chain in embedding has non-adjacenct nodes"""
        h = {}
        j = {(0, 1): 1}
        embeddings = {0: {0, 1}, 1: {3, 4}}

        adj = nx.Graph()
        adj.add_edges_from({(0, 1), (1, 2), (2, 3), (3, 4)})

        with self.assertRaises(ValueError):
            eutil.embed_ising(h, j, embeddings, adj)

    def test_embed_h_embedding_mismatch(self):
        """there is not a mapping in embedding for every var in h"""
        h = dict(zip(range(3), [1, 2, 3]))
        j = {}
        embedding = {0: (0, 1), 1: (2,)}
        adj = nx.Graph()
        adj.add_edges_from({(0, 1), (1, 2)})

        with self.assertRaises(ValueError):
            eutil.embed_ising(h, j, embedding, adj)

    def test_embed_j_index_too_large(self):
        """j references a variable not mentioned in embedding"""
        h = {}
        j = {(0, 1): -1, (0, 2): 1}
        embedding = {0: (0, 1), 1: (2,)}
        adj = nx.Graph()
        adj.add_edges_from({(0, 1), (1, 2)})

        with self.assertRaises(ValueError):
            eutil.embed_ising(h, j, embedding, adj)

    def test_embed_typical(self):
        h = {0: 1, 1: 10}
        j = {(0, 1): 15, (2, 1): -8, (0, 2): 5, (2, 0): -2}
        embeddings = {0: [1], 1: [2, 3], 2: [0]}

        adj = nx.Graph()
        adj.add_edges_from({(0, 1), (1, 2), (2, 3), (3, 0), (2, 0)})

        expected_h0 = {0: 0, 1: 1, 2: 5, 3: 5}
        expected_j0 = {(0, 1): 3, (0, 2): -4, (0, 3): -4, (1, 2): 15}
        expected_jc = {(2, 3): -1}

        h0, j0, jc = eutil.embed_ising(h, j, embeddings, adj)
        self.assertEqual(h0, expected_h0)

        # check j0
        for (u, v), bias in j0.items():
            self.assertTrue((u, v) in expected_j0 or (v, u) in expected_j0)
            self.assertFalse((u, v) in expected_j0 and (v, u) in expected_j0)

            if (u, v) in expected_j0:
                self.assertEqual(expected_j0[(u, v)], bias)
            else:
                self.assertEqual(expected_j0[(v, u)], bias)

        # check jc
        for (u, v), bias in jc.items():
            self.assertTrue((u, v) in expected_jc or (v, u) in expected_jc)
            self.assertFalse((u, v) in expected_jc and (v, u) in expected_jc)

            if (u, v) in expected_jc:
                self.assertEqual(expected_jc[(u, v)], bias)
            else:
                self.assertEqual(expected_jc[(v, u)], bias)

    def test_embedding_not_in_adj(self):
        """embedding refers to a variable not in the adjacency"""
        h = {0: 0, 1: 0}
        J = {(0, 1): 1}

        embedding = {0: [0, 1], 1: [2]}

        adjacency = {0: {1}, 1: {0}}

        with self.assertRaises(ValueError):
            eutil.embed_ising(h, J, embedding, adjacency)

    def test_docstring_examples(self):
        logical_h = {'a': 1, 'b': 1}
        logical_J = {('a', 'b'): -1}
        embedding = {'a': [0, 1], 'b': [2]}
        adjacency = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        emb_h, emb_J, chain_J = eutil.embed_ising(logical_h, logical_J,
                                                  embedding, adjacency)
        self.assertEqual(emb_h, {0: 0.5, 1: 0.5, 2: 1.0})
        self.assertEqual(emb_J, {(0, 2): -0.5, (1, 2): -0.5})
        self.assertEqual(chain_J, {(0, 1): -1.0})


class TestUnembedding(unittest.TestCase):
    def test_typical(self):
        """simple smoke tests trying to replicate 'normal' usage."""

        nodes = range(10)
        target_samples = [{v: random.choice((-1, 1)) for v in nodes} for __ in range(100)]

        # embedding is map half the target vars to one source, and the other half to
        # another
        embedding = {'a': range(5), 'b': range(5, 10)}

        source_samples = eutil.unembed_samples(target_samples, embedding)

        # not checking correctness of samples, just that they have the correct form
        self.assertEqual(len(source_samples), 100)
        for sample in source_samples:
            self.assertIsInstance(sample, dict)
            self.assertEqual(set(sample), {'a', 'b'})  # maps to source vars
            self.assertTrue(all(bias in (-1, 1) for bias in sample.values()))

    def test_typical_dimod_response(self):
        """unembed should be compatible with the dimod response object"""
        nodes = range(10)
        target_samples = [{v: random.choice((-1, 1)) for v in nodes} for __ in range(100)]

        # load the samples into a dimod response
        response = dimod.SpinResponse()
        response.add_samples_from(target_samples, (0 for __ in target_samples))

        # embedding is map half the target vars to one source, and the other half to
        # another
        embedding = {'a': range(5), 'b': range(5, 10)}

        # use the response as the target_samples
        source_samples = eutil.unembed_samples(response, embedding)

        # not checking correctness of samples, just that they have the correct form
        self.assertEqual(len(source_samples), 100)
        for sample in source_samples:
            self.assertIsInstance(sample, dict)
            self.assertEqual(set(sample), {'a', 'b'})  # maps to source vars
            self.assertTrue(all(bias in (-1, 1) for bias in sample.values()))

    def test_majority_vote(self):
        """should return the most common value in the chain"""

        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        embedding = {'a': {0, 1, 2}}

        # specify that majority vote should be used
        source_samples = eutil.unembed_samples(samples, embedding, chain_break_method=eutil.majority_vote)

        source0, source1 = source_samples

        self.assertEqual(source0['a'], -1)
        self.assertEqual(source1['a'], +1)

    def test_majority_vote_with_dimod(self):
        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        embedding = {'a': {0, 1, 2}}

        # load the samples into a dimod response
        response = dimod.SpinResponse()
        response.add_samples_from(samples, (0 for __ in samples))

        # specify that majority vote should be used
        source_samples = eutil.unembed_samples(response, embedding, chain_break_method=eutil.majority_vote)

        source0, source1 = source_samples

        self.assertEqual(source0['a'], -1)
        self.assertEqual(source1['a'], +1)

    def test_discard(self):
        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        embedding = {'a': {0, 1, 2}}

        # specify that majority vote should be used
        source_samples = eutil.unembed_samples(samples, embedding, chain_break_method=eutil.discard)

        # no samples should be returned because they are all broken
        self.assertEqual(len(source_samples), 0)

        # now set up an embedding that works for one sample and not the other
        embedding = {'a': {0, 1}, 'b': {2}}

        # specify that majority vote should be used
        source_samples = eutil.unembed_samples(samples, embedding, chain_break_method=eutil.discard)

        # only the first sample should be returned
        self.assertEqual(len(source_samples), 1)
        self.assertEqual(source_samples, [{'a': -1, 'b': +1}])

    def test_discard_with_dimod(self):
        sample0 = {0: -1, 1: -1, 2: +1}
        sample1 = {0: +1, 1: -1, 2: +1}
        samples = [sample0, sample1]

        # now set up an embedding that works for one sample and not the other
        embedding = {'a': {0, 1}, 'b': {2}}

        # load the samples into a dimod response
        response = dimod.SpinResponse()
        response.add_samples_from(samples, (0 for __ in samples))

        # specify that majority vote should be used
        source_samples = eutil.unembed_samples(response, embedding, chain_break_method=eutil.discard)

        # only the first sample should be returned
        self.assertEqual(len(source_samples), 1)
        self.assertEqual(source_samples, [{'a': -1, 'b': +1}])

    def test_energy_minimization(self):
        sample0 = {0: -1, 1: -1, 2: +1, 3: +1}
        sample1 = {0: +1, 1: -1, 2: +1, 3: -1}
        samples = [sample0, sample1]

        embedding = {'a': {0, 1}, 'b': {2}, 'c': {3}}

        # minimize energy requires `linear` and `quadratic` keyword args
        with self.assertRaises(TypeError):
            eutil.unembed_samples(samples, embedding, chain_break_method=eutil.minimize_energy)

        linear = {'a': -1, 'b': 0, 'c': 0}
        quadratic = {}

        source_samples = eutil.unembed_samples(samples, embedding, chain_break_method=eutil.minimize_energy,
                                               linear=linear, quadratic=quadratic)

        source0, source1 = source_samples

        # no broken chains
        self.assertEqual(source0, {'a': -1, 'b': +1, 'c': +1})

        # in this case 'a' being spin-up minimizes the energy
        self.assertEqual(source1, {'a': +1, 'b': +1, 'c': -1})

        linear = {'a': 1, 'b': 0, 'c': 0}
        quadratic = {('a', 'b'): -5}

        source_samples = eutil.unembed_samples(samples, embedding, chain_break_method=eutil.minimize_energy,
                                               linear=linear, quadratic=quadratic)

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

        source_samples = eutil.unembed_samples(samples, embedding, chain_break_method=eutil.minimize_energy,
                                               quadratic=quadratic)

        source, = source_samples

        self.assertEqual(source, {'b': -1, 'c': +1, 'a': -1})


class Testedgelist_to_adjacency(unittest.TestCase):
    def test_typical(self):
        graph = nx.barbell_graph(17, 8)

        edgelist = set(graph.edges())

        adj = eutil.edgelist_to_adjacency(edgelist)

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
