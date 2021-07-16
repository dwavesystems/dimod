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

import unittest
import itertools
import random

import dimod
from dimod.reference.samplers.simulated_annealing import ising_simulated_annealing, greedy_coloring
from dimod.exceptions import SamplerUnknownArgWarning

@dimod.testing.load_sampler_bqm_tests(dimod.SimulatedAnnealingSampler)
class TestSASampler(unittest.TestCase):
    def setUp(self):
        self.sampler = dimod.SimulatedAnnealingSampler()
        self.sampler_factory = dimod.SimulatedAnnealingSampler

    def test_basic(self):

        sampler = self.sampler

        h = {0: -.5, 1: 0, 2: 1, 3: -.5}
        J = {(0, 2): -1, (1, 2): -1, (0, 3): .5, (1, 3): -1}

        response0 = sampler.sample_ising(h, J, num_reads=10)

        for sample, energy in response0.data(['sample', 'energy']):
            self.assertEqual(dimod.ising_energy(sample, h, J), energy)

        # make sure we actully got back 100 samples
        self.assertEqual(len(response0), 10)

        Q = {(0, 0): 0, (1, 1): 0, (0, 1): -1}

        response4 = sampler.sample_qubo(Q, num_reads=10)
        self.assertEqual(len(response4), 10)

        for sample, energy in response4.data(['sample', 'energy']):
            self.assertEqual(dimod.qubo_energy(sample, Q), energy)

    def test_bug1(self):
        # IN IN OUT AUX
        h = {0: -.5, 1: 0, 2: 1, 3: -.5}
        J = {(0, 2): -1, (1, 2): -1, (0, 3): .5, (1, 3): -1}

        J[(0, 4)] = -.1
        J[(4, 5)] = -.1
        J[(5, 6)] = -.1
        h[4] = 0
        h[5] = 0
        h[6] = .1

        response = dimod.SimulatedAnnealingSampler().sample_ising(h, J, num_reads=100)

    def test_setting_beta_range(self):
        sampler = self.sampler

        sampler.sample_ising({}, {}, beta_range=(0.1, 1))

    def test_inputchecking(self):
        sampler = self.sampler

        with self.assertRaises(TypeError):
            sampler.sample_ising({}, {}, num_reads=[])

        with self.assertRaises(ValueError):
            sampler.sample_ising({}, {}, num_reads=-7)

        with self.assertRaises(TypeError):
            sampler.sample_ising({}, {}, num_sweeps=[])

        with self.assertRaises(ValueError):
            sampler.sample_ising({}, {}, num_sweeps=-7)

        with self.assertRaises(TypeError):
            sampler.sample_ising({}, {}, beta_range=-7)

        with self.assertRaises(ValueError):
            sampler.sample_ising({}, {}, beta_range=[-7, 6])

        with self.assertRaises(TypeError):
            sampler.sample_ising({}, {}, beta_range=[(0,), 6])

        with self.assertRaises(ValueError):
            sampler.sample_ising({}, {}, beta_range=[7, 1, 6])

    def test_kwargs(self):
        sampler = self.sampler
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
        with self.assertWarns(SamplerUnknownArgWarning):
            sampler.sample(bqm, a=5, b=True)

class TestSimulatedAnnealingAlgorithm(unittest.TestCase):
    def test_ising_simulated_annealing_basic(self):
        # AND gate
        h = {0: -.5, 1: 0, 2: 1, 3: -.5}
        J = {(0, 2): -1, (1, 2): -1, (0, 3): .5, (1, 3): -1}

        sample, energy = ising_simulated_annealing(h, J)

        self.assertIsInstance(sample, dict)
        self.assertIsInstance(energy, float)

        # make sure all of the nodes are present in sample
        for v in range(4):
            self.assertIn(v, sample)

    def test_ising_simulated_annealing_empty_J(self):
        h = {0: -1, 1: 1, 2: -.5}
        J = {}
        sample, energy = ising_simulated_annealing(h, J)

        self.assertIsInstance(sample, dict)
        self.assertIsInstance(energy, float)

        # make sure all of the nodes are present in sample
        for v in range(3):
            self.assertIn(v, sample)

    def test_ising_simulated_annealing_sample_quality(self):
        # because simulated annealing has randomness, we cannot
        # really test that it finds the solution. Instead we
        # note that it should return better-than-average solutions,
        # so if we test the returned energy against the energy of
        # 100 random samples, it should do better than the average
        nV = 100  # number of variables in h,J
        nS = 100  # number of samples

        h = {v: random.uniform(-2, 2) for v in range(nV)}
        J = {}
        for u, v in itertools.combinations(h, 2):
            if random.random() < .05:
                J[(u, v)] = random.uniform(-1, 1)

        random_energies = [dimod.ising_energy({v: random.choice((-1, 1)) for v in h}, h, J)
                           for __ in range(nS)]

        average_energy = sum(random_energies) / float(nS)

        sample, energy = ising_simulated_annealing(h, J)

        self.assertLess(energy, average_energy)

    def test_greedy_coloring(self):
        # set up an adjacency matrix

        N = 100  # number of nodes

        adj = {node: set() for node in range(N)}

        # randomly add approximately 5% of the edges
        for u, v in itertools.combinations(range(N), 2):
            if random.random() < .05:
                adj[u].add(v)
                adj[v].add(u)

        # add one disconnected node
        adj[N] = set()

        # run
        coloring, colors = greedy_coloring(adj)

        # check output types
        self.assertIsInstance(coloring, dict)
        self.assertIsInstance(colors, dict)

        # we want to check that coloring and colors agree
        for v, c in coloring.items():
            self.assertIn(v, colors[c])
        for c, nodes in colors.items():
            for v in nodes:
                self.assertEqual(c, coloring[v])

        # next we want to make sure that no two neighbors share the same color
        for v in adj:
            for u in adj[v]:
                self.assertNotEqual(coloring[u], coloring[v])
