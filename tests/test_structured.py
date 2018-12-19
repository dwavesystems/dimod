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

import dimod


class TestStructuredClass(unittest.TestCase):
    def test_instantiation_base_class(self):
        with self.assertRaises(TypeError):
            dimod.Structured()

    def test_adjacency_property(self):
        class Dummy(dimod.Structured):
            @property
            def nodelist(self):
                return list(range(5))

            @property
            def edgelist(self):
                return list(itertools.combinations(self.nodelist, 2))

        sampler = Dummy()

        for u, v in sampler.edgelist:
            self.assertIn(u, sampler.adjacency[v])
            self.assertIn(v, sampler.adjacency[u])
        for u in sampler.adjacency:
            for v in sampler.adjacency[u]:
                self.assertTrue((u, v) in sampler.edgelist or (v, u) in sampler.edgelist)

        # check that we are not rebuilding each time
        self.assertIs(sampler.adjacency, sampler.adjacency)

    def test_structured_property(self):
        class Dummy(dimod.Structured):
            @property
            def nodelist(self):
                return [0, 1]

            @property
            def edgelist(self):
                return [(0, 1)]

        sampler = Dummy()

        self.assertEqual(sampler.structure.nodelist, [0, 1])
        self.assertEqual(sampler.structure.edgelist, [(0, 1)])
        self.assertEqual(sampler.structure.adjacency, {0: {1}, 1: {0}})
