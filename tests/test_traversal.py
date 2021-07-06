# Copyright 2019 D-Wave Systems Inc.
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

import dimod


class TestBFS(unittest.TestCase):
    def test_imbalanced_tree(self):
        bqm = dimod.BQM.from_ising({}, {'ab': -1, 'ac': -1, 'bd': -1})

        bfs = list(dimod.bfs_variables(bqm, 'a'))

        self.assertIn(bfs, [['a', 'b', 'c', 'd'],
                            ['a', 'c', 'b', 'd']])


class TestConnectedComponents(unittest.TestCase):
    def test_connected(self):
        bqm = dimod.BQM.from_ising({}, {'ab': -1, 'ac': -1, 'bd': -1})

        cc = list(dimod.connected_components(bqm))
        self.assertEqual(cc, [set(bqm.variables)])

    def test_disconnected(self):
        bqm = dimod.BQM.from_ising({}, {'ab': -1, 'ac': -1, 'de': -1})

        cc = list(dimod.connected_components(bqm))
        self.assertIn(cc, [[set(['a', 'b', 'c']), set(['d', 'e'])],
                           [set(['d', 'e']), set(['a', 'b', 'c'])]])
