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

from operator import itemgetter

try:
    import networkx as nx
except ImportError:
    _networkx = False
else:
    _networkx = True

from dimod.vartypes import SPIN, BINARY
from dimod.decorators import vartype_argument, graph_argument


class TestVartypeArgument(unittest.TestCase):
    def test_single_explicit_arg(self):
        @vartype_argument('x')
        def f(x):
            return x

        self.assertEqual(f(SPIN), SPIN)
        self.assertEqual(f(BINARY), BINARY)
        self.assertEqual(f('SPIN'), SPIN)
        self.assertEqual(f(x='SPIN'), SPIN)
        self.assertRaises(TypeError, f, 'x')
        self.assertRaises(TypeError, f, x='x')
        self.assertRaises(TypeError, f)

    def test_multiple_explicit_args(self):
        @vartype_argument('x', 'y')
        def f(x, y):
            return x, y

        self.assertEqual(f(SPIN, BINARY), (SPIN, BINARY))
        self.assertEqual(f('SPIN', 'BINARY'), (SPIN, BINARY))
        self.assertEqual(f(y='BINARY', x='SPIN'), (SPIN, BINARY))
        self.assertRaises(TypeError, f, 'x', 'y')
        self.assertRaises(TypeError, f, x='x', y='SPIN')
        self.assertRaises(TypeError, f)

    def test_kwargs(self):
        @vartype_argument('x', 'y')
        def f(**kwargs):
            return itemgetter('x', 'y')(kwargs)

        self.assertEqual(f(x=SPIN, y=BINARY), (SPIN, BINARY))
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, x=SPIN)
        self.assertRaises(TypeError, f, x='x', y='SPIN')

    def test_explicit_with_kwargs(self):
        @vartype_argument('x', 'y')
        def f(x, **kwargs):
            return x, kwargs.pop('y')

        self.assertEqual(f('SPIN', y='BINARY'), (SPIN, BINARY))
        self.assertEqual(f(y='BINARY', x='SPIN'), (SPIN, BINARY))
        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, x=SPIN)
        self.assertRaises(TypeError, f, x='x', y='SPIN')

    def test_default_argname(self):
        @vartype_argument()
        def f(vartype):
            return vartype

        self.assertEqual(f('SPIN'), SPIN)
        self.assertEqual(f(vartype='SPIN'), SPIN)
        self.assertRaises(TypeError, f, 'invalid')
        self.assertRaises(TypeError, f, None)

    def test_argname_mismatch(self):
        @vartype_argument()
        def f(x=None):
            return x

        self.assertRaises(TypeError, f)
        self.assertRaises(TypeError, f, x=SPIN)

        @vartype_argument('vartype')
        def g(x=None):
            return x

        self.assertRaises(TypeError, g)
        self.assertRaises(TypeError, g, x=SPIN)
        self.assertRaises(TypeError, g, vartype=SPIN)

    def test_arg_with_default_value(self):
        @vartype_argument('vartype')
        def f(vartype='SPIN'):
            return vartype

        self.assertEqual(f(), SPIN)
        self.assertEqual(f('BINARY'), BINARY)
        self.assertEqual(f(vartype='BINARY'), BINARY)


class TestGraphArgument(unittest.TestCase):
    @unittest.skipUnless(_networkx, "no networkx installed")
    def test_networkx_graph(self):
        @graph_argument('G')
        def f(G):
            return G

        G = nx.complete_graph(3)

        nodes, edges = f(G)

        for n in nodes:
            self.assertIn(n, G.nodes)
        for edge in edges:
            self.assertIn(edge, G.edges)

    def test_nodelist_edgelist(self):
        @graph_argument('G')
        def f(G):
            return G

        nodelist, edgelist = ([0, 1, 2, 3, 4],
                              [(0, 1), (2, 3), (0, 2)])

        nodes, edges = f((nodelist, edgelist))

        self.assertIs(nodes, nodelist)
        self.assertIs(edges, edgelist)

        for n in nodes:
            self.assertIn(n, nodelist)
        for edge in edges:
            self.assertIn(edge, edgelist)

    def test_complete_number(self):
        @graph_argument('G')
        def f(G):
            return G

        nodes, edges = f(5)

        self.assertEqual(nodes, list(range(5)))
        for u, v in itertools.combinations(range(5), 2):
            self.assertTrue((u, v) in edges or (v, u) in edges)

    def test_nodelist_edgelist(self):
        @graph_argument('G')
        def f(G):
            return G

        N, edgelist = (5,
                       [(0, 1), (2, 3), (0, 2)])

        nodes, edges = f((N, edgelist))

        self.assertIs(edges, edgelist)

        for n in nodes:
            self.assertIn(n, range(N))
        for edge in edges:
            self.assertIn(edge, edgelist)

    def test_allow_None_False(self):
        @graph_argument('G')
        def f(G=None):
            return G

        with self.assertRaises(ValueError):
            f()

    def test_allow_None_True(self):

        @graph_argument('G', allow_None=True)
        def f(G=None):
            return G

        self.assertIs(f(), None)

    def test_other_kwarg(self):
        with self.assertRaises(TypeError):
            @graph_argument('G', b=True)
            def f(G):
                pass
