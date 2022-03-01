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

import concurrent.futures
import typing
import unittest
import itertools

from collections.abc import Sequence
from operator import itemgetter

try:
    import networkx as nx
except ImportError:
    _networkx = False
else:
    _networkx = True

import dimod

from dimod.vartypes import SPIN, BINARY
from dimod.decorators import vartype_argument, graph_argument


class IndexSampler(dimod.NullSampler):
    @dimod.decorators.bqm_index_labels
    def sample(self, bqm):
        if isinstance(bqm.variables, Sequence):
            assert bqm.variables == range(bqm.num_variables)
        else:
            # unordered
            assert sorted(bqm.variables) == list(range(bqm.num_variables))
        return super().sample(bqm)


@dimod.testing.load_sampler_bqm_tests(IndexSampler)
class TestBQMIndexLabels(unittest.TestCase):
    pass


class TestNonblockingSampleMethod(unittest.TestCase):
    def test_done(self):
        class Sampler:
            @dimod.decorators.nonblocking_sample_method
            def sample(self, bqm):
                self.future = future = concurrent.futures.Future()
                yield future
                if not future.done():
                    raise Exception('boom')
                sample = {v: 1 for v in bqm.variables}
                yield dimod.SampleSet.from_samples_bqm(sample, bqm)

        sampler = Sampler()

        bqm = dimod.BQM.from_ising({'a': 1}, {('a', 'b'): -2})
        ss = sampler.sample(bqm)
        self.assertFalse(ss.done())
        sampler.future.set_result(0)
        ss.resolve()

    def test_simple(self):

        class Sampler:
            def __init__(self):
                self.first = False
                self.second = False

            @dimod.decorators.nonblocking_sample_method
            def sample(self, bqm):
                self.first = True
                yield
                self.second = True
                sample = {v: 1 for v in bqm.variables}
                yield dimod.SampleSet.from_samples_bqm(sample, bqm)

        sampler = Sampler()

        bqm = dimod.BQM.from_ising({'a': 1}, {('a', 'b'): -2})
        ss = sampler.sample(bqm)

        self.assertIsInstance(ss, dimod.SampleSet)
        self.assertTrue(sampler.first)
        self.assertFalse(sampler.second)

        ss.resolve()

        self.assertTrue(sampler.first)
        self.assertTrue(sampler.second)


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

    def test_edgelist(self):
        @graph_argument('G')
        def f(G):
            return G

        edgelist = [(0, 1), (2, 3), (0, 2)]

        nodes, edges = f(edgelist)

        self.assertEqual(set(nodes), set().union(*edgelist))
        for edge in edges:
            self.assertIn(edge, edgelist)

    def test_edgelist_len2(self):
        # need to distinguish between edgelist and (nodes, edges)
        @graph_argument('G')
        def f(G):
            return G

        edgelist = [(0, 1), (2, 3)]

        nodes, edges = f(edgelist)

        self.assertEqual(set(nodes), set().union(*edgelist))
        for edge in edges:
            self.assertIn(edge, edgelist)

    def test_nodelist_edgelist_len2(self):
        # need to distinguish between edgelist and (nodes, edges)
        @graph_argument('G')
        def f(G):
            return G

        nodelist, edgelist = ([0, 1],
                              [(0, 1), (1, 0)])

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


class TestForwardingMethod(unittest.TestCase):
    def setUp(self):

        class Inner:
            def func(self, a: int, b: int = 0):
                """Inner.func docstring"""
                return a + b

        class Outer:
            def __init__(self):
                self.inner = Inner()
                self.func_count = 0

            @dimod.decorators.forwarding_method
            def func(self, a: int, b: int = 0) -> int:
                """Outer.func docstring"""
                self.func_count += 1
                return self.inner.func

        self.outer = Outer()

    def test_annotation(self):
        outer = self.outer
        self.assertEqual(typing.get_type_hints(outer.func),
                        {'a': int, 'b': int, 'return': int})

    def test_call_count(self):
        outer = self.outer
        for _ in range(10):
            outer.func(1, 5)
        self.assertEqual(outer.func_count, 1)

    def test_doc(self):
        outer = self.outer
        self.assertEqual(outer.func.__doc__, "Outer.func docstring")
        outer.func(2, 3)
        self.assertEqual(outer.func.__doc__, "Inner.func docstring")

    def test_output(self):
        outer = self.outer
        self.assertEqual(outer.func(2, 3), 5)

class EmptyStructuredSampler(dimod.Structured):
    edgelist = nodelist = []

    @dimod.decorators.bqm_structured
    def sample(self, bqm):
        pass

class StructuredSampler(dimod.Structured):
    nodelist = ['a', 'b', 'c']
    edgelist = [('a', 'b')]

    @dimod.decorators.bqm_structured
    def sample(self, bqm):
        pass

class TestBQMStructured(unittest.TestCase):
    def test_empty_sampler_empty_bqm(self):
        EmptyStructuredSampler().sample(dimod.BQM.empty('SPIN'))  # should do nothing

    def test_empty_sampler(self):
        with self.assertRaises(dimod.exceptions.BinaryQuadraticModelStructureError):
            EmptyStructuredSampler().sample(dimod.BQM({'a': 1}, {}, 0.0, 'BINARY'))

    def test_bqm_empty(self):
        StructuredSampler().sample(dimod.BQM('BINARY'))  # should do nothing

    def test_bqm_subset(self):
        StructuredSampler().sample(dimod.BQM({'a': 1}, {'ab': 1}, 0, 'BINARY'))  # should do nothing

    def test_bqm_extra_edge(self):
        with self.assertRaises(dimod.exceptions.BinaryQuadraticModelStructureError):
            StructuredSampler().sample(dimod.BQM({}, {'ca': 1}, 0.0, 'BINARY'))

    def test_bqm_extra_node(self):
        with self.assertRaises(dimod.exceptions.BinaryQuadraticModelStructureError):
            StructuredSampler().sample(dimod.BQM({'a': 1, 'd': 1}, {}, 0.0, 'BINARY'))

    def test_bqm_extra_node_disjoint(self):
        with self.assertRaises(dimod.exceptions.BinaryQuadraticModelStructureError):
            StructuredSampler().sample(dimod.BQM({'d': 1}, {}, 0.0, 'BINARY'))
