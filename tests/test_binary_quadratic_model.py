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

import copy
import collections
import fractions
import itertools
import json
import pickle
import random
import shutil
import tempfile
import unittest

from os import path

import numpy as np

import dimod

from dimod.exceptions import WriteableError

try:
    import networkx as nx
    _networkx = True
except ImportError:
    _networkx = False

try:
    import pandas as pd
    _pandas = True
except ImportError:
    _pandas = False


class TestBinaryQuadraticModel(unittest.TestCase):

    def assertConsistentBQM(self, bqm):
        # adjacency and linear are self-consistent
        for v in bqm.linear:
            self.assertIn(v, bqm.adj)
        for v in bqm.adj:
            self.assertIn(v, bqm.linear)

        # adjacency and quadratic are self-consistent
        for u, v in bqm.quadratic:
            self.assertIn(v, bqm.linear)
            self.assertIn(v, bqm.adj)
            self.assertIn(u, bqm.adj[v])

            self.assertIn(u, bqm.linear)
            self.assertIn(u, bqm.adj)
            self.assertIn(v, bqm.adj[u])

            self.assertEqual(bqm.adj[u][v], bqm.quadratic[(u, v)])
            self.assertEqual(bqm.adj[v][u], bqm.adj[u][v])

            # self.assertNotIn((v, u), bqm.quadratic)

        for u in bqm.adj:
            for v in bqm.adj[u]:
                self.assertTrue((u, v) in bqm.quadratic and (v, u) in bqm.quadratic)
                # self.assertFalse((u, v) in bqm.quadratic and (v, u) in bqm.quadratic)

        self.assertEqual(len(bqm.quadratic), len(set(bqm.quadratic)))
        self.assertEqual(len(bqm.variables), len(bqm.linear))

    def test_construction(self):
        # spin model
        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = 1.4
        vartype = dimod.SPIN
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        self.assertConsistentBQM(bqm)

        for v, bias in linear.items():
            self.assertEqual(bqm.linear[v], bias)
        for v in bqm.linear:
            self.assertIn(v, linear)

        for (u, v), bias in quadratic.items():
            self.assertEqual(bqm.adj[u][v], bias)
        for interaction in bqm.quadratic:
            self.assertIn(interaction, quadratic)

        self.assertEqual(bqm.offset, offset)

        #

        # binary model
        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = 1.4
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        self.assertConsistentBQM(bqm)

        for v, bias in linear.items():
            self.assertEqual(bqm.linear[v], bias)
        for v in bqm.linear:
            self.assertIn(v, linear)

        for (u, v), bias in quadratic.items():
            self.assertEqual(bqm.adj[u][v], bias)
        for interaction in bqm.quadratic:
            self.assertIn(interaction, quadratic)

        self.assertEqual(bqm.offset, offset)

    def test_construction_vartype(self):
        """Check that exceptions get thrown for broken inputs"""

        # this biases values are themselves not important, so just choose them randomly
        linear = {v: v * .01 for v in range(10)}
        quadratic = {(u, v): u * v * .01 for u, v in itertools.combinations(linear, 2)}
        offset = 1.2

        with self.assertRaises(TypeError):
            dimod.BinaryQuadraticModel(linear, quadratic, offset, 147)

        with self.assertRaises(TypeError):
            dimod.BinaryQuadraticModel(linear, quadratic, offset, 'my made up type')

        self.assertEqual(dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY).vartype, dimod.BINARY)

        self.assertEqual(dimod.BinaryQuadraticModel(linear, quadratic, offset, {-1, 1}).vartype, dimod.SPIN)

        self.assertEqual(dimod.BinaryQuadraticModel(linear, quadratic, offset, 'BINARY').vartype, dimod.BINARY)

    def test__eq__(self):
        linear = {v: v * -.13 for v in range(10)}
        quadratic = {(u, v): u * v * .021 for u, v in itertools.combinations(linear, 2)}
        offset = 1.2
        vartype = dimod.BINARY

        self.assertEqual(dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype),
                         dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype))

        # mismatched type
        self.assertNotEqual(dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype), -1)

        # models of different type
        self.assertNotEqual(dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN),
                            dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY))

        #

        linear = {v: v * -.13 for v in range(10)}
        quadratic = {(u, v): u * v * .021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.SPIN

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        reversed_quadratic = {(v, u): bias for (u, v), bias in quadratic.items()}

        reversed_bqm = dimod.BinaryQuadraticModel(linear, reversed_quadratic, offset, vartype)

        self.assertEqual(bqm, reversed_bqm)

    def test__repr__(self):
        bqm = dimod.BinaryQuadraticModel({0: 1, 1: -1}, {(0, 1): .5, (1, 2): 1.5}, 1.4, dimod.SPIN)

        # should recreate the model
        from dimod import BinaryQuadraticModel, Vartype
        new_bqm = eval(bqm.__repr__())

        self.assertEqual(bqm, new_bqm)

    def test__len__(self):
        linear = {v: v * -.13 for v in range(10)}
        quadratic = {}
        offset = -1.2
        vartype = dimod.SPIN
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        self.assertEqual(len(bqm), len(linear))

    def test__contains__(self):
        bqm = dimod.BinaryQuadraticModel({'a': -1}, {}, 0.0, dimod.SPIN)

        with self.assertWarns(DeprecationWarning):
            self.assertIn('a', bqm)
        with self.assertWarns(DeprecationWarning):
            self.assertNotIn('b', bqm)

        bqm.add_interaction('a', 'b', .5)

        with self.assertWarns(DeprecationWarning):
            self.assertIn('b', bqm)

    def test__iter__(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(set(bqm), set())

        bqm.add_interaction('a', 'b', -1)

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(set(bqm), {'a', 'b'})

    def test_variables(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        self.assertEqual(set(bqm.variables), set())

        bqm.add_interaction('a', 'b', -1)

        self.assertEqual(set(bqm.variables), {'a', 'b'})
        self.assertIn('a', bqm.variables)
        self.assertEqual(bqm.variables & {'b'}, {'b'})
        self.assertEqual(bqm.variables | {'c'}, {'a', 'b', 'c'})

    def test_add_variables_from(self):
        linear = {'a': .5, 'b': -.5}
        offset = 0.0
        vartype = dimod.SPIN

        # create an empty model then add linear
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, vartype)
        bqm.add_variables_from(linear)

        self.assertEqual(bqm, dimod.BinaryQuadraticModel(linear, {}, 0.0, vartype))

        # add from 2-tuples
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, vartype)
        bqm.add_variables_from((key, value) for key, value in linear.items())

        self.assertEqual(bqm, dimod.BinaryQuadraticModel(linear, {}, 0.0, vartype))

        with self.assertRaises(TypeError):
            bqm.add_variables_from(1)

    def test_add_interactions_from(self):
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
        bqm.add_interactions_from({('a', 'b'): -.5})
        self.assertEqual(bqm, dimod.BinaryQuadraticModel({}, {('a', 'b'): -.5}, 0.0, dimod.SPIN))

        bqm.add_interactions_from([('a', 'b', -.5), ('a', 'b', -.5)])
        self.assertEqual(bqm.adj['a']['b'], -1.5)
        self.assertConsistentBQM(bqm)

    def test_remove_variable(self):
        linear = {v: v * -.13 for v in range(10)}
        quadratic = {(u, v): u * v * .021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.SPIN
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        bqm.remove_variable(2)

        self.assertNotIn(2, bqm.linear)
        self.assertConsistentBQM(bqm)  # checks the other stuff

        self.assertNotIn(2, bqm.binary.linear)
        self.assertConsistentBQM(bqm.binary)

        self.assertAlmostEqual(bqm.binary.energy({v: v % 2 for v in bqm.linear}),
                               bqm.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

        #

        linear = {v: v * -.13 for v in range(10)}
        quadratic = {(u, v): u * v * .021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        bqm.remove_variable(2)

        self.assertNotIn(2, bqm.linear)
        self.assertConsistentBQM(bqm)  # checks the other stuff

        self.assertNotIn(2, bqm.spin.linear)
        self.assertConsistentBQM(bqm.spin)

        self.assertAlmostEqual(bqm.energy({v: v % 2 for v in bqm.linear}),
                               bqm.spin.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

        #

        linear = {v: v * -.13 for v in range(10)}
        quadratic = {(u, v): u * v * .021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.SPIN
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
        __ = bqm.binary  # create counterpart

        bqm.remove_variable(2)

        self.assertNotIn(2, bqm.linear)
        self.assertConsistentBQM(bqm)  # checks the other stuff

        self.assertNotIn(2, bqm.binary.linear)
        self.assertConsistentBQM(bqm.binary)

        self.assertAlmostEqual(bqm.binary.energy({v: v % 2 for v in bqm.linear}),
                               bqm.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

        #

        linear = {v: v * -.13 for v in range(10)}
        quadratic = {(u, v): u * v * .021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
        __ = bqm.spin  # create counterpart

        bqm.remove_variable(2)

        self.assertNotIn(2, bqm.linear)
        self.assertConsistentBQM(bqm)  # checks the other stuff

        self.assertNotIn(2, bqm.spin.linear)
        self.assertConsistentBQM(bqm.spin)

        self.assertAlmostEqual(bqm.energy({v: v % 2 for v in bqm.linear}),
                               bqm.spin.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

    def test_remove_variables_from(self):
        linear = {v: v * -.13 for v in range(10)}
        quadratic = {(u, v): u * v * .021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        bqm.remove_variables_from([0, 1])

        self.assertNotIn(0, bqm.linear)
        self.assertNotIn(1, bqm.linear)
        self.assertConsistentBQM(bqm)

    def test_remove_interaction(self):
        linear = {v: v * -.13 for v in range(10)}
        quadratic = {(u, v): u * v * .021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        bqm.remove_interaction(1, 4)

        self.assertNotIn((1, 4), bqm.quadratic)
        self.assertNotIn((4, 1), bqm.quadratic)
        self.assertConsistentBQM(bqm)

        bqm.remove_interaction(5, 3)

        self.assertNotIn((5, 3), bqm.quadratic)
        self.assertNotIn((3, 5), bqm.quadratic)
        self.assertConsistentBQM(bqm)

        # remove all interactions
        interactions = list(bqm.quadratic)
        for u, v in interactions:
            bqm.remove_interaction(u, v)
        for v in range(10):
            self.assertIn(v, bqm.linear)  # all variables should still be there
        self.assertFalse(len(bqm.quadratic))
        self.assertConsistentBQM(bqm)

        #

        linear = {v: v * -.13 for v in range(10)}
        quadratic = {(u, v): u * v * .021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.SPIN
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        bqm.remove_interaction(1, 4)

        self.assertAlmostEqual(bqm.binary.energy({v: v % 2 for v in bqm.linear}),
                               bqm.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

        #

        linear = {v: v * -.13 for v in range(10)}
        quadratic = {(u, v): u * v * .021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.SPIN
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
        __ = bqm.binary  # create counterpart

        bqm.remove_interaction(1, 4)

        assert len(linear) == 10  # should not effect linear
        self.assertAlmostEqual(bqm.binary.energy({v: v % 2 for v in bqm.linear}),
                               bqm.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

        #

        linear = {v: v * -.13 for v in range(10)}
        quadratic = {(u, v): u * v * .021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        bqm.remove_interaction(1, 4)

        self.assertAlmostEqual(bqm.energy({v: v % 2 for v in bqm.linear}),
                               bqm.spin.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

        #

        linear = {v: v * -.13 for v in range(10)}
        quadratic = {(u, v): u * v * .021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
        __ = bqm.spin  # create counterpart

        bqm.remove_interaction(1, 4)

        assert len(linear) == 10  # should not effect linear
        self.assertAlmostEqual(bqm.energy({v: v % 2 for v in bqm.linear}),
                               bqm.spin.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

    def test_remove_interactions_from(self):
        linear = {v: v * -.43 for v in range(10)}
        quadratic = {(u, v): u * v * -.021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        bqm.remove_interactions_from(quadratic.copy())
        for v in range(10):
            self.assertIn(v, bqm.linear)  # all variables should still be there
        self.assertFalse(len(bqm.quadratic))
        self.assertConsistentBQM(bqm)

    def test_add_offset(self):
        bqm = dimod.BinaryQuadraticModel({0: 1.4, 1: -1}, {(1, 0): 1}, 1.2, dimod.SPIN)
        bqm.add_offset(-1.3)
        self.assertAlmostEqual(bqm.offset, -.1)
        self.assertAlmostEqual(bqm.binary.energy({v: v % 2 for v in bqm.linear}),
                               bqm.spin.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

        #

        bqm = dimod.BinaryQuadraticModel({0: 1.4, 1: -1}, {(1, 0): 1}, 1.2, dimod.SPIN)
        __ = bqm.binary
        bqm.add_offset(-1.3)
        self.assertAlmostEqual(bqm.offset, -.1)

        self.assertAlmostEqual(bqm.binary.energy({v: v % 2 for v in bqm.linear}),
                               bqm.spin.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

    def test_remove_offset(self):
        linear = {v: v * -.43 for v in range(10)}
        quadratic = {(u, v): u * v * -.021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        bqm.remove_offset()

        self.assertAlmostEqual(bqm.offset, 0.0)

    def test_scale(self):
        bqm = dimod.BinaryQuadraticModel({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.scale(.5)
        self.assertAlmostEqual(bqm.linear, {0: -1., 1: 1.})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -.5})
        self.assertAlmostEqual(bqm.offset, .5)
        self.assertConsistentBQM(bqm)

        self.assertAlmostEqual(bqm.binary.energy({v: v % 2 for v in bqm.linear}),
                               bqm.spin.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

        #

        bqm = dimod.BinaryQuadraticModel({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        __ = bqm.binary
        bqm.scale(.5)
        self.assertAlmostEqual(bqm.linear, {0: -1., 1: 1.})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -.5})
        self.assertAlmostEqual(bqm.offset, .5)
        self.assertConsistentBQM(bqm)

        self.assertAlmostEqual(bqm.binary.energy({v: v % 2 for v in bqm.linear}),
                               bqm.spin.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

        #

        with self.assertRaises(TypeError):
            bqm.scale('a')

    def test_normalize(self):
        bqm = dimod.BinaryQuadraticModel({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.normalize(.5)
        self.assertAlmostEqual(bqm.linear, {0: -.5, 1: .5})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -.25})
        self.assertAlmostEqual(bqm.offset, .25)
        self.assertConsistentBQM(bqm)

        self.assertAlmostEqual(bqm.binary.energy({v: v % 2 for v in bqm.linear}),
                               bqm.spin.energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

        #

        with self.assertRaises(TypeError):
            bqm.scale('a')

    def test_normalize_exclusions(self):
        bqm = dimod.BinaryQuadraticModel({0: -2, 1: 2}, {(0, 1): -1}, 1.,
                                         dimod.SPIN)
        bqm.normalize(.5, ignored_variables=[0])
        self.assertAlmostEqual(bqm.linear, {0: -2, 1: .5})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -.25})
        self.assertAlmostEqual(bqm.offset, .25)
        self.assertConsistentBQM(bqm)

        bqm = dimod.BinaryQuadraticModel({0: -2, 1: 2}, {(0, 1): -1}, 1.,
                                         dimod.SPIN)
        bqm.normalize(.5, ignored_interactions=[(1, 0)])
        self.assertAlmostEqual(bqm.linear, {0: -.5, 1: .5})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -1})
        self.assertAlmostEqual(bqm.offset, .25)
        self.assertConsistentBQM(bqm)

        bqm = dimod.BinaryQuadraticModel({0: -2, 1: 2}, {(0, 1): -1}, 1.,
                                         dimod.SPIN)
        bqm.normalize(.5, ignore_offset=True)
        self.assertAlmostEqual(bqm.linear, {0: -.5, 1: .5})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -.25})
        self.assertAlmostEqual(bqm.offset, 1.)
        self.assertConsistentBQM(bqm)

        bqm = dimod.BinaryQuadraticModel({0: -2, 1: 2}, {(0, 1): -5}, 1.,
                                         dimod.SPIN)
        bqm.normalize(0.5, ignored_interactions=[(0, 1)])
        self.assertAlmostEqual(bqm.linear, {0: -.5, 1: .5})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -5})
        self.assertAlmostEqual(bqm.offset, 0.25)
        self.assertConsistentBQM(bqm)

    def test_scale_exclusions(self):
        bqm = dimod.BinaryQuadraticModel({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.scale(.5, ignored_variables=[0])
        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm, dimod.BinaryQuadraticModel({0: -2, 1: 1}, {(0, 1): -.5}, .5, dimod.SPIN))

        bqm = dimod.BinaryQuadraticModel({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.scale(.5, ignored_interactions=[(1, 0)])
        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm, dimod.BinaryQuadraticModel({0: -1, 1: 1}, {(0, 1): -1.}, .5, dimod.SPIN))

        bqm = dimod.BinaryQuadraticModel({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.scale(.5, ignore_offset=True)
        self.assertConsistentBQM(bqm)
        self.assertEqual(bqm, dimod.BinaryQuadraticModel({0: -1, 1: 1},
                                                         {(0, 1): -.5},
                                                         1., dimod.SPIN))

    def test_fix_variables(self):
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        bqm.fix_variables({'a': -1, 'b': +1})


    def test_fix_variable(self):
        # spin model, fix variable to +1
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        bqm.fix_variable('a', +1)

        self.assertAlmostEqual(bqm.energy({'b': +1}), original_bqm.energy({'b': +1, 'a': +1}))
        self.assertAlmostEqual(bqm.energy({'b': -1}), original_bqm.energy({'b': -1, 'a': +1}))
        self.assertAlmostEqual(bqm.binary.energy({'b': 1}), original_bqm.binary.energy({'b': 1, 'a': 1}))
        self.assertAlmostEqual(bqm.binary.energy({'b': 0}), original_bqm.binary.energy({'b': 0, 'a': 1}))

        #

        # spin model with binary built, fix variable to +1
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        __ = bqm.binary  # create the binary version
        bqm.fix_variable('a', +1)

        self.assertAlmostEqual(bqm.energy({'b': +1}), original_bqm.energy({'b': +1, 'a': +1}))
        self.assertAlmostEqual(bqm.energy({'b': -1}), original_bqm.energy({'b': -1, 'a': +1}))
        self.assertAlmostEqual(bqm.binary.energy({'b': 1}), original_bqm.binary.energy({'b': 1, 'a': 1}))
        self.assertAlmostEqual(bqm.binary.energy({'b': 0}), original_bqm.binary.energy({'b': 0, 'a': 1}))

        #

        # spin model, fix variable to -1
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        bqm.fix_variable('a', -1)

        self.assertAlmostEqual(bqm.energy({'b': +1}), original_bqm.energy({'b': +1, 'a': -1}))
        self.assertAlmostEqual(bqm.energy({'b': -1}), original_bqm.energy({'b': -1, 'a': -1}))
        self.assertAlmostEqual(bqm.binary.energy({'b': 1}), original_bqm.binary.energy({'b': 1, 'a': 0}))
        self.assertAlmostEqual(bqm.binary.energy({'b': 0}), original_bqm.binary.energy({'b': 0, 'a': 0}))

        #

        # spin model with binary built, fix variable to -1
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        __ = bqm.binary  # create the binary version
        bqm.fix_variable('a', -1)

        self.assertAlmostEqual(bqm.energy({'b': +1}), original_bqm.energy({'b': +1, 'a': -1}))
        self.assertAlmostEqual(bqm.energy({'b': -1}), original_bqm.energy({'b': -1, 'a': -1}))
        self.assertAlmostEqual(bqm.binary.energy({'b': 1}), original_bqm.binary.energy({'b': 1, 'a': 0}))
        self.assertAlmostEqual(bqm.binary.energy({'b': 0}), original_bqm.binary.energy({'b': 0, 'a': 0}))

        #

        # binary model, fix variable to +1
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        bqm.fix_variable('a', 1)

        self.assertAlmostEqual(bqm.spin.energy({'b': +1}), original_bqm.energy({'b': 1, 'a': 1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': -1}), original_bqm.energy({'b': 0, 'a': 1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': +1}), original_bqm.spin.energy({'b': +1, 'a': +1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': -1}), original_bqm.spin.energy({'b': -1, 'a': +1}))

        #

        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        __ = bqm.spin
        bqm.fix_variable('a', 1)

        self.assertAlmostEqual(bqm.spin.energy({'b': +1}), original_bqm.energy({'b': 1, 'a': 1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': -1}), original_bqm.energy({'b': 0, 'a': 1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': +1}), original_bqm.spin.energy({'b': +1, 'a': +1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': -1}), original_bqm.spin.energy({'b': -1, 'a': +1}))

        #

        # binary model, fix variable to 0
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        bqm.fix_variable('a', 0)

        self.assertAlmostEqual(bqm.spin.energy({'b': +1}), original_bqm.energy({'b': 1, 'a': 0}))
        self.assertAlmostEqual(bqm.spin.energy({'b': -1}), original_bqm.energy({'b': 0, 'a': 0}))
        self.assertAlmostEqual(bqm.spin.energy({'b': +1}), original_bqm.spin.energy({'b': +1, 'a': -1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': -1}), original_bqm.spin.energy({'b': -1, 'a': -1}))

        #

        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        __ = bqm.spin
        bqm.fix_variable('a', 0)

        self.assertAlmostEqual(bqm.spin.energy({'b': +1}), original_bqm.energy({'b': 1, 'a': 0}))
        self.assertAlmostEqual(bqm.spin.energy({'b': -1}), original_bqm.energy({'b': 0, 'a': 0}))
        self.assertAlmostEqual(bqm.spin.energy({'b': +1}), original_bqm.spin.energy({'b': +1, 'a': -1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': -1}), original_bqm.spin.energy({'b': -1, 'a': -1}))

        #

        with self.assertRaises(ValueError):
            bqm.fix_variable('b', -1)  # spin for binary

    def test_update(self):
        binary_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 0, dimod.BINARY)
        spin_bqm = dimod.BinaryQuadraticModel({'c': -1}, {('b', 'c'): 1}, 1.2, dimod.SPIN)

        binary_bqm.update(spin_bqm)

        # binary contribution is 0.0
        self.assertAlmostEqual(binary_bqm.energy({'a': 0, 'b': 0, 'c': 0}), spin_bqm.energy({'c': -1, 'b': -1}))

    def test_contract_variables(self):
        bqm = dimod.BinaryQuadraticModel({'a': .3, 'b': -.7}, {('a', 'b'): -1, ('b', 'c'): 1}, 1.2, dimod.BINARY)
        original_bqm = bqm.copy()

        bqm.contract_variables('a', 'b')
        self.assertNotIn('b', bqm.linear)
        self.assertConsistentBQM(bqm)

        self.assertAlmostEqual(bqm.energy({'a': 0, 'c': 1}), original_bqm.energy({'a': 0, 'b': 0, 'c': 1}))
        self.assertAlmostEqual(bqm.energy({'a': 1, 'c': 1}), original_bqm.energy({'a': 1, 'b': 1, 'c': 1}))
        self.assertAlmostEqual(bqm.energy({'a': 0, 'c': 0}), original_bqm.energy({'a': 0, 'b': 0, 'c': 0}))
        self.assertAlmostEqual(bqm.energy({'a': 1, 'c': 0}), original_bqm.energy({'a': 1, 'b': 1, 'c': 0}))

        #

        bqm = dimod.BinaryQuadraticModel({'a': .3, 'b': -.7}, {('a', 'b'): -1, ('b', 'c'): 1}, 1.2, dimod.SPIN)
        original_bqm = bqm.copy()

        bqm.contract_variables('a', 'b')
        self.assertNotIn('b', bqm.linear)
        self.assertConsistentBQM(bqm)

        self.assertAlmostEqual(bqm.energy({'a': -1, 'c': +1}), original_bqm.energy({'a': -1, 'b': -1, 'c': +1}))
        self.assertAlmostEqual(bqm.energy({'a': +1, 'c': +1}), original_bqm.energy({'a': +1, 'b': +1, 'c': +1}))
        self.assertAlmostEqual(bqm.energy({'a': -1, 'c': -1}), original_bqm.energy({'a': -1, 'b': -1, 'c': -1}))
        self.assertAlmostEqual(bqm.energy({'a': +1, 'c': -1}), original_bqm.energy({'a': +1, 'b': +1, 'c': -1}))

        #

        with self.assertRaises(ValueError):
            bqm.contract_variables(0, 1)

        with self.assertRaises(ValueError):
            bqm.contract_variables('a', 1)

    def test_relabel_typical(self):
        linear = {0: .5, 1: 1.3}
        quadratic = {(0, 1): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        mapping = {0: 'a', 1: 'b'}
        newmodel = model.relabel_variables(mapping, inplace=False)

        # check that new model is the same as old model
        linear = {'a': .5, 'b': 1.3}
        quadratic = {('a', 'b'): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        testmodel = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        self.assertEqual(newmodel, testmodel)

    def test_relabel_typical_copy(self):
        linear = {0: .5, 1: 1.3}
        quadratic = {(0, 1): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        mapping = {0: 'a', 1: 'b'}
        newmodel = model.relabel_variables(mapping, inplace=False)
        self.assertIsNot(model, newmodel)
        self.assertIsNot(model.linear, newmodel.linear)
        self.assertIsNot(model.quadratic, newmodel.quadratic)

        # check that new model is the same as old model
        linear = {'a': .5, 'b': 1.3}
        quadratic = {('a', 'b'): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        testmodel = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        self.assertEqual(newmodel, testmodel)

    def test_relabel_typical_inplace(self):
        linear = {0: .5, 1: 1.3}
        quadratic = {(0, 1): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        mapping = {0: 'a', 1: 'b'}
        newmodel = model.relabel_variables(mapping)
        self.assertIs(model, newmodel)

        # check that model is the same as old model
        linear = {'a': .5, 'b': 1.3}
        quadratic = {('a', 'b'): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        testmodel = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
        self.assertEqual(model, testmodel)

        self.assertEqual(model.adj, testmodel.adj)

    def test_relabel_with_overlap(self):
        linear = {v: .1 * v for v in range(-5, 4)}
        quadratic = {(u, v): .1 * u * v for u, v in itertools.combinations(linear, 2)}
        offset = 1.2
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        partial_overlap_mapping = {v: -v for v in linear}  # has variables mapped to other old labels

        # construct a test model by using copy
        testmodel = model.relabel_variables(partial_overlap_mapping, inplace=False)

        # now apply in place
        model.relabel_variables(partial_overlap_mapping, inplace=True)

        # should have stayed the same
        self.assertEqual(testmodel, model)
        self.assertEqual(testmodel.adj, model.adj)

    def test_relabel_with_identity(self):
        linear = {v: .1 * v for v in range(-5, 4)}
        quadratic = {(u, v): .1 * u * v for u, v in itertools.combinations(linear, 2)}
        offset = 1.2
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
        old_model = model.copy()

        identity_mapping = {v: v for v in linear}

        model.relabel_variables(identity_mapping, inplace=True)

        # should have stayed the same
        self.assertEqual(old_model, model)
        self.assertEqual(old_model.adj, model.adj)

    def test_partial_relabel_copy(self):
        linear = {v: .1 * v for v in range(-5, 5)}
        quadratic = {(u, v): .1 * u * v for u, v in itertools.combinations(linear, 2)}
        offset = 1.2
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        mapping = {0: 'a', 1: 'b'}  # partial mapping
        newmodel = model.relabel_variables(mapping, inplace=False)

        newlinear = linear.copy()
        newlinear['a'] = newlinear[0]
        newlinear['b'] = newlinear[1]
        del newlinear[0]
        del newlinear[1]

        self.assertEqual(newlinear, newmodel.linear)

    def test_partial_relabel_inplace(self):
        linear = {v: .1 * v for v in range(-5, 5)}
        quadratic = {(u, v): .1 * u * v for u, v in itertools.combinations(linear, 2)}
        offset = 1.2
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        newlinear = linear.copy()
        newlinear['a'] = newlinear[0]
        newlinear['b'] = newlinear[1]
        del newlinear[0]
        del newlinear[1]

        mapping = {0: 'a', 1: 'b'}  # partial mapping
        model.relabel_variables(mapping, inplace=True)

        self.assertEqual(newlinear, model.linear)

    def test_copy(self):
        linear = {0: .5, 1: 1.3}
        quadratic = {(0, 1): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        new_bqm = bqm.copy()

        # everything should have a new id
        self.assertIsNot(bqm.linear, new_bqm.linear)
        self.assertIsNot(bqm.quadratic, new_bqm.quadratic)
        self.assertIsNot(bqm.adj, new_bqm.adj)

        for v in bqm.linear:
            self.assertIsNot(bqm.adj[v], new_bqm.adj[v])

        # values should all be equal
        self.assertEqual(bqm.linear, new_bqm.linear)
        self.assertEqual(bqm.quadratic, new_bqm.quadratic)
        self.assertEqual(bqm.adj, new_bqm.adj)

        for v in bqm.linear:
            self.assertEqual(bqm.adj[v], new_bqm.adj[v])

        self.assertEqual(bqm, new_bqm)

    def test_change_vartype(self):

        #
        # copy
        #

        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = 1.4
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        # should not change
        new_model = bqm.change_vartype(dimod.BINARY, inplace=False)
        self.assertEqual(bqm, new_model)
        self.assertNotEqual(id(bqm), id(new_model))

        # change vartype
        new_model = bqm.change_vartype(dimod.SPIN, inplace=False)

        # check all of the energies
        for spins in itertools.product((-1, 1), repeat=len(linear)):
            spin_sample = {v: spins[v] for v in linear}
            binary_sample = {v: (spins[v] + 1) // 2 for v in linear}

            self.assertAlmostEqual(bqm.energy(binary_sample),
                                   new_model.energy(spin_sample))

        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = -1.4
        vartype = dimod.SPIN
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        # should not change
        new_model = bqm.change_vartype(dimod.SPIN, inplace=False)
        self.assertEqual(bqm, new_model)
        self.assertNotEqual(id(bqm), id(new_model))

        # change vartype
        new_model = bqm.change_vartype(dimod.BINARY, inplace=False)

        # check all of the energies
        for spins in itertools.product((-1, 1), repeat=len(linear)):
            spin_sample = {v: spins[v] for v in linear}
            binary_sample = {v: (spins[v] + 1) // 2 for v in linear}

            self.assertAlmostEqual(bqm.energy(spin_sample),
                                   new_model.energy(binary_sample))

        with self.assertRaises(TypeError):
            bqm.change_vartype('BOOLEAN')

        #
        # in place
        #

        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = 1.4
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
        original_bqm = bqm.copy()

        # should not change
        new_model = bqm.change_vartype(dimod.BINARY)
        self.assertEqual(bqm, new_model)
        self.assertEqual(id(bqm), id(new_model))

        new_model = bqm.change_vartype(dimod.BINARY, inplace=True)
        self.assertEqual(bqm, new_model)
        self.assertEqual(id(bqm), id(new_model))

        # change vartype
        bqm.change_vartype(dimod.SPIN)

        # check all of the energies
        for spins in itertools.product((-1, 1), repeat=len(linear)):
            spin_sample = {v: spins[v] for v in linear}
            binary_sample = {v: (spins[v] + 1) // 2 for v in linear}

            self.assertAlmostEqual(bqm.energy(spin_sample),
                                   original_bqm.energy(binary_sample))

        # SPIN model
        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = -1.4
        vartype = dimod.SPIN
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
        original_bqm = bqm.copy()

        # should not change
        new_model = bqm.change_vartype(dimod.SPIN)
        self.assertEqual(bqm, new_model)
        self.assertEqual(id(bqm), id(new_model))

        # change vartype
        bqm.change_vartype(dimod.BINARY)

        # check all of the energies
        for spins in itertools.product((-1, 1), repeat=len(linear)):
            spin_sample = {v: spins[v] for v in linear}
            binary_sample = {v: (spins[v] + 1) // 2 for v in linear}

            self.assertAlmostEqual(original_bqm.energy(spin_sample),
                                   bqm.energy(binary_sample))

    def test_spin_property(self):
        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = -1.4
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        self.assertIs(model, model.spin)

        #

        # create a binary model
        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = -1.4
        vartype = dimod.BINARY
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        # get a new spin-model from binary
        spin_model = model.change_vartype(dimod.SPIN, inplace=False)

        # this spin model should be equal to model.spin
        self.assertEqual(model.spin, spin_model)

        # we don't want to make the spin model anew each time, so make sure they
        # are the same object
        self.assertEqual(model.spin, model.spin)  # should always be equal
        self.assertEqual(id(model.spin), id(model.spin))  # should always refer to the same object

        # make sure that model.spin.binary == model
        self.assertEqual(model.spin.binary, model)
        self.assertEqual(id(model.spin.binary), id(model))

    def test_binary_property(self):
        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = -1.4
        vartype = dimod.BINARY
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        self.assertIs(model, model.binary)

        #

        # create a binary model
        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = -1.4
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        binary_model = model.change_vartype(dimod.BINARY, inplace=False)

        self.assertEqual(model.binary, binary_model)

        self.assertEqual(model.binary, model.binary)  # should always be equal
        self.assertEqual(id(model.binary), id(model.binary))  # should always refer to the same object

        self.assertEqual(model.binary.spin, model)
        self.assertEqual(id(model.binary.spin), id(model))

    def test_spin_property_relabel(self):
        # create a spin model
        linear = {v: .1 * v for v in range(-5, 5)}
        quadratic = {(u, v): .1 * u * v for u, v in itertools.combinations(linear, 2)}
        offset = 1.2
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
        __ = model.binary

        # change a variable label in place
        model.relabel_variables({0: 'a'})

        self.assertIn('a', model.binary.linear)
        self.assertNotIn(0, model.binary.linear)

    def test_binary_property_relabel(self):
        # create a spin model
        linear = {v: .1 * v for v in range(-5, 5)}
        quadratic = {(u, v): .1 * u * v for u, v in itertools.combinations(linear, 2)}
        offset = 1.2
        vartype = dimod.BINARY
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
        __ = model.spin

        # change a variable label in place
        model.relabel_variables({0: 'a'})

        self.assertIn('a', model.spin.linear)
        self.assertNotIn(0, model.spin.linear)

    def test_pickle_empty(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        retrieved = pickle.loads(pickle.dumps(bqm))
        self.assertEqual(bqm, retrieved)

    def test_consistency_after_retrieve(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
        retrieved = pickle.loads(pickle.dumps(bqm))
        # make sure the retrieved one was reconstructed properly
        retrieved.add_variable('a', -1)
        bqm.add_variable('a', -1)
        retrieved.add_interaction(0, 1, -.5)
        bqm.add_interaction(0, 1, -.5)
        self.assertEqual(bqm, retrieved)
        self.assertConsistentBQM(retrieved)

    def test_pickle_full(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': -1}, {'ab': 1, 'bc': -1})
        retrieved = pickle.loads(pickle.dumps(bqm))
        self.assertEqual(bqm, retrieved)


class TestConvert(unittest.TestCase):

    def test_to_ising_spin_to_ising(self):
        linear = {0: 7.1, 1: 103}
        quadratic = {(0, 1): .97}
        offset = 0.3
        vartype = dimod.SPIN

        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        h, J, off = model.to_ising()

        self.assertEqual(off, offset)
        self.assertEqual(linear, h)
        self.assertEqual(quadratic, J)

    def test_to_ising_binary_to_ising(self):
        """binary model's to_ising method"""
        linear = {0: 7.1, 1: 103}
        quadratic = {(0, 1): .97}
        offset = 0.3
        vartype = dimod.BINARY

        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        h, J, off = model.to_ising()

        for spins in itertools.product((-1, 1), repeat=len(model)):
            spin_sample = dict(zip(range(len(spins)), spins))
            bin_sample = {v: (s + 1) // 2 for v, s in spin_sample.items()}

            # calculate the qubo's energy
            energy = off
            for (u, v), bias in J.items():
                energy += spin_sample[u] * spin_sample[v] * bias
            for v, bias in h.items():
                energy += spin_sample[v] * bias

            # and the energy of the model
            self.assertAlmostEqual(energy, model.energy(bin_sample))

    def test_to_qubo_binary_to_qubo(self):
        """Binary model's to_qubo method"""
        linear = {0: 0, 1: 0}
        quadratic = {(0, 1): 1}
        offset = 0.0
        vartype = dimod.BINARY

        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        Q, off = model.to_qubo()

        self.assertEqual(off, offset)
        self.assertEqual({(0, 0): 0, (1, 1): 0, (0, 1): 1}, Q)

    def test_to_qubo_spin_to_qubo(self):
        """Spin model's to_qubo method"""
        linear = {0: .5, 1: 1.3}
        quadratic = {(0, 1): -.435}
        offset = 1.2
        vartype = dimod.SPIN

        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        Q, off = model.to_qubo()

        for spins in itertools.product((-1, 1), repeat=len(model)):
            spin_sample = dict(zip(range(len(spins)), spins))
            bin_sample = {v: (s + 1) // 2 for v, s in spin_sample.items()}

            # calculate the qubo's energy
            energy = off
            for (u, v), bias in Q.items():
                energy += bin_sample[u] * bin_sample[v] * bias

            # and the energy of the model
            self.assertAlmostEqual(energy, model.energy(spin_sample))

    def test_to_numpy_vectors(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {(0, 1): .5, (2, 3): -1, (0, 3): 1.5})

        h, (i, j, values), off = bqm.to_numpy_vectors()

        self.assertEqual(len(h), len(bqm.linear))
        for idx, bias in enumerate(h):
            self.assertAlmostEqual(bias, bqm.linear[idx])

        self.assertEqual(len(i), len(j))
        self.assertEqual(len(j), len(values))
        self.assertEqual(len(values), len(bqm.quadratic))
        for u, v, bias in zip(i, j, values):
            self.assertIn(u, bqm.adj)
            self.assertIn(v, bqm.adj[u])
            self.assertAlmostEqual(bias, bqm.adj[u][v])

    def test_to_numpy_vectors_sorted(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {(0, 1): .5, (3, 2): -1, (0, 3): 1.5})

        h, (i, j, values), off = bqm.to_numpy_vectors(sort_indices=True)

        np.testing.assert_array_equal(h, [0, 0, 0, 0])
        np.testing.assert_array_equal(i, [0, 0, 2])
        np.testing.assert_array_equal(j, [1, 3, 3])
        np.testing.assert_array_equal(values, [.5, 1.5, -1])

    def test_to_numpy_vectors_labels_sorted(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): .5, ('d', 'c'): -1, ('a', 'd'): 1.5})

        # with self.assertRaises(ValueError):
        #     bqm.to_numpy_vectors(sort_indices=True)

        h, (i, j, values), off = bqm.to_numpy_vectors(sort_indices=True, variable_order=['a', 'b', 'c', 'd'])

        np.testing.assert_array_equal(h, [0, 0, 0, 0])
        np.testing.assert_array_equal(i, [0, 0, 2])
        np.testing.assert_array_equal(j, [1, 3, 3])
        np.testing.assert_array_equal(values, [.5, 1.5, -1])

    def test_energies_cpp(self):
        num_variables = 25
        p = .1
        num_samples = 10

        linear = np.random.rand(num_variables)
        row, col = zip(*(pair for pair in itertools.combinations(range(num_variables), 2) if np.random.rand() < p))
        quad = np.random.rand(len(row))

        h = dict(enumerate(linear))
        J = {(u, v): b for u, v, b in zip(row, col, quad)}

        bqm = dimod.BinaryQuadraticModel(h, J, np.random.rand(), dimod.BINARY)

        samples = np.random.randint(2, size=(num_samples, num_variables))

        def _energies(bqm, samples):

            row, col = samples.shape

            energies=[]
            for r in range(row):
                sample = {c: samples[r, c] for c in range(col)}
                energies.append(bqm.energy(sample))
            return energies

        np.testing.assert_array_almost_equal(bqm.energies(samples), _energies(bqm, samples))


    def test_energies_misordered(self):
        h = {0: -2.5, 1: -2.5, 3: -2.5, 4: -5.0, 2: 0.0}
        J = {(0, 1): 2.5, (0, 3): 2.5, (0, 4): 5.0,
             (1, 3): 2.5, (1, 4): 5.0, (3, 4): 5.0,
             (2, 3): -1.0}
        bqm = dimod.BinaryQuadraticModel(h, J, 10.0, 'SPIN')
        bqm.relabel_variables(dict(enumerate('abcde')), inplace=True)

        sample = collections.OrderedDict([('a', -1), ('b', -1), ('e', -1), ('c', -1), ('d', 1)])

        en = 10
        en += sum(sample[v] * b for v, b in bqm.linear.items())
        en += sum(sample[u] * sample[v] * b for (u, v), b in bqm.quadratic.items())
        self.assertAlmostEqual(en, bqm.energies(sample))

    def test_energies_all_ordering(self):
        h = {0: -2.5, 1: -2.5, 3: -2.5, 4: -5.0, 2: 0.0}
        J = {(0, 1): 2.5, (0, 3): 2.5, (0, 4): 5.0,
             (1, 3): 2.5, (1, 4): 5.0, (3, 4): 5.0,
             (2, 3): -1.0}
        bqm = dimod.BinaryQuadraticModel(h, J, 10.0, 'SPIN')

        for variables in itertools.permutations(bqm.variables, len(bqm)):
            for config in itertools.product((-1, 1), repeat=len(bqm)):
                sample = collections.OrderedDict(zip(variables, config))

                en = 10
                en += sum(sample[v] * b for v, b in h.items())
                en += sum(sample[u] * sample[v] * b for (u, v), b in J.items())
                self.assertAlmostEqual(en, bqm.energies(sample))

    def test_from_qubo(self):
        Q = {('a', 'a'): 1, ('a', 'b'): -1}
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        self.assertEqual(bqm, dimod.BinaryQuadraticModel({'a': 1}, {('a', 'b'): -1}, 0.0, dimod.BINARY))

    def test_from_ising(self):
        h = {'a': 1}
        J = {('a', 'b'): -1}

        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
        self.assertEqual(bqm, dimod.BinaryQuadraticModel({'a': 1}, {('a', 'b'): -1}, 0.0, dimod.SPIN))

        #

        # h list
        h = [-1, 1]
        J = {(0, 1): 1}
        bqm = dimod.BinaryQuadraticModel.from_ising(h, J, offset=1)
        self.assertEqual(bqm, dimod.BinaryQuadraticModel({0: -1, 1: 1}, {(0, 1): 1}, 1, dimod.SPIN))

    def test_info(self):
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN, tag=1)

        self.assertIn('tag', bqm.info)
        self.assertEqual(bqm.info['tag'], 1)

        new_bqm = bqm.copy()

        self.assertIn('tag', new_bqm.info)
        self.assertEqual(new_bqm.info['tag'], 1)

        # another_bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN, id=5)

        # bqm.update(another_bqm, ignore_info=False)
        # self.assertIn('tag', bqm.info)
        # self.assertEqual(bqm.info['tag'], 1)
        # self.assertIn('id', bqm.info)
        # self.assertEqual(bqm.info['id'], 5)

        # new_bqm.update(another_bqm, ignore_info=True)
        # self.assertIn('tag', new_bqm.info)
        # self.assertEqual(new_bqm.info['tag'], 1)
        # self.assertNotIn('id', new_bqm.info)

    def test_empty(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        self.assertEqual(bqm, dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN))


class TestSerialization(unittest.TestCase):

    def test_from_serializable_empty_v3(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        s = {'bias_type': 'float32',
             'index_type': 'uint16',
             'info': {},
             'linear_biases': [],
             'num_interactions': 0,
             'num_variables': 0,
             'offset': 0.0,
             'quadratic_biases': [],
             'quadratic_head': [],
             'quadratic_tail': [],
             'type': 'BinaryQuadraticModel',
             'use_bytes': False,
             'variable_labels': [],
             'variable_type': 'SPIN',
             'version': {'bqm_schema': '3.0.0'}}

        self.assertEqual(bqm, dimod.BinaryQuadraticModel.from_serializable(s))

    def test_from_serializable_v3(self):
        linear = {'a': -1, 4: 1, ('a', "complex key"): 3}
        quadratic = {('a', 'c'): 1, ('b', 'c'): 3.0, ('a', 3): -1}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 3, dimod.SPIN)

        s = {'bias_type': 'float32',
             'index_type': 'uint16',
             'info': {},
             'linear_biases': [-1.0, 1.0, 3.0, 0.0, 0.0, 0.0],
             'num_interactions': 3,
             'num_variables': 6,
             'offset': 3.0,
             'quadratic_biases': [1.0, 3.0, -1.0],
             'quadratic_head': [0, 3, 0],
             'quadratic_tail': [3, 4, 5],
             'type': 'BinaryQuadraticModel',
             'use_bytes': False,
             'variable_labels': ['a', 4, ('a', 'complex key'), 'c', 'b', 3],
             'variable_type': 'SPIN',
             'version': {'bqm_schema': '3.0.0'}}

        self.assertEqual(dimod.BinaryQuadraticModel.from_serializable(s), bqm)

    def test_from_serializable_bytes_empty_v3(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        s = {'bias_type': 'float32',
             'index_type': 'uint16',
             'info': {},
             'linear_biases': b'',
             'num_interactions': 0,
             'num_variables': 0,
             'offset': 0.0,
             'quadratic_biases': b'',
             'quadratic_head': b'',
             'quadratic_tail': b'',
             'type': 'BinaryQuadraticModel',
             'use_bytes': True,
             'variable_labels': [],
             'variable_type': 'SPIN',
             'version': {'bqm_schema': '3.0.0'}}

        self.assertEqual(bqm, dimod.BinaryQuadraticModel.from_serializable(s))

    def test_from_serializable_bytes_v3(self):
        linear = {'a': -1, 4: 1, ('a', "complex key"): 3}
        quadratic = {('a', 'c'): 1, ('b', 'c'): 3.0, ('a', 3): -1}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 3, dimod.SPIN)

        s = {'bias_type': 'float32',
             'index_type': 'uint16',
             'info': {},
             'linear_biases': b'\x00\x00\x80\xbf\x00\x00\x80?\x00\x00@@\x00\x00\x00\x00'
                              b'\x00\x00\x00\x00\x00\x00\x00\x00',
             'num_interactions': 3,
             'num_variables': 6,
             'offset': 3.0,
             'quadratic_biases': b'\x00\x00\x80?\x00\x00@@\x00\x00\x80\xbf',
             'quadratic_head': b'\x00\x00\x03\x00\x00\x00',
             'quadratic_tail': b'\x03\x00\x04\x00\x05\x00',
             'type': 'BinaryQuadraticModel',
             'use_bytes': True,
             'variable_labels': ['a', 4, ('a', 'complex key'), 'c', 'b', 3],
             'variable_type': 'SPIN',
             'version': {'bqm_schema': '3.0.0'}}

        self.assertEqual(dimod.BinaryQuadraticModel.from_serializable(s), bqm)

    def test_functional_empty(self):
        # round trip
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        new = dimod.BinaryQuadraticModel.from_serializable(bqm.to_serializable())

        self.assertEqual(bqm, new)

    def test_functional(self):
        linear = {'a': -1, 4: 1, ('a', "complex key"): 3}
        quadratic = {('a', 'c'): 1.5, ('b', 'c'): 3., ('a', 3): -1}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 3, dimod.SPIN)

        new = dimod.BinaryQuadraticModel.from_serializable(bqm.to_serializable())

        self.assertEqual(bqm, new)

    def test_functional_info(self):
        h = {'a': -1, 4: 1, ('a', "complex key"): 3}
        J = {('a', 'c'): 1.5, ('b', 'c'): 3., ('a', 3): -1}
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.BinaryQuadraticModel(h, J, 3, dimod.SPIN, tag=5)

        new = dimod.BinaryQuadraticModel.from_serializable(bqm.to_serializable())

        self.assertEqual(bqm, new)
        self.assertEqual(bqm.info, {"tag": 5})

    def test_functional_bytes_empty(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        new = dimod.BinaryQuadraticModel.from_serializable(bqm.to_serializable(use_bytes=True))

        self.assertEqual(bqm, new)

    def test_functional_bytes(self):
        linear = {'a': -1, 4: 1, ('a', "complex key"): 3}
        quadratic = {('a', 'c'): 1.5, ('b', 'c'): 3., ('a', 3): -1}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 3, dimod.SPIN)

        new = dimod.BinaryQuadraticModel.from_serializable(bqm.to_serializable(use_bytes=True))

        self.assertEqual(bqm, new)

    def test_functional_bytes_info(self):
        h = {'a': -1, 4: 1, ('a', "complex key"): 3}
        J = {('a', 'c'): 1.5, ('b', 'c'): 3., ('a', 3): -1}
        with self.assertWarns(DeprecationWarning):
            bqm = dimod.BinaryQuadraticModel(h, J, 3, dimod.SPIN, tag=5)

        new = dimod.BinaryQuadraticModel.from_serializable(bqm.to_serializable(use_bytes=True))

        self.assertEqual(bqm, new)
        self.assertEqual(new.info, {"tag": 5})

    def test_variable_labels(self):
        h = {0: 0, 1: 1, np.int64(2): 2, np.float(3): 3,
             fractions.Fraction(4, 1): 4, fractions.Fraction(5, 2): 5,
             '6': 6}
        J = {}

        bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

        json.dumps(bqm.to_serializable())


class TestZeroField(unittest.TestCase):
    # test when there are only quadratic biases
    def test_one_interaction(self):
        bqm = dimod.BinaryQuadraticModel({}, {'ab': -1}, 0, dimod.SPIN)
        self.assertEqual(bqm.linear, {'a': 0, 'b': 0})
        self.assertEqual(bqm.adj, {'a': {'b': -1}, 'b': {'a': -1}})
        self.assertEqual(set(bqm.linear.keys()), {'a', 'b'})
        self.assertEqual(set(bqm.linear.items()), {('a', 0), ('b', 0)})
