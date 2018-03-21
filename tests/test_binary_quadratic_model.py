import unittest
import random
import itertools

import numpy as np

import dimod

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

            self.assertNotIn((v, u), bqm.quadratic)

        for u in bqm.adj:
            for v in bqm.adj[u]:
                self.assertTrue((u, v) in bqm.quadratic or (v, u) in bqm.quadratic)
                self.assertFalse((u, v) in bqm.quadratic and (v, u) in bqm.quadratic)

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


    def test_construction_quadratic(self):
        linear = {v: v * .01 for v in range(10)}
        quadratic = {(u, v): u * v * .01 for u, v in itertools.combinations(linear, 2)}
        offset = 1.2
        vartype = dimod.SPIN

        self.assertEqual(dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY).quadratic, quadratic)

        # quadratic should be a dict or an iterable of 3-tuples
        with self.assertRaises(ValueError):
            dimod.BinaryQuadraticModel(linear, ['a'], offset, dimod.BINARY)
        with self.assertRaises(TypeError):
            dimod.BinaryQuadraticModel(linear, 1, offset, dimod.BINARY)

        # not 2-tuple
        with self.assertRaises(ValueError):
            dimod.BinaryQuadraticModel(linear, {'edge': .5}, offset, dimod.BINARY)

        # no self-loops
        with self.assertRaises(ValueError):
            dimod.BinaryQuadraticModel(linear, {(0, 0): .5}, offset, dimod.BINARY)

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

    def test_add_variable(self):
        bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        bqm.add_variable('a', .5)
        self.assertEqual(bqm.linear['a'], .5)

        # add a single variable of a different type
        bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        bqm.add_variable('a', .5, vartype=dimod.BINARY)

        self.assertEqual(bqm.energy({'a': -1, 'b': -1}), -1)
        self.assertEqual(bqm.energy({'a': 1, 'b': 1}), -.5)

        # and again
        bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1}, 0.0, dimod.BINARY)
        bqm.add_variable('a', .4, vartype=dimod.SPIN)

        self.assertEqual(bqm.energy({'a': 0, 'b': 0}), -.4)
        self.assertEqual(bqm.energy({'a': 1, 'b': 1}), -.6)

        # add a new variable
        bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        bqm.add_variable('c', .5)
        self.assertEqual({'a': 0, 'b': 0, 'c': .5}, bqm.linear)
        bqm.add_variable('c', .5)
        self.assertEqual({'a': 0, 'b': 0, 'c': 1}, bqm.linear)

        # bad type
        with self.assertRaises(ValueError):
            bqm.add_variable('a', 1.2, -1)

    def test_add_variable_counterpart(self):
        # spin
        bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        bqm.add_variable('a', .5)

        for av, bv in itertools.product((0, 1), repeat=2):
            self.assertEqual(bqm.energy({'a': 2 * av - 1, 'b': 2 * bv - 1}),
                             bqm.binary.energy({'a': av, 'b': bv}))

        #

        # spin
        bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        __ = bqm.binary  # create counterpart
        bqm.add_variable('a', .5)

        for av, bv in itertools.product((0, 1), repeat=2):
            self.assertEqual(bqm.energy({'a': 2 * av - 1, 'b': 2 * bv - 1}),
                             bqm.binary.energy({'a': av, 'b': bv}))

        #

        # binary
        bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1}, 0.0, dimod.BINARY)
        bqm.add_variable('a', .5)

        for av, bv in itertools.product((0, 1), repeat=2):
            self.assertEqual(bqm.spin.energy({'a': 2 * av - 1, 'b': 2 * bv - 1}),
                             bqm.energy({'a': av, 'b': bv}))

        #

        # binary
        bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1}, 0.0, dimod.BINARY)
        __ = bqm.spin  # create counterpart
        bqm.add_variable('a', .5)

        for av, bv in itertools.product((0, 1), repeat=2):
            self.assertEqual(bqm.spin.energy({'a': 2 * av - 1, 'b': 2 * bv - 1}),
                             bqm.energy({'a': av, 'b': bv}))

        #

        bqm = dimod.BinaryQuadraticModel({'a': 1.}, {}, 0, dimod.SPIN)
        self.assertEqual(bqm.energy({'a': -1}), bqm.binary.energy({'a': 0}))
        bqm.add_variables_from({'a': .5, 'b': -2})
        self.assertEqual(bqm.linear, {'a': 1.5, 'b': -2})

        self.assertEqual(bqm.energy({'a': -1, 'b': -1}), bqm.binary.energy({'a': 0, 'b': 0}))
        self.assertEqual(bqm.energy({'a': +1, 'b': -1}), bqm.binary.energy({'a': 1, 'b': 0}))
        self.assertEqual(bqm.energy({'a': -1, 'b': +1}), bqm.binary.energy({'a': 0, 'b': 1}))
        self.assertEqual(bqm.energy({'a': +1, 'b': +1}), bqm.binary.energy({'a': 1, 'b': 1}))

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

    def test_add_interaction(self):
        # spin-to-binary
        bqm = dimod.BinaryQuadraticModel({'a': 0, 'b': 0}, {}, 0.0, dimod.BINARY)
        bqm.add_interaction('a', 'b', -1, vartype=dimod.SPIN)  # add a chain link

        self.assertEqual(bqm.energy({'a': 0, 'b': 0}), -1)
        self.assertEqual(bqm.energy({'a': 1, 'b': 1}), -1)
        self.assertConsistentBQM(bqm)

        bqm = dimod.BinaryQuadraticModel({'b': 0}, {}, 0.0, dimod.BINARY)
        bqm.add_interaction('a', 'b', -1, vartype=dimod.SPIN)  # add a chain link

        self.assertEqual(bqm.energy({'a': 0, 'b': 0}), -1)
        self.assertEqual(bqm.energy({'a': 1, 'b': 1}), -1)
        self.assertConsistentBQM(bqm)

        bqm = dimod.BinaryQuadraticModel({'a': 0}, {}, 0.0, dimod.BINARY)
        bqm.add_interaction('a', 'b', -1, vartype=dimod.SPIN)  # add a chain link

        self.assertEqual(bqm.energy({'a': 0, 'b': 0}), -1)
        self.assertEqual(bqm.energy({'a': 1, 'b': 1}), -1)
        self.assertConsistentBQM(bqm)

        # binary-to-spin
        bqm = dimod.BinaryQuadraticModel({'a': 0, 'b': 0}, {}, 0.0, dimod.SPIN)
        bqm.add_interaction('a', 'b', -1, vartype=dimod.BINARY)  # add a chain link

        self.assertEqual(bqm.energy({'a': +1, 'b': +1}), -1)
        self.assertEqual(bqm.energy({'a': -1, 'b': +1}), 0)
        self.assertEqual(bqm.energy({'a': +1, 'b': -1}), 0)
        self.assertEqual(bqm.energy({'a': -1, 'b': -1}), 0)
        self.assertConsistentBQM(bqm)

        bqm = dimod.BinaryQuadraticModel({'b': 0}, {}, 0.0, dimod.SPIN)
        bqm.add_interaction('a', 'b', -1, vartype=dimod.BINARY)  # add a chain link

        self.assertEqual(bqm.energy({'a': +1, 'b': +1}), -1)
        self.assertEqual(bqm.energy({'a': -1, 'b': +1}), 0)
        self.assertEqual(bqm.energy({'a': +1, 'b': -1}), 0)
        self.assertEqual(bqm.energy({'a': -1, 'b': -1}), 0)
        self.assertConsistentBQM(bqm)

        bqm = dimod.BinaryQuadraticModel({'a': 0}, {}, 0.0, dimod.SPIN)
        bqm.add_interaction('a', 'b', -1, vartype=dimod.BINARY)  # add a chain link

        self.assertEqual(bqm.energy({'a': +1, 'b': +1}), -1)
        self.assertEqual(bqm.energy({'a': -1, 'b': +1}), 0)
        self.assertEqual(bqm.energy({'a': +1, 'b': -1}), 0)
        self.assertEqual(bqm.energy({'a': -1, 'b': -1}), 0)
        self.assertConsistentBQM(bqm)

        # no type specified
        bqm = dimod.BinaryQuadraticModel({'a': 0, 'b': 0}, {}, 0.0, dimod.SPIN)
        bqm.add_interaction('a', 'b', -1)
        self.assertEqual(bqm.adj, {'a': {'b': -1}, 'b': {'a': -1}})
        self.assertConsistentBQM(bqm)

        # add to empty
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
        bqm.add_interaction('a', 'b', -1)
        self.assertEqual(bqm.adj, {'a': {'b': -1}, 'b': {'a': -1}})
        self.assertConsistentBQM(bqm)

        # existing biases
        bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        bqm.add_interaction('a', 'b', -1)
        self.assertEqual(bqm.adj, {'a': {'b': -2}, 'b': {'a': -2}})
        bqm.add_interaction('b', 'a', -1)
        self.assertEqual(bqm.adj, {'a': {'b': -3}, 'b': {'a': -3}})
        self.assertConsistentBQM(bqm)

        # unknown vartype
        bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
        with self.assertRaises(ValueError):
            bqm.add_interaction('a', 'b', -1, vartype=-1)

    def test_add_interaction_counterpart(self):
        # spin
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
        bqm.add_interaction('a', 'b', .5)
        self.assertConsistentBQM(bqm)

        for av, bv in itertools.product((0, 1), repeat=2):
            self.assertEqual(bqm.energy({'a': 2 * av - 1, 'b': 2 * bv - 1}),
                             bqm.binary.energy({'a': av, 'b': bv}))

        bqm.add_interaction('a', 'b', .5)
        self.assertConsistentBQM(bqm)

        for av, bv in itertools.product((0, 1), repeat=2):
            self.assertEqual(bqm.energy({'a': 2 * av - 1, 'b': 2 * bv - 1}),
                             bqm.binary.energy({'a': av, 'b': bv}))

        #

        # spin
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
        __ = bqm.binary  # create counterpart
        bqm.add_interaction('a', 'b', -1)
        self.assertConsistentBQM(bqm)

        for av, bv in itertools.product((0, 1), repeat=2):
            self.assertEqual(bqm.energy({'a': 2 * av - 1, 'b': 2 * bv - 1}),
                             bqm.binary.energy({'a': av, 'b': bv}))

        #

        # binary
        bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1}, 0.0, dimod.BINARY)
        bqm.add_interaction('a', 'b', .5)
        self.assertConsistentBQM(bqm)

        for av, bv in itertools.product((0, 1), repeat=2):
            self.assertEqual(bqm.spin.energy({'a': 2 * av - 1, 'b': 2 * bv - 1}),
                             bqm.energy({'a': av, 'b': bv}))

        #

        # binary
        bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1}, 0.0, dimod.BINARY)
        __ = bqm.spin  # create counterpart
        bqm.add_interaction('a', 'b', -1)
        self.assertConsistentBQM(bqm)

        for av, bv in itertools.product((0, 1), repeat=2):
            self.assertEqual(bqm.spin.energy({'a': 2 * av - 1, 'b': 2 * bv - 1}),
                             bqm.energy({'a': av, 'b': bv}))

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

        #

        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
        bqm.remove_variable(3)  # silent fail

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

        bqm.remove_interaction('a', 1)  # silent fail

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

    def test_fix_variable(self):
        # spin model, fix variable to +1
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        bqm.fix_variable('a', +1)

        self.assertEqual(bqm.energy({'b': +1}), original_bqm.energy({'b': +1, 'a': +1}))
        self.assertEqual(bqm.energy({'b': -1}), original_bqm.energy({'b': -1, 'a': +1}))
        self.assertEqual(bqm.binary.energy({'b': 1}), original_bqm.binary.energy({'b': 1, 'a': 1}))
        self.assertEqual(bqm.binary.energy({'b': 0}), original_bqm.binary.energy({'b': 0, 'a': 1}))

        #

        # spin model with binary built, fix variable to +1
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        __ = bqm.binary  # create the binary version
        bqm.fix_variable('a', +1)

        self.assertEqual(bqm.energy({'b': +1}), original_bqm.energy({'b': +1, 'a': +1}))
        self.assertEqual(bqm.energy({'b': -1}), original_bqm.energy({'b': -1, 'a': +1}))
        self.assertEqual(bqm.binary.energy({'b': 1}), original_bqm.binary.energy({'b': 1, 'a': 1}))
        self.assertEqual(bqm.binary.energy({'b': 0}), original_bqm.binary.energy({'b': 0, 'a': 1}))

        #

        # spin model, fix variable to -1
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        bqm.fix_variable('a', -1)

        self.assertEqual(bqm.energy({'b': +1}), original_bqm.energy({'b': +1, 'a': -1}))
        self.assertEqual(bqm.energy({'b': -1}), original_bqm.energy({'b': -1, 'a': -1}))
        self.assertEqual(bqm.binary.energy({'b': 1}), original_bqm.binary.energy({'b': 1, 'a': 0}))
        self.assertEqual(bqm.binary.energy({'b': 0}), original_bqm.binary.energy({'b': 0, 'a': 0}))

        #

        # spin model with binary built, fix variable to -1
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        __ = bqm.binary  # create the binary version
        bqm.fix_variable('a', -1)

        self.assertEqual(bqm.energy({'b': +1}), original_bqm.energy({'b': +1, 'a': -1}))
        self.assertEqual(bqm.energy({'b': -1}), original_bqm.energy({'b': -1, 'a': -1}))
        self.assertEqual(bqm.binary.energy({'b': 1}), original_bqm.binary.energy({'b': 1, 'a': 0}))
        self.assertEqual(bqm.binary.energy({'b': 0}), original_bqm.binary.energy({'b': 0, 'a': 0}))

        #

        # binary model, fix variable to +1
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        bqm.fix_variable('a', 1)

        self.assertEqual(bqm.spin.energy({'b': +1}), original_bqm.energy({'b': 1, 'a': 1}))
        self.assertEqual(bqm.spin.energy({'b': -1}), original_bqm.energy({'b': 0, 'a': 1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': +1}), original_bqm.spin.energy({'b': +1, 'a': +1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': -1}), original_bqm.spin.energy({'b': -1, 'a': +1}))

        #

        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        __ = bqm.spin
        bqm.fix_variable('a', 1)

        self.assertEqual(bqm.spin.energy({'b': +1}), original_bqm.energy({'b': 1, 'a': 1}))
        self.assertEqual(bqm.spin.energy({'b': -1}), original_bqm.energy({'b': 0, 'a': 1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': +1}), original_bqm.spin.energy({'b': +1, 'a': +1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': -1}), original_bqm.spin.energy({'b': -1, 'a': +1}))

        #

        # binary model, fix variable to 0
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        bqm.fix_variable('a', 0)

        self.assertEqual(bqm.spin.energy({'b': +1}), original_bqm.energy({'b': 1, 'a': 0}))
        self.assertEqual(bqm.spin.energy({'b': -1}), original_bqm.energy({'b': 0, 'a': 0}))
        self.assertAlmostEqual(bqm.spin.energy({'b': +1}), original_bqm.spin.energy({'b': +1, 'a': -1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': -1}), original_bqm.spin.energy({'b': -1, 'a': -1}))

        #

        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        original_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        __ = bqm.spin
        bqm.fix_variable('a', 0)

        self.assertEqual(bqm.spin.energy({'b': +1}), original_bqm.energy({'b': 1, 'a': 0}))
        self.assertEqual(bqm.spin.energy({'b': -1}), original_bqm.energy({'b': 0, 'a': 0}))
        self.assertAlmostEqual(bqm.spin.energy({'b': +1}), original_bqm.spin.energy({'b': +1, 'a': -1}))
        self.assertAlmostEqual(bqm.spin.energy({'b': -1}), original_bqm.spin.energy({'b': -1, 'a': -1}))

        #

        with self.assertRaises(ValueError):
            bqm.fix_variable('b', -1)  # spin for binary

    def test_flip_variable(self):

        # single spin variable, trivial
        bqm = dimod.BinaryQuadraticModel({'a': -1}, {}, 0.0, dimod.SPIN)
        original_bqm = dimod.BinaryQuadraticModel({'a': -1}, {}, 0.0, dimod.SPIN)
        bqm.flip_variable('a')
        self.assertAlmostEqual(bqm.energy({'a': +1}), original_bqm.energy({'a': -1}))
        self.assertAlmostEqual(bqm.energy({'a': -1}), original_bqm.energy({'a': +1}))
        self.assertConsistentBQM(bqm)

        bqm.flip_variable('a')  # should return to original
        self.assertEqual(bqm, original_bqm)

        #

        # more complicated spin model
        linear = {v: v * -.43 for v in range(10)}
        quadratic = {(u, v): u * v * -.021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.SPIN
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
        original_bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        bqm.flip_variable(4)
        self.assertConsistentBQM(bqm)
        self.assertEqual(linear[4], bqm.linear[4] * -1)
        self.assertNotEqual(bqm, original_bqm)

        sample = {v: 1 for v in linear}
        flipped_sample = sample.copy()
        flipped_sample[4] = -1
        self.assertAlmostEqual(bqm.energy(flipped_sample), original_bqm.energy(sample))

        bqm.flip_variable(4)  # should return to original
        self.assertEqual(bqm, original_bqm)

        #

        # single binary variable
        bqm = dimod.BinaryQuadraticModel({'a': -1}, {}, 0.0, dimod.BINARY)
        original_bqm = dimod.BinaryQuadraticModel({'a': -1}, {}, 0.0, dimod.BINARY)
        bqm.flip_variable('a')
        self.assertAlmostEqual(bqm.energy({'a': 1}), original_bqm.energy({'a': 0}))
        self.assertAlmostEqual(bqm.energy({'a': 0}), original_bqm.energy({'a': 1}))
        self.assertConsistentBQM(bqm)

        bqm.flip_variable('a')  # should return to original
        self.assertEqual(bqm, original_bqm)

        #

        linear = {v: v * -.43 for v in range(10)}
        quadratic = {(u, v): u * v * -.021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
        original_bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        bqm.flip_variable(4)
        self.assertConsistentBQM(bqm)
        self.assertNotEqual(bqm, original_bqm)

        sample = {v: 1 for v in linear}
        flipped_sample = sample.copy()
        flipped_sample[4] = 0
        self.assertAlmostEqual(bqm.energy(flipped_sample), original_bqm.energy(sample))

        bqm.flip_variable(4)  # should return to original
        self.assertEqual(bqm, original_bqm)

        #

        bqm.flip_variable(100000)  # silent fail

    def test_update(self):
        binary_bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1}, 0, dimod.BINARY)
        spin_bqm = dimod.BinaryQuadraticModel({'c': -1}, {('b', 'c'): 1}, 1.2, dimod.SPIN)

        binary_bqm.update(spin_bqm)

        # binary contribution is 0.0
        self.assertEqual(binary_bqm.energy({'a': 0, 'b': 0, 'c': 0}), spin_bqm.energy({'c': -1, 'b': -1}))

    def test_constract_variables(self):
        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1, ('b', 'c'): 1}, 1.2, dimod.BINARY)
        original_bqm = bqm.copy()

        bqm.contract_variables('a', 'b')
        self.assertNotIn('b', bqm.linear)
        self.assertConsistentBQM(bqm)

        self.assertAlmostEqual(bqm.energy({'a': 0, 'c': 1}), original_bqm.energy({'a': 0, 'b': 0, 'c': 1}))
        self.assertAlmostEqual(bqm.energy({'a': 1, 'c': 1}), original_bqm.energy({'a': 1, 'b': 1, 'c': 1}))
        self.assertAlmostEqual(bqm.energy({'a': 0, 'c': 0}), original_bqm.energy({'a': 0, 'b': 0, 'c': 0}))
        self.assertAlmostEqual(bqm.energy({'a': 1, 'c': 0}), original_bqm.energy({'a': 1, 'b': 1, 'c': 0}))

        #

        bqm = dimod.BinaryQuadraticModel({'a': .3}, {('a', 'b'): -1, ('b', 'c'): 1}, 1.2, dimod.SPIN)
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
        self.assertNotEqual(id(model), id(newmodel))
        self.assertNotEqual(id(model.linear), id(newmodel.linear))
        self.assertNotEqual(id(model.quadratic), id(newmodel.quadratic))

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
        self.assertEqual(id(model), id(newmodel))
        self.assertEqual(id(model.linear), id(newmodel.linear))
        self.assertEqual(id(model.quadratic), id(newmodel.quadratic))

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
        self.assertNotEqual(id(bqm.linear), id(new_bqm.linear))
        self.assertNotEqual(id(bqm.quadratic), id(new_bqm.quadratic))
        self.assertNotEqual(id(bqm.adj), id(new_bqm.adj))

        for v in bqm.linear:
            self.assertNotEqual(id(bqm.adj[v]), id(new_bqm.adj[v]))

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


class TestConvert(unittest.TestCase):
    @unittest.skipUnless(_networkx, "No networkx installed")
    def test_to_networkx_graph(self):
        graph = nx.barbell_graph(7, 6)

        # build a BQM
        model = dimod.BinaryQuadraticModel({v: -.1 for v in graph},
                                           {edge: -.4 for edge in graph.edges},
                                           1.3,
                                           vartype=dimod.SPIN)

        # get the graph
        BQM = model.to_networkx_graph()

        self.assertEqual(set(graph), set(BQM))
        for u, v in graph.edges:
            self.assertIn(u, BQM[v])

        for v, bias in model.linear.items():
            self.assertEqual(bias, BQM.nodes[v]['bias'])

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

    def test_to_numpy_matrix(self):
        # integer-indexed, binary bqm
        linear = {v: v * .01 for v in range(10)}
        quadratic = {(v, u): u * v * .01 for u, v in itertools.combinations(linear, 2)}
        quadratic[(0, 1)] = quadratic[(1, 0)]
        del quadratic[(1, 0)]
        offset = 1.2
        vartype = dimod.BINARY
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        M = bqm.to_numpy_matrix()

        self.assertTrue(np.array_equal(M, np.triu(M)))  # upper triangular

        for (row, col), bias in np.ndenumerate(M):
            if row == col:
                self.assertEqual(bias, linear[row])
            else:
                self.assertTrue((row, col) in quadratic or (col, row) in quadratic)
                self.assertFalse((row, col) in quadratic and (col, row) in quadratic)

                if row > col:
                    self.assertEqual(bias, 0)
                else:
                    if (row, col) in quadratic:
                        self.assertEqual(quadratic[(row, col)], bias)
                    else:
                        self.assertEqual(quadratic[(col, row)], bias)

        #

        # integer-indexed, not contiguous
        bqm = dimod.BinaryQuadraticModel({}, {(0, 3): -1}, 0.0, dimod.BINARY)

        with self.assertRaises(ValueError):
            M = bqm.to_numpy_matrix()

        #

        # string-labeled, variable_order provided
        linear = {'a': -1}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

        with self.assertRaises(ValueError):
            bqm.to_numpy_matrix(['a', 'c'])  # incomplete variable order

        M = bqm.to_numpy_matrix(['a', 'c', 'b'])

        self.assertTrue(np.array_equal(M, [[-1., 1.2, 0.], [0., 0., 0.3], [0., 0., 0.]]))

    def test_from_numpy_matrix(self):

        linear = {'a': -1}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

        variable_order = ['a', 'c', 'b']

        M = bqm.to_numpy_matrix(variable_order=variable_order)

        new_bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(M, variable_order=variable_order)

        self.assertEqual(bqm, new_bqm)

        #

        # zero-interactions get ignored unless provided in interactions
        linear = {'a': -1}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3, ('a', 'b'): 0}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)
        variable_order = ['a', 'c', 'b']
        M = bqm.to_numpy_matrix(variable_order=variable_order)

        new_bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(M, variable_order=variable_order)

        self.assertNotIn(('a', 'b'), new_bqm.quadratic)
        self.assertNotIn(('b', 'a'), new_bqm.quadratic)

        new_bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(M, variable_order=variable_order, interactions=quadratic)

        self.assertEqual(bqm, new_bqm)

        #

        M = np.asarray([[0, 1], [0, 0]])
        bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(M)
        self.assertEqual(bqm, dimod.BinaryQuadraticModel({0: 0, 1: 0}, {(0, 1): 1}, 0, dimod.BINARY))

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

    @unittest.skipUnless(_pandas, "No pandas installed")
    def test_to_pandas_dataframe(self):
        linear = {'a': -1}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3, ('a', 'b'): 0}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

        bqm_df = bqm.to_pandas_dataframe()

        for config in itertools.product((0, 1), repeat=3):
            sample = dict(zip('abc', config))
            sample_series = pd.Series(sample)

            self.assertAlmostEqual(bqm.energy(sample), sample_series.dot(bqm_df.dot(sample_series)))

        bqm_new = dimod.BinaryQuadraticModel.from_pandas_dataframe(bqm_df, interactions=quadratic)

        self.assertAlmostEqual(bqm.linear, bqm_new.linear)
        for u in bqm.adj:
            for v in bqm.adj[u]:
                self.assertAlmostEqual(bqm.adj[u][v], bqm_new.adj[u][v])

        #

        # unlike var names
        linear = {'a': -1, 16: 0.}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3, ('a', 'b'): 0}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

        bqm_df = bqm.to_pandas_dataframe()

        for config in itertools.product((0, 1), repeat=4):
            sample = dict(zip(['a', 'b', 'c', 16], config))
            sample_series = pd.Series(sample)

            self.assertAlmostEqual(bqm.energy(sample), sample_series.dot(bqm_df.dot(sample_series)))

        bqm_new = dimod.BinaryQuadraticModel.from_pandas_dataframe(bqm_df, interactions=quadratic)

        self.assertAlmostEqual(bqm.linear, bqm_new.linear)
        for u in bqm.adj:
            for v in bqm.adj[u]:
                self.assertAlmostEqual(bqm.adj[u][v], bqm_new.adj[u][v])

    def test_info(self):
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN, tag=1)

        self.assertIn('tag', bqm.info)
        self.assertEqual(bqm.info['tag'], 1)

        new_bqm = bqm.copy()

        self.assertIn('tag', new_bqm.info)
        self.assertEqual(new_bqm.info['tag'], 1)

        another_bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN, id=5)

        bqm.update(another_bqm, ignore_info=False)
        self.assertIn('tag', bqm.info)
        self.assertEqual(bqm.info['tag'], 1)
        self.assertIn('id', bqm.info)
        self.assertEqual(bqm.info['id'], 5)

        new_bqm.update(another_bqm, ignore_info=True)
        self.assertIn('tag', new_bqm.info)
        self.assertEqual(new_bqm.info['tag'], 1)
        self.assertNotIn('id', new_bqm.info)

    def test_empty(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        self.assertEqual(bqm, dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN))
