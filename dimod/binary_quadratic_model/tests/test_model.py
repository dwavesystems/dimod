import unittest
import random
import itertools

import networkx as nx

import dimod


class TestBinaryQuadraticModel(unittest.TestCase):

    def test_construction_typical_spin(self):
        # set up a model
        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = 1.4
        vartype = dimod.SPIN
        m = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        self.assertEqual(linear, m.linear)
        self.assertEqual(quadratic, m.quadratic)
        self.assertEqual(offset, m.offset)
        self.assertEqual(vartype, m.vartype)

        for (u, v), bias in quadratic.items():
            self.assertIn(u, m.adj)
            self.assertIn(v, m.adj[u])
            self.assertEqual(m.adj[u][v], bias)

            v, u = u, v
            self.assertIn(u, m.adj)
            self.assertIn(v, m.adj[u])
            self.assertEqual(m.adj[u][v], bias)

        for u in m.adj:
            for v in m.adj[u]:
                self.assertTrue((u, v) in quadratic or (v, u) in quadratic)

    def test_construction_typical_binary(self):

        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = 1.4
        vartype = dimod.BINARY
        m = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        self.assertEqual(linear, m.linear)
        self.assertEqual(quadratic, m.quadratic)
        self.assertEqual(offset, m.offset)

        for (u, v), bias in quadratic.items():
            self.assertIn(u, m.adj)
            self.assertIn(v, m.adj[u])
            self.assertEqual(m.adj[u][v], bias)

            v, u = u, v
            self.assertIn(u, m.adj)
            self.assertIn(v, m.adj[u])
            self.assertEqual(m.adj[u][v], bias)

        for u in m.adj:
            for v in m.adj[u]:
                self.assertTrue((u, v) in quadratic or (v, u) in quadratic)

    def test_input_checking_vartype(self):
        """Check that exceptions get thrown for broken inputs"""

        # this biases values are themselves not important, so just choose them randomly
        linear = {v: random.uniform(-2, 2) for v in range(10)}
        quadratic = {(u, v): random.uniform(-1, 1) for (u, v) in itertools.combinations(linear, 2)}
        offset = random.random()

        with self.assertRaises(TypeError):
            dimod.BinaryQuadraticModel(linear, quadratic, offset, 147)

        with self.assertRaises(TypeError):
            dimod.BinaryQuadraticModel(linear, quadratic, offset, 'my made up type')

        self.assertEqual(dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY).vartype, dimod.BINARY)

        self.assertEqual(dimod.BinaryQuadraticModel(linear, quadratic, offset, {-1, 1}).vartype, dimod.SPIN)

        self.assertEqual(dimod.BinaryQuadraticModel(linear, quadratic, offset, 'BINARY').vartype, dimod.BINARY)

    def test_input_checking_quadratic(self):
        linear = {v: random.uniform(-2, 2) for v in range(11)}
        quadratic = {(u, v): random.uniform(-1, 1) for (u, v) in itertools.combinations(linear, 2)}
        offset = random.random()
        vartype = dimod.SPIN

        self.assertEqual(dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY).quadratic, quadratic)

        # quadratic should be a dict
        with self.assertRaises(TypeError):
            dimod.BinaryQuadraticModel(linear, [], offset, dimod.BINARY)

        # unknown varialbe (vars must be in linear)
        with self.assertRaises(ValueError):
            dimod.BinaryQuadraticModel(linear, {('a', 1): .5}, offset, dimod.BINARY)

        # not 2-tuple
        with self.assertRaises(ValueError):
            dimod.BinaryQuadraticModel(linear, {'edge': .5}, offset, dimod.BINARY)

        # not upper triangular
        with self.assertRaises(ValueError):
            dimod.BinaryQuadraticModel(linear, {(0, 1): .5, (1, 0): -.5}, offset, dimod.BINARY)

        # no self-loops
        with self.assertRaises(ValueError):
            dimod.BinaryQuadraticModel(linear, {(0, 0): .5}, offset, dimod.BINARY)

    def test__repr__(self):
        """check that repr works correctly."""
        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = 1.4

        m = dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.SPIN)

        # should recreate the model
        from dimod import BinaryQuadraticModel, Vartype
        m2 = eval(m.__repr__())

        self.assertEqual(m, m2)

    def test__eq__(self):
        linear = {v: random.uniform(-2, 2) for v in range(11)}
        quadratic = {(u, v): random.uniform(-1, 1) for (u, v) in itertools.combinations(linear, 2)}
        offset = random.random()
        vartype = dimod.SPIN

        self.assertEqual(dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype),
                         dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype))

        # mismatched type
        self.assertNotEqual(dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype), -1)

        self.assertNotEqual(dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN),
                            dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY))

    def test__eq__quadratic_ordering(self):
        linear = {v: random.uniform(-2, 2) for v in range(11)}
        quadratic = {(u, v): random.uniform(-1, 1) for (u, v) in itertools.combinations(linear, 2)}
        offset = random.random()
        vartype = dimod.SPIN

        model0 = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        reversed_quadratic = {(v, u): bias for (u, v), bias in quadratic.items()}

        model1 = dimod.BinaryQuadraticModel(linear, reversed_quadratic, offset, vartype)

        self.assertEqual(model1, model0)

    def test_to_qubo_binary_to_qubo(self):
        """Binary model's to_qubo method"""
        linear = {0: 0, 1: 0}
        quadratic = {(0, 1): 1}
        offset = 0.0
        vartype = dimod.BINARY

        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        Q, off = dimod.to_qubo(model)

        self.assertEqual(off, offset)
        self.assertEqual({(0, 0): 0, (1, 1): 0, (0, 1): 1}, Q)

    def test_to_qubo_spin_to_qubo(self):
        """Spin model's to_qubo method"""
        linear = {0: .5, 1: 1.3}
        quadratic = {(0, 1): -.435}
        offset = 1.2
        vartype = dimod.SPIN

        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        Q, off = dimod.to_qubo(model)

        for spins in itertools.product((-1, 1), repeat=len(model)):
            spin_sample = dict(zip(range(len(spins)), spins))
            bin_sample = {v: (s + 1) // 2 for v, s in spin_sample.items()}

            # calculate the qubo's energy
            energy = off
            for (u, v), bias in Q.items():
                energy += bin_sample[u] * bin_sample[v] * bias

            # and the energy of the model
            self.assertAlmostEqual(energy, model.energy(spin_sample))

    def test_to_ising_spin_to_ising(self):
        linear = {0: 7.1, 1: 103}
        quadratic = {(0, 1): .97}
        offset = 0.3
        vartype = dimod.SPIN

        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        h, J, off = dimod.to_ising(model)

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

        h, J, off = dimod.to_ising(model)

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

    def test_relabel_typical(self):
        linear = {0: .5, 1: 1.3}
        quadratic = {(0, 1): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        mapping = {0: 'a', 1: 'b'}
        newmodel = model.relabel_variables(mapping)

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
        newmodel = model.relabel_variables(mapping, copy=True)
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
        newmodel = model.relabel_variables(mapping, copy=False)
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
        testmodel = model.relabel_variables(partial_overlap_mapping, copy=True)

        # now apply in place
        model.relabel_variables(partial_overlap_mapping, copy=False)

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

        model.relabel_variables(identity_mapping, copy=False)

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
        newmodel = model.relabel_variables(mapping, copy=True)

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
        model.relabel_variables(mapping, copy=False)

        self.assertEqual(newlinear, model.linear)

    def test_copy(self):
        linear = {0: .5, 1: 1.3}
        quadratic = {(0, 1): -.435}
        offset = 1.2
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        new_model = model.copy()

        # everything should have a new id
        self.assertNotEqual(id(model.linear), id(new_model.linear))
        self.assertNotEqual(id(model.quadratic), id(new_model.quadratic))
        self.assertNotEqual(id(model.adj), id(new_model.adj))

        for v in model.linear:
            self.assertNotEqual(id(model.adj[v]), id(new_model.adj[v]))

        # values should all be equal
        self.assertEqual(model.linear, new_model.linear)
        self.assertEqual(model.quadratic, new_model.quadratic)
        self.assertEqual(model.adj, new_model.adj)

        for v in model.linear:
            self.assertEqual(model.adj[v], new_model.adj[v])

        self.assertEqual(model, new_model)

    def test_change_vartype(self):
        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = 1.4
        vartype = dimod.BINARY
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        # should not change
        new_model = model.change_vartype(dimod.BINARY)
        self.assertEqual(model, new_model)
        self.assertNotEqual(id(model), id(new_model))

        # change vartype
        new_model = model.change_vartype(dimod.SPIN)

        # check all of the energies
        for spins in itertools.product((-1, 1), repeat=len(linear)):
            spin_sample = {v: spins[v] for v in linear}
            binary_sample = {v: (spins[v] + 1) // 2 for v in linear}

            self.assertAlmostEqual(model.energy(binary_sample),
                                   new_model.energy(spin_sample))

        linear = {0: 1, 1: -1, 2: .5}
        quadratic = {(0, 1): .5, (1, 2): 1.5}
        offset = -1.4
        vartype = dimod.SPIN
        model = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

        # should not change
        new_model = model.change_vartype(dimod.SPIN)
        self.assertEqual(model, new_model)
        self.assertNotEqual(id(model), id(new_model))

        # change vartype
        new_model = model.change_vartype(dimod.BINARY)

        # check all of the energies
        for spins in itertools.product((-1, 1), repeat=len(linear)):
            spin_sample = {v: spins[v] for v in linear}
            binary_sample = {v: (spins[v] + 1) // 2 for v in linear}

            self.assertAlmostEqual(model.energy(spin_sample),
                                   new_model.energy(binary_sample))

    def test_to_networkx_graph(self):
        graph = nx.barbell_graph(7, 6)

        # build a BQM
        model = dimod.BinaryQuadraticModel({v: -.1 for v in graph},
                                        {edge: -.4 for edge in graph.edges},
                                        1.3,
                                        vartype=dimod.SPIN)

        # get the graph
        BQM = dimod.to_networkx_graph(model)

        self.assertEqual(set(graph), set(BQM))
        for u, v in graph.edges:
            self.assertIn(u, BQM[v])

        for v, bias in model.linear.items():
            self.assertEqual(bias, BQM.nodes[v]['bias'])
