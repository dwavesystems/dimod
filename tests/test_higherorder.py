import unittest
import itertools

import dimod


class TestMakeQuadratic(unittest.TestCase):

    def test__spin_prod(self):

        bqm = dimod.higherorder._spin_product(['a', 'b', 'p', 'aux'])

        for v in ['a', 'b', 'p', 'aux']:
            self.assertIn(v, bqm)
        self.assertEqual(len(bqm), 4)  # has an auxiliary variable

        for sample, energy in dimod.ExactSolver().sample(bqm).data(['sample', 'energy']):
            if energy == 0:
                self.assertEqual(sample['a'] * sample['b'], sample['p'])
            if sample['a'] * sample['b'] != sample['p']:
                self.assertGreaterEqual(energy, 1)

    def test__binary_prod(self):

        variables = ['a', 'b', 'p']

        bqm = dimod.higherorder._binary_product(variables)

        for v in variables:
            self.assertIn(v, bqm)
        self.assertEqual(len(bqm), 3)

        seen_configs = set()
        for sample, energy in dimod.ExactSolver().sample(bqm).data(['sample', 'energy']):
            if energy == 0:
                self.assertEqual(sample['a'] * sample['b'], sample['p'])
                seen_configs.add(tuple(sample[v] for v in variables))
            if sample['a'] * sample['b'] != sample['p']:
                self.assertGreaterEqual(energy, 1)

        self.assertEqual(len(seen_configs), 4)

    def test_no_higher_order(self):
        h = {0: 0, 1: 0, 2: 0}
        J = {(0, 1): -1, (1, 2): 1}
        off = 0

        bqm = dimod.make_quadratic(J, 1.0, dimod.SPIN)

        variables = set(h).union(*J)
        aux_variables = tuple(set(bqm.linear) - variables)
        variables = tuple(variables)
        for config in itertools.product((-1, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))

            energy = poly_energy(sample, h, J, offset=off)

            reduced_energies = []
            for aux_config in itertools.product((-1, 1), repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                reduced_energies.append(bqm.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_simple(self):
        h = {0: 0, 1: 0, 2: 0}
        J = {(0, 1, 2): -1}
        off = 0

        bqm = dimod.make_quadratic(J, 5.0, dimod.SPIN)

        variables = set(h).union(*J)
        aux_variables = tuple(set(bqm.linear) - variables)
        variables = tuple(variables)
        self.assertTrue(aux_variables)
        for config in itertools.product((-1, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))

            energy = poly_energy(sample, h, J, offset=off)

            reduced_energies = []
            for aux_config in itertools.product((-1, 1), repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                reduced_energies.append(bqm.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_several_terms(self):
        h = {0: .4, 1: 0, 2: 0, 3: 0}
        J = {(0, 1, 2): -1, (1, 2, 3): 1, (0, 2, 3): .5}
        off = .5

        bqm = dimod.make_quadratic(J, 5.0, bqm=dimod.BinaryQuadraticModel.from_ising(h, {}, off))

        variables = set(h).union(*J)
        aux_variables = tuple(set(bqm.linear) - variables)
        variables = tuple(variables)
        self.assertTrue(aux_variables)
        for config in itertools.product((-1, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))

            energy = poly_energy(sample, h, J, offset=off)

            reduced_energies = []
            for aux_config in itertools.product((-1, 1), repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                reduced_energies.append(bqm.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_another(self):
        J = {(0, 1, 2): -1, (0, 1, 3): -1, (2, 3, 0): 1, (3, 2, 0): -1}
        h = {0: 0, 1: 0, 2: 0, 3: 0}
        off = .5

        bqm = dimod.make_quadratic(J, 5.0, bqm=dimod.BinaryQuadraticModel.from_ising(h, {}, off))

        variables = set(h).union(*J)
        aux_variables = tuple(set(bqm.linear) - variables)
        variables = tuple(variables)
        self.assertTrue(aux_variables)
        for config in itertools.product((-1, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))

            energy = poly_energy(sample, h, J, offset=off)

            reduced_energies = []
            for aux_config in itertools.product((-1, 1), repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                reduced_energies.append(bqm.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_quad_to_linear(self):
        J = {(0, 1): -1, (0, 1, 2): 1, (0, 1, 3): 1}
        h = {}

        off = .5

        bqm = dimod.make_quadratic(J, 10.0, bqm=dimod.BinaryQuadraticModel.from_ising(h, {}, off))

        variables = set(h).union(*J)
        aux_variables = tuple(set(bqm.linear) - variables)
        variables = tuple(variables)
        self.assertTrue(aux_variables)
        for config in itertools.product((-1, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))

            energy = poly_energy(sample, h, J, offset=off)

            reduced_energies = []
            for aux_config in itertools.product((-1, 1), repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                reduced_energies.append(bqm.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_linear_and_offset(self):

        poly = {(0,): .5, tuple(): 1.3}

        bqm = dimod.make_quadratic(poly, 10.0, dimod.BINARY)

        self.assertEqual(bqm, dimod.BinaryQuadraticModel({0: .5}, {}, 1.3, dimod.BINARY))

    def test_binary_polynomial(self):

        HUBO = {(0, 1, 2): .5, (0, 1): 1.3, (2, 4, 1): -1, (3, 2): -1}

        bqm = dimod.make_quadratic(HUBO, 1000.0, dimod.BINARY)

        variables = set().union(*HUBO)
        aux_variables = tuple(set(bqm.linear) - variables)
        variables = tuple(variables)
        self.assertTrue(aux_variables)
        for config in itertools.product((0, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))

            energy = poly_energy(sample, {}, HUBO, offset=0.0)

            reduced_energies = []
            for aux_config in itertools.product((0, 1), repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                reduced_energies.append(bqm.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))


def poly_energy(sample, h, J, offset=0.0):
    """calculate the energy of the sample for h, J.

    J can be higher-order.
    """
    en = offset
    en += sum(h[v] * sample[v] for v in h)
    en += sum(_prod(sample[v] for v in variables) * bias for variables, bias in J.items())

    return en


def _prod(iterable):
    val = 1
    for v in iterable:
        val *= v
    return val
