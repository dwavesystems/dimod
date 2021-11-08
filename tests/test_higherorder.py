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

import itertools
import unittest

import dimod
from dimod import make_quadratic, poly_energy, poly_energies


class TestMakeQuadratic(unittest.TestCase):

    def test__spin_prod(self):

        bqm = dimod.higherorder.utils._spin_product(['a', 'b', 'p', 'aux'])

        for v in ['a', 'b', 'p', 'aux']:
            self.assertIn(v, bqm.variables)
        self.assertEqual(len(bqm), 4)  # has an auxiliary variable

        seen = set()
        samples = dimod.ExactSolver().sample(bqm)
        for sample, energy in samples.data(['sample', 'energy']):
            if energy == 0:
                self.assertEqual(sample['a'] * sample['b'], sample['p'])
                seen.add((sample['a'], sample['b'], sample['p']))
            if sample['a'] * sample['b'] != sample['p']:
                self.assertGreaterEqual(energy, 1)  # gap 1
        self.assertEqual(seen, {(-1, -1, +1),
                                (-1, +1, -1),
                                (+1, -1, -1),
                                (+1, +1, +1)})

    def test_empty(self):
        bqm = make_quadratic({}, 1.0, dimod.SPIN)
        self.assertEqual(bqm, dimod.BinaryQuadraticModel.empty(dimod.SPIN))

    def test_no_higher_order(self):
        poly = {(0, 1): -1, (1, 2): 1}

        bqm = make_quadratic(poly, 1.0, dimod.SPIN)

        variables = set().union(*poly)
        aux_variables = tuple(set(bqm.linear) - variables)
        variables = tuple(variables)
        for config in itertools.product((-1, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))

            energy = poly_energy(sample, poly)

            reduced_energies = []
            for aux_config in itertools.product((-1, 1),
                                                repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                reduced_energies.append(bqm.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_simple(self):
        poly = {(0, 1, 2): -1}

        bqm = make_quadratic(poly, 5.0, dimod.SPIN)

        variables = set().union(*poly)
        aux_variables = tuple(set(bqm.linear) - variables)
        variables = tuple(variables)
        self.assertTrue(aux_variables)
        for config in itertools.product((-1, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))

            energy = poly_energy(sample, poly)

            reduced_energies = []
            for aux_config in itertools.product((-1, 1),
                                                repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                reduced_energies.append(bqm.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_several_terms(self):
        poly = {(0, 1, 2): -1, (1, 2, 3): 1, (0, 2, 3): .5,
                (0,): .4,
                (): .5}

        bqm = make_quadratic(poly, 5.0,
                             bqm=dimod.BinaryQuadraticModel.empty(dimod.SPIN))

        variables = set().union(*poly)
        aux_variables = tuple(set(bqm.linear) - variables)
        variables = tuple(variables)
        self.assertTrue(aux_variables)
        for config in itertools.product((-1, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))

            energy = poly_energy(sample, poly)

            reduced_energies = []
            for aux_config in itertools.product((-1, 1),
                                                repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                reduced_energies.append(bqm.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_bug(self):
        # from a bug report 
        # https://support.dwavesys.com/hc/en-us/community/posts/360035719954-dimod-make-quadratic-returns-error
        H = {(0, 1, 0, 1): -4.61898,
             (0, 1, 1, 0): 4.61898,
             (0, 2, 0, 2): -5.18353,
             (0, 2, 2, 0): 5.18353,
             (1, 0, 0, 1): 4.61898,
             (1, 0, 1, 0): -4.61898,
             (1, 2, 2, 1): 4.97017,
             (2, 0, 0, 2): 5.18353,
             (2, 0, 2, 0): -5.18353,
             (2, 1, 1, 2): 4.97017,
             }

        bqm = make_quadratic(H, 1, 'BINARY')

        # should be no aux variables
        self.assertEqual(set(bqm.variables), {0, 1, 2})

    def test_another(self):
        J = {(0, 1, 2): -1, (0, 1, 3): -1, (2, 3, 0): 1, (3, 2, 0): -1}
        h = {0: 0, 1: 0, 2: 0, 3: 0}
        off = .5

        poly = J.copy()
        poly.update({(v,): bias for v, bias in h.items()})
        poly[()] = off

        bqm = make_quadratic(J, 5.0,
                             bqm=dimod.BinaryQuadraticModel.from_ising(h, {},
                                                                       off))

        variables = set(h).union(*J)
        aux_variables = tuple(set(bqm.linear) - variables)
        variables = tuple(variables)
        self.assertTrue(aux_variables)
        for config in itertools.product((-1, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))

            energy = poly_energy(sample, poly)

            reduced_energies = []
            for aux_config in itertools.product((-1, 1),
                                                repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                reduced_energies.append(bqm.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_vartype(self):
        poly = {(0, 1, 2): 10, (0, 1): 5}  # make sure (0, 1) is most common

        self.assertEqual(dimod.make_quadratic(poly, 1.0, 'SPIN'),
                         dimod.make_quadratic(poly, 1.0, dimod.SPIN))
        self.assertEqual(dimod.make_quadratic(poly, 1.0, 'BINARY'),
                         dimod.make_quadratic(poly, 1.0, dimod.BINARY))

    def test_quad_to_linear(self):
        J = {(0, 1): -1, (0, 1, 2): 1, (0, 1, 3): 1}
        h = {}
        off = .5

        poly = J.copy()
        poly.update({(v,): bias for v, bias in h.items()})
        poly[()] = off

        bqm = make_quadratic(J, 10.0,
                             bqm=dimod.BinaryQuadraticModel.from_ising(h, {},
                                                                       off))

        variables = set(h).union(*J)
        aux_variables = tuple(set(bqm.linear) - variables)
        variables = tuple(variables)
        self.assertTrue(aux_variables)
        for config in itertools.product((-1, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))

            energy = poly_energy(sample, poly)

            reduced_energies = []
            for aux_config in itertools.product((-1, 1),
                                                repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                reduced_energies.append(bqm.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_linear_and_offset(self):

        poly = {(0,): .5, tuple(): 1.3}

        bqm = dimod.make_quadratic(poly, 10.0, dimod.BINARY)

        self.assertEqual(bqm, dimod.BinaryQuadraticModel({0: .5}, {}, 1.3,
                                                         dimod.BINARY))

    def test_binary_polynomial(self):

        HUBO = {(0, 1, 2): .5, (0, 1): 1.3, (2, 4, 1): -1, (3, 2): -1}

        bqm = make_quadratic(HUBO, 1000.0, dimod.BINARY)

        variables = set().union(*HUBO)
        aux_variables = tuple(set(bqm.linear) - variables)
        variables = tuple(variables)
        self.assertTrue(aux_variables)
        for config in itertools.product((0, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))

            energy = poly_energy(sample, HUBO)

            reduced_energies = []
            for aux_config in itertools.product((0, 1),
                                                repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                reduced_energies.append(bqm.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_poly_energies(self):
        linear = {0: 1.0, 1: 1.0}
        j = {(0, 1, 2): 0.5}
        poly = dimod.BinaryPolynomial.from_hising(linear, j)
        samples = [[1, 1, -1], [1, -1, 1], [1, 1, 1], [-1, 1, -1]]

        en = poly_energies(samples, poly)
        self.assertListEqual(list(en), [1.5, -0.5, 2.5, 0.5])

        en = poly_energy(samples[0], poly)
        self.assertAlmostEqual(en, 1.5)

        with self.assertRaises(ValueError):
            poly_energy(samples, poly)

        poly = {('a',): 1.0, ('b',): 1.0, ('a', 'b', 'c'): 0.5}
        samples = [{'a': 1, 'b': 1, 'c': -1},
                   {'a': 1, 'b': -1, 'c': 1},
                   {'a': 1, 'b': 1, 'c': 1},
                   {'a': -1, 'b': 1, 'c': -1}]
        en = poly_energies(samples, poly)
        self.assertListEqual(list(en), [1.5, -0.5, 2.5, 0.5])

        en = poly_energy(samples[0], poly)
        self.assertAlmostEqual(en, 1.5)

        with self.assertRaises(ValueError):
            poly_energy(samples, poly)
