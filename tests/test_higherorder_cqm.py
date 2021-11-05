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
from dimod import make_quadratic, make_quadratic_cqm, poly_energy, poly_energies


class TestMakeQuadratic(unittest.TestCase):

    def test__spin_prod(self):

        cqm = dimod.higherorder.utils._spin_product(['a', 'b', 'p', 'aux'])

        for v in ['a', 'b', 'p', 'aux']:
            self.assertIn(v, cqm.variables)
        self.assertEqual(len(cqm), 4)  # has an auxiliary variable

        seen = set()
        samples = dimod.ExactSolver().sample(cqm)
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
        cqm = make_quadratic_cqm({}, dimod.SPIN)
        self.assertTrue(cqm.is_almost_equal(dimod.ConstrainedQuadraticModel()))

    def test_no_higher_order(self):
        poly = {(0, 1): -1, (1, 2): 1}

        cqm = make_quadratic_cqm(poly, dimod.SPIN)

        variables = set().union(*poly)
        aux_variables = tuple(set(cqm.variables) - variables)
        variables = tuple(variables)
        for config in itertools.product((-1, 1), repeat=len(variables)):
            sample = dict(zip(variables, config))
            energy = poly_energy(sample, poly)

            reduced_energies = []
            for aux_config in itertools.product((-1, 1),
                                                repeat=len(aux_variables)):
                aux_sample = dict(zip(aux_variables, aux_config))
                aux_sample.update(sample)
                if cqm.check_feasible(aux_sample):
                    reduced_energies.append(cqm.objective.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_simple(self):
        poly = {(0, 1, 2): -1}

        cqm = make_quadratic_cqm(poly, dimod.SPIN)

        variables = set().union(*poly)
        aux_variables = tuple(set(cqm.variables) - variables)
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
                if cqm.check_feasible(aux_sample):
                    reduced_energies.append(cqm.objective.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))


    def test_several_terms(self):
        poly = {(0, 1, 2): -1, (1, 2, 3): 1, (0, 2, 3): .5,
                (0,): .4,
                (): .5}

        cqm = make_quadratic_cqm(poly, dimod.SPIN, cqm=dimod.ConstrainedQuadraticModel())

        variables = set().union(*poly)
        aux_variables = tuple(set(cqm.variables) - variables)
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
                if cqm.check_feasible(aux_sample):
                    reduced_energies.append(cqm.objective.energy(aux_sample))

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

        cqm = make_quadratic_cqm(H, 'BINARY')

        # should be no aux variables
        self.assertEqual(set(cqm.variables), {0, 1, 2})

    def test_another(self):
        J = {(0, 1, 2): -1, (0, 1, 3): -1, (2, 3, 0): 1, (3, 2, 0): -1}
        h = {0: 0, 1: 0, 2: 0, 3: 0}
        off = .5
        obj = dimod.BinaryQuadraticModel.from_ising(h, {}, off)

        poly = J.copy()
        poly.update({(v,): bias for v, bias in h.items()})
        poly[()] = off

        cqm0 = dimod.ConstrainedQuadraticModel()
        cqm0.set_objective(obj)
        cqm = make_quadratic_cqm(J, dimod.SPIN, cqm=cqm0)

        variables = set(h).union(*J)
        aux_variables = tuple(set(cqm.variables) - variables)
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
                if cqm.check_feasible(aux_sample):
                    reduced_energies.append(cqm.objective.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))
            

    def test_vartype(self):
        poly = {(0, 1, 2): 10, (0, 1): 5}  # make sure (0, 1) is most common

        self.assertTrue(dimod.make_quadratic_cqm(poly, 'SPIN')
                        .is_almost_equal(dimod.make_quadratic_cqm(poly, dimod.SPIN)))
        self.assertTrue(dimod.make_quadratic_cqm(poly, 'BINARY')
                         .is_almost_equal(dimod.make_quadratic_cqm(poly,  dimod.BINARY)))

    def test_quad_to_linear(self):
        J = {(0, 1): -1, (0, 1, 2): 1, (0, 1, 3): 1}
        h = {}
        off = .5

        poly = J.copy()
        poly.update({(v,): bias for v, bias in h.items()})
        poly[()] = off

        cqm0 = dimod.ConstrainedQuadraticModel()
        cqm0.set_objective(dimod.BinaryQuadraticModel.from_ising(h, {}, off))
        cqm = make_quadratic_cqm(J, dimod.SPIN, cqm=cqm0)

        variables = set(h).union(*J)
        aux_variables = tuple(set(cqm.variables) - variables)
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
                if cqm.check_feasible(aux_sample):
                    reduced_energies.append(cqm.objective.energy(aux_sample))

            self.assertAlmostEqual(energy, min(reduced_energies))

    def test_linear_and_offset(self):

        poly = {(0,): .5, tuple(): 1.3}

        cqm = dimod.make_quadratic_cqm(poly, dimod.BINARY)
        obj = dimod.BinaryQuadraticModel({0: .5}, {}, 1.3, dimod.BINARY)

        self.assertTrue(cqm.is_almost_equal(dimod.ConstrainedQuadraticModel.from_bqm(obj)))

    def test_binary_polynomial(self):

        HUBO = {(0, 1, 2): .5, (0, 1): 1.3, (2, 4, 1): -1, (3, 2): -1}

        cqm = make_quadratic_cqm(HUBO, dimod.BINARY)

        variables = set().union(*HUBO)
        aux_variables = tuple(set(cqm.variables) - variables)
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
                if cqm.check_feasible(aux_sample):
                    reduced_energies.append(cqm.objective.energy(aux_sample))

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
