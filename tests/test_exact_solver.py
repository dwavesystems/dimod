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

import numpy as np
import numpy.testing as npt

import dimod
import dimod.testing as dtest


class TestExactSolver(unittest.TestCase):
    def test_instantiation(self):
        sampler = dimod.ExactSolver()

        dtest.assert_sampler_api(sampler)

        # this sampler has no properties and has no accepted parameters
        self.assertEqual(sampler.properties, {})
        self.assertEqual(sampler.parameters, {})

    def test_sample_SPIN_empty(self):
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
        response = dimod.ExactSolver().sample(bqm)

        self.assertEqual(response.record.sample.shape, (0, 0))
        self.assertIs(response.vartype, bqm.vartype)

    def test_sample_BINARY_empty(self):
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
        response = dimod.ExactSolver().sample(bqm)

        self.assertEqual(response.record.sample.shape, (0, 0))
        self.assertIs(response.vartype, bqm.vartype)

    def test_sample_SPIN(self):
        bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0},
                                         {(0, 1): -1.0, (1, 2): 1.0, (0, 2): 1.0},
                                         1.0,
                                         dimod.SPIN)

        response = dimod.ExactSolver().sample(bqm)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**len(bqm))
        self.assertEqual(response.record.sample.shape, (2**len(bqm), len(bqm)))

        # confirm vartype
        self.assertIs(response.vartype, bqm.vartype)

        dtest.assert_response_energies(response, bqm)

    def test_sample_BINARY(self):
        bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0},
                                         {(0, 1): -1.0, (1, 2): 1.0, (0, 2): 1.0},
                                         1.0,
                                         dimod.BINARY)

        response = dimod.ExactSolver().sample(bqm)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**len(bqm))
        self.assertEqual(response.record.sample.shape, (2**len(bqm), len(bqm)))

        # confirm vartype
        self.assertIs(response.vartype, bqm.vartype)

        dtest.assert_response_energies(response, bqm)

    def test_sample_ising(self):
        h = {0: 0.0, 1: 0.0, 2: 0.0}
        J = {(0, 1): -1.0, (1, 2): 1.0, (0, 2): 1.0}

        response = dimod.ExactSolver().sample_ising(h, J)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**3)
        self.assertEqual(response.record.sample.shape, (2**3, 3))

        # confirm vartype
        self.assertIs(response.vartype, dimod.SPIN)

        # check their energies
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, dimod.ising_energy(sample, h, J))

    def test_sample_qubo(self):
        Q = {(0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0}
        Q.update({(0, 1): -1.0, (1, 2): 1.0, (0, 2): 1.0})

        response = dimod.ExactSolver().sample_qubo(Q)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**3)
        self.assertEqual(response.record.sample.shape, (2**3, 3))

        # confirm vartype
        self.assertIs(response.vartype, dimod.BINARY)

        # check their energies
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, dimod.qubo_energy(sample, Q))

    def test_sample_mixed_labels(self):
        h = {'3': 0.6669921875, 4: -2.0, 5: -1.334375, 6: 0.0, 7: -2.0, '1': 1.3328125,
             '2': -1.3330078125, '0': -0.666796875}
        J = {(5, '2'): 1.0, (7, '0'): 0.9998046875, (4, '0'): 0.9998046875, ('3', 4): 0.9998046875,
             (7, '1'): -1.0, (5, '1'): 0.6671875, (6, '2'): 1.0, ('3', 6): 0.6671875,
             (7, '2'): 0.9986328125, (5, '0'): -1.0, ('3', 5): -0.6671875, ('3', 7): 0.998828125,
             (4, '1'): -1.0, (6, '0'): -0.3328125, (4, '2'): 1.0, (6, '1'): 0.0}

        response = dimod.ExactSolver().sample_ising(h, J)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**len(h))
        self.assertEqual(response.record.sample.shape, (2**len(h), len(h)))

        # confirm vartype
        self.assertIs(response.vartype, dimod.SPIN)

        # check their energies
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, dimod.ising_energy(sample, h, J))

    def test_arbitrary_labels(self):
        bqm = dimod.BQM.from_ising({}, {'ab': -1})
        sampleset = dimod.ExactSolver().sample(bqm)
        self.assertEqual(set(sampleset.variables), set(bqm))


class TestExactPolySolver(unittest.TestCase):
    def test_instantiation(self):
        sampler = dimod.ExactPolySolver()

        # this sampler has no properties and has no accepted parameters
        self.assertEqual(sampler.properties, {})
        self.assertEqual(sampler.parameters, {})

    def test_sample_SPIN_empty(self):
        poly= dimod.BinaryPolynomial({}, dimod.SPIN)
        response = dimod.ExactPolySolver().sample_poly(poly)

        self.assertEqual(response.record.sample.shape, (0, 0))
        self.assertIs(response.vartype, poly.vartype)

    def test_sample_BINARY_empty(self):
        poly = dimod.BinaryPolynomial({}, dimod.BINARY)
        response = dimod.ExactPolySolver().sample_poly(poly)

        self.assertEqual(response.record.sample.shape, (0, 0))
        self.assertIs(response.vartype, poly.vartype)

    def test_sample_SPIN(self):
        poly = dimod.BinaryPolynomial.from_hising({0: 0.0, 1: 0.0, 2: 0.0},
                                         {(0, 1): -1.0, (1, 2): 1.0, (0, 1, 2): 1.0},
                                         1.0)

        response = dimod.ExactPolySolver().sample_poly(poly)

        # every possible combination should be present
        self.assertEqual(len(response), 2**len(poly.variables))
        self.assertEqual(response.record.sample.shape, (2**len(poly.variables), len(poly.variables)))

        # confirm vartype
        self.assertIs(response.vartype, poly.vartype)

        dtest.assert_response_energies(response, poly)

    def test_sample_BINARY(self):
        poly = dimod.BinaryPolynomial({(): 1.0, (0,): 0.0, (1,): 0.0, (0, 1): -1.0, (1, 2): 1.0, (0, 1, 2): 1.0},
                                      dimod.BINARY)

        response = dimod.ExactPolySolver().sample_poly(poly)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**len(poly.variables))
        self.assertEqual(response.record.sample.shape, (2**len(poly.variables), len(poly.variables)))

        # confirm vartype
        self.assertIs(response.vartype, poly.vartype)

        dtest.assert_response_energies(response, poly)

    def test_sample_hising(self):
        h = {0: 0.0, 1: 0.0, 2: 0.0}
        J = {(0, 1): -1.0, (1, 2): 1.0, (0, 1, 2): 1.0}

        response = dimod.ExactPolySolver().sample_hising(h, J)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**3)
        self.assertEqual(response.record.sample.shape, (2**3, 3))

        # confirm vartype
        self.assertIs(response.vartype, dimod.SPIN)

    def test_sample_hubo(self):
        Q = {(0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0}
        Q.update({(0, 1): -1.0, (1, 2): 1.0, (0, 1, 2): 1.0})

        response = dimod.ExactPolySolver().sample_hubo(Q)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**3)
        self.assertEqual(response.record.sample.shape, (2**3, 3))

        # confirm vartype
        self.assertIs(response.vartype, dimod.BINARY)

    def test_sample_ising(self):
        h = {0: 0.0, 1: 0.0, 2: 0.0}
        J = {(0, 1): -1.0, (1, 2): 1.0, (0, 2): 1.0}

        response = dimod.ExactPolySolver().sample_hising(h, J)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**3)
        self.assertEqual(response.record.sample.shape, (2**3, 3))

        # confirm vartype
        self.assertIs(response.vartype, dimod.SPIN)

        # check their energies
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, dimod.ising_energy(sample, h, J))

    def test_sample_qubo(self):
        Q = {(0, 0): 0.0, (1, 1): 0.0, (2, 2): 0.0}
        Q.update({(0, 1): -1.0, (1, 2): 1.0, (0, 2): 1.0})

        response = dimod.ExactPolySolver().sample_hubo(Q)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**3)
        self.assertEqual(response.record.sample.shape, (2**3, 3))

        # confirm vartype
        self.assertIs(response.vartype, dimod.BINARY)

        # check their energies
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, dimod.qubo_energy(sample, Q))

    def test_sample_mixed_labels(self):
        h = {'3': 0.6669921875, 4: -2.0, 5: -1.334375, 6: 0.0, 7: -2.0, '1': 1.3328125,
             '2': -1.3330078125, '0': -0.666796875}
        J = {(5, '2'): 1.0, (7, '0'): 0.9998046875, (4, '0'): 0.9998046875, ('3', 4): 0.9998046875,
             (7, '1'): -1.0, (5, '1'): 0.6671875, (6, '2'): 1.0, ('3', 6): 0.6671875,
             (7, '2'): 0.9986328125, (5, '0'): -1.0, ('3', 5): -0.6671875, ('3', 7): 0.998828125,
             (4, '1'): -1.0, (6, '0'): -0.3328125, (4, '2'): 1.0, (6, '1'): 0.0}

        response = dimod.ExactPolySolver().sample_hising(h, J)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**len(h))
        self.assertEqual(response.record.sample.shape, (2**len(h), len(h)))

        # confirm vartype
        self.assertIs(response.vartype, dimod.SPIN)

    def test_arbitrary_labels(self):
        poly = dimod.BinaryPolynomial.from_hising({}, {('a','b','c'): -1})
        sampleset = dimod.ExactPolySolver().sample_poly(poly)
        self.assertEqual(set(sampleset.variables), set(poly.variables))
