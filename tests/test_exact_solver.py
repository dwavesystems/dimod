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

import unittest
from itertools import product

import numpy as np
import numpy.testing as npt

import dimod
import dimod.testing
from dimod.exceptions import SamplerUnknownArgWarning


@dimod.testing.load_sampler_bqm_tests(dimod.ExactSolver)
class TestExactSolver(unittest.TestCase):
    def test_instantiation(self):
        sampler = dimod.ExactSolver()

        dimod.testing.assert_sampler_api(sampler)

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

    def test_sample_DISCRETE_empty(self):
        dqm = dimod.DiscreteQuadraticModel()
        response = dimod.ExactDQMSolver().sample_dqm(dqm)

        self.assertEqual(response.record.sample.shape, (0, 0))
        self.assertIs(response.vartype, dimod.DISCRETE)
        
    def test_sample_CONSTRAINED_empty(self):
        cqm = dimod.ConstrainedQuadraticModel()
        response = dimod.ExactCQMSolver().sample_cqm(cqm)

        self.assertEqual(response.record.sample.shape, (0, 0))

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

        dimod.testing.assert_response_energies(response, bqm)

    def test_sample_BINARY(self):
        bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0},
                                         {(0, 1): -1.0, (1, 2): 1.0, (0, 2): 1.0},
                                         1.0,
                                         dimod.BINARY)

        response = dimod.ExactSolver().sample(bqm)

        # every possible combination should be present
        self.assertEqual(len(response), 2**len(bqm))
        self.assertEqual(response.record.sample.shape, (2**len(bqm), len(bqm)))

        # confirm vartype
        self.assertIs(response.vartype, bqm.vartype)

        dimod.testing.assert_response_energies(response, bqm)

    def test_sample_DISCRETE(self):
        dqm = dimod.DiscreteQuadraticModel.from_numpy_vectors(
                        case_starts =   [0, 3],
                        linear_biases = [0, 1, 2, 0, 1, 2, 3, 4, 5],
                        quadratic =     ([5, 5, 5, 7, 7, 7], [0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 2, 3])
                                )
        response = dimod.ExactDQMSolver().sample_dqm(dqm)

        # every possible combination should be present
        self.assertEqual(len(response), 18)
        self.assertEqual(response.record.sample.shape, (18, dqm.num_variables()))

        #confirm vartype
        self.assertIs(response.vartype, dimod.DISCRETE)

        dimod.testing.assert_sampleset_energies_dqm(response, dqm)
        
    def test_sample_CONSTRAINED_BINARY(self):
        # using Binary variables, with equality constraint:
        cqm = dimod.ConstrainedQuadraticModel()
        x, y, z = dimod.Binary('x'), dimod.Binary('y'), dimod.Binary('z')
        cqm.set_objective(x*y + 2*y*z)
        cqm.add_constraint(x*y == 1)
        
        response = dimod.ExactCQMSolver().sample_cqm(cqm)
        
        # every possible combination should be present
        self.assertEqual(len(response), 8)
        self.assertEqual(response.record.sample.shape, (8, len(cqm.variables)))
        
        # only two samples should be feasible
        feasible_responses = response.filter(lambda d: d.is_feasible)
        self.assertEqual(len(feasible_responses), 2)
        
        dimod.testing.assert_sampleset_energies_cqm(response, cqm)
        
    def test_sample_CONSTRAINED_SPIN(self):    
        # using Spin variables, with inequality constraint:
        cqm = dimod.ConstrainedQuadraticModel()
        x, y, z = dimod.Spin('x'), dimod.Spin('y'), dimod.Spin('z')
        cqm.set_objective(x*y + 2*y*z)
        cqm.add_constraint(x*y <= 0)
        
        response = dimod.ExactCQMSolver().sample_cqm(cqm)
        
        # every possible combination should be present
        self.assertEqual(len(response), 8)
        self.assertEqual(response.record.sample.shape, (8, len(cqm.variables)))
        
        # four samples should be feasible
        feasible_responses = response.filter(lambda d: d.is_feasible)
        self.assertEqual(len(feasible_responses), 4)
        
        dimod.testing.assert_sampleset_energies_cqm(response, cqm)
        
    def test_sample_CONSTRAINED_INTEGER(self):    
        # using Integer variables, with inequality constraint:
        cqm = dimod.ConstrainedQuadraticModel()
        x, y, z = dimod.Integer('x', lower_bound=1, upper_bound=3), dimod.Integer('y', lower_bound=4, upper_bound=5), dimod.Integer('z', lower_bound=-2, upper_bound=1)
        cqm.set_objective(x*y + 2*y*z)
        cqm.add_constraint(x*z >= 1)
        
        response = dimod.ExactCQMSolver().sample_cqm(cqm)
        
        # every possible combination should be present
        self.assertEqual(len(response), 24)
        self.assertEqual(response.record.sample.shape, (24, len(cqm.variables)))
        
        # only six samples should be feasible
        feasible_responses = response.filter(lambda d: d.is_feasible)
        self.assertEqual(len(feasible_responses), 6)
        
        dimod.testing.assert_sampleset_energies_cqm(response, cqm)
    
    def test_sample_CONSTRAINED_MIXED(self):
        # Using various Vartypes, including the `add_discrete` method
        cqm = dimod.ConstrainedQuadraticModel()
        x = {(u,v): dimod.Binary((u,v)) for u,v in product(['u1','u2'], range(3))}
        
        cqm.add_discrete([('u1',v) for v in range(3)], label='u1')
        cqm.add_discrete([('u2',v) for v in range(3)], label='u2')
        
        y, z = dimod.Spin('y'), dimod.Integer('z', lower_bound=1, upper_bound=3)
        
        obj1 = x[('u1',0)] * y - x[('u2',1)] * y 
        obj2 = x[('u1',0)] * z + 2 * x[('u1',2)] * x[('u2',2)]
        
        cqm.set_objective(obj1+obj2)
        cqm.add_constraint(z==2)
        
        response = dimod.ExactCQMSolver().sample_cqm(cqm)
        
        # every possible combination should be present, respecting the discrete constraints
        self.assertEqual(len(response), 54)
        self.assertEqual(response.record.sample.shape, (54, len(cqm.variables)))
        
        # only 18 samples should be feasible
        feasible_responses = response.filter(lambda d: d.is_feasible)
        self.assertEqual(len(feasible_responses), 18)
        
        dimod.testing.assert_sampleset_energies_cqm(response, cqm)
        

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
        self.assertEqual(set(sampleset.variables), set(bqm.variables))

        dqm = dimod.DQM()
        dqm.add_variable(2, 'a')
        sampleset = dimod.ExactDQMSolver().sample_dqm(dqm)
        self.assertEqual(set(sampleset.variables), set(dqm.variables))
        
        cqm = dimod.CQM.from_dqm(dqm)
        sampleset = dimod.ExactCQMSolver().sample_cqm(cqm)
        self.assertEqual(set(sampleset.variables), set(cqm.variables))

    def test_kwargs(self):
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
        with self.assertWarns(SamplerUnknownArgWarning):
            sampleset = dimod.ExactSolver().sample(bqm, a=1, b="abc")

        dqm = dimod.DiscreteQuadraticModel()
        with self.assertWarns(SamplerUnknownArgWarning):
            sampleset = dimod.ExactDQMSolver().sample_dqm(dqm, a=1, b="abc")
      
        cqm = dimod.ConstrainedQuadraticModel()
        with self.assertWarns(SamplerUnknownArgWarning):
            sampleset = dimod.ExactCQMSolver().sample_cqm(cqm, a=1, b="abc")

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

        dimod.testing.assert_response_energies(response, poly)

    def test_sample_BINARY(self):
        poly = dimod.BinaryPolynomial({(): 1.0, (0,): 0.0, (1,): 0.0, (0, 1): -1.0, (1, 2): 1.0, (0, 1, 2): 1.0},
                                      dimod.BINARY)

        response = dimod.ExactPolySolver().sample_poly(poly)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**len(poly.variables))
        self.assertEqual(response.record.sample.shape, (2**len(poly.variables), len(poly.variables)))

        # confirm vartype
        self.assertIs(response.vartype, poly.vartype)

        dimod.testing.assert_response_energies(response, poly)

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

    def test_kwargs(self):
        poly = dimod.BinaryPolynomial({}, dimod.SPIN)
        with self.assertWarns(SamplerUnknownArgWarning):
            response = dimod.ExactPolySolver().sample_poly(poly, a=True, b=2)
