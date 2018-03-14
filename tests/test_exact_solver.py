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

        self.assertEqual(response.samples_matrix.shape, (0, 0))
        self.assertIs(response.vartype, bqm.vartype)

    def test_sample_BINARY_empty(self):
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
        response = dimod.ExactSolver().sample(bqm)

        self.assertEqual(response.samples_matrix.shape, (0, 0))
        self.assertIs(response.vartype, bqm.vartype)

    def test_sample_SPIN(self):
        bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0},
                                         {(0, 1): -1.0, (1, 2): 1.0, (0, 2): 1.0},
                                         1.0,
                                         dimod.SPIN)

        response = dimod.ExactSolver().sample(bqm)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**len(bqm))
        self.assertEqual(np.unique(response.samples_matrix, axis=0).shape, (2**len(bqm), len(bqm)))

        # confirm vartype
        self.assertIs(response.vartype, bqm.vartype)

        # check their energies
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, bqm.energy(sample))

    def test_sample_BINARY(self):
        bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0},
                                         {(0, 1): -1.0, (1, 2): 1.0, (0, 2): 1.0},
                                         1.0,
                                         dimod.BINARY)

        response = dimod.ExactSolver().sample(bqm)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**len(bqm))
        self.assertEqual(np.unique(response.samples_matrix, axis=0).shape, (2**len(bqm), len(bqm)))

        # confirm vartype
        self.assertIs(response.vartype, bqm.vartype)

        # check their energies
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, bqm.energy(sample))

    def test_sample_ising(self):
        h = {0: 0.0, 1: 0.0, 2: 0.0}
        J = {(0, 1): -1.0, (1, 2): 1.0, (0, 2): 1.0}

        response = dimod.ExactSolver().sample_ising(h, J)

        # every possible conbination should be present
        self.assertEqual(len(response), 2**3)
        self.assertEqual(np.unique(response.samples_matrix, axis=0).shape, (2**3, 3))

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
        self.assertEqual(np.unique(response.samples_matrix, axis=0).shape, (2**3, 3))

        # confirm vartype
        self.assertIs(response.vartype, dimod.BINARY)

        # check their energies
        for sample, energy in response.data(['sample', 'energy']):
            self.assertAlmostEqual(energy, dimod.qubo_energy(sample, Q))
