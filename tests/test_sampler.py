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

import dimod


class TestSamplerClass(unittest.TestCase):
    """Tests for the template Sampler class"""

    def test_instantiation_base_class(self):
        with self.assertRaises(TypeError):
            dimod.Sampler()

    def test_instantiation_missing_properties(self):
        # overwrite sample and parameters
        class Dummy(dimod.Sampler):
            def sample(self, bqm):
                # just override
                pass

            @property
            def parameters(self):
                return {}

            # @property
            # def properties(self):
            #     return {}

        with self.assertRaises(TypeError):
            Dummy()

    def test_instantiation_missing_parameters(self):
        # overwrite sample and parameters
        class Dummy(dimod.Sampler):
            def sample(self, bqm):
                # just override
                pass

            # @property
            # def parameters(self):
            #     return {}

            @property
            def properties(self):
                return {}

        with self.assertRaises(TypeError):
            Dummy()

    def test_instantiation_missing_sample(self):
        # overwrite sample and parameters
        class Dummy(dimod.Sampler):
            # def sample(self, bqm):
            #     # just override
            #     pass

            @property
            def parameters(self):
                return {}

            @property
            def properties(self):
                return {}

        with self.assertRaises(TypeError):
            Dummy()

    def test_instantiation_overwrite_sample(self):
        class Dummy(dimod.Sampler):
            def sample(self, bqm):
                # just override
                pass

            @property
            def parameters(self):
                return {}

            @property
            def properties(self):
                return {}

        sampler = Dummy()

        self.assertTrue(hasattr(sampler, 'sample'),
                        "sampler must have a 'sample' method")
        self.assertTrue(callable(sampler.sample),
                        "sampler must have a 'sample' method")

        self.assertTrue(hasattr(sampler, 'sample_ising'),
                        "sampler must have a 'sample_ising' method")
        self.assertTrue(callable(sampler.sample_ising),
                        "sampler must have a 'sample_ising' method")

        self.assertTrue(hasattr(sampler, 'sample_qubo'),
                        "sampler must have a 'sample_qubo' method")
        self.assertTrue(callable(sampler.sample_qubo),
                        "sampler must have a 'sample_qubo' method")

        self.assertTrue(hasattr(sampler, 'parameters'),
                        "sampler must have a 'parameters' property")
        self.assertFalse(callable(sampler.parameters),
                         "sampler must have a 'parameters' property")

        self.assertTrue(hasattr(sampler, 'properties'),
                        "sampler must have a 'properties' property")
        self.assertFalse(callable(sampler.properties),
                         "sampler must have a 'properties' property")

    def test_instantiation_overwrite_sample_ising(self):
        class Dummy(dimod.Sampler):
            def sample_ising(self, h, J):
                # just override
                pass

            @property
            def parameters(self):
                return {}

            @property
            def properties(self):
                return {}

        sampler = Dummy()

        self.assertTrue(hasattr(sampler, 'sample'),
                        "sampler must have a 'sample' method")
        self.assertTrue(callable(sampler.sample),
                        "sampler must have a 'sample' method")

        self.assertTrue(hasattr(sampler, 'sample_ising'),
                        "sampler must have a 'sample_ising' method")
        self.assertTrue(callable(sampler.sample_ising),
                        "sampler must have a 'sample_ising' method")

        self.assertTrue(hasattr(sampler, 'sample_qubo'),
                        "sampler must have a 'sample_qubo' method")
        self.assertTrue(callable(sampler.sample_qubo),
                        "sampler must have a 'sample_qubo' method")

        self.assertTrue(hasattr(sampler, 'parameters'),
                        "sampler must have a 'parameters' property")
        self.assertFalse(callable(sampler.parameters),
                         "sampler must have a 'parameters' property")

        self.assertTrue(hasattr(sampler, 'properties'),
                        "sampler must have a 'properties' property")
        self.assertFalse(callable(sampler.properties),
                         "sampler must have a 'properties' property")

    def test_instantiation_overwrite_sample_qubo(self):
        class Dummy(dimod.Sampler):
            def sample_qubo(self, Q):
                # just override
                pass

            @property
            def parameters(self):
                return {}

            @property
            def properties(self):
                return {}

        sampler = Dummy()

        self.assertTrue(hasattr(sampler, 'sample'),
                        "sampler must have a 'sample' method")
        self.assertTrue(callable(sampler.sample),
                        "sampler must have a 'sample' method")

        self.assertTrue(hasattr(sampler, 'sample_ising'),
                        "sampler must have a 'sample_ising' method")
        self.assertTrue(callable(sampler.sample_ising),
                        "sampler must have a 'sample_ising' method")

        self.assertTrue(hasattr(sampler, 'sample_qubo'),
                        "sampler must have a 'sample_qubo' method")
        self.assertTrue(callable(sampler.sample_qubo),
                        "sampler must have a 'sample_qubo' method")

        self.assertTrue(hasattr(sampler, 'parameters'),
                        "sampler must have a 'parameters' property")
        self.assertFalse(callable(sampler.parameters),
                         "sampler must have a 'parameters' property")

        self.assertTrue(hasattr(sampler, 'properties'),
                        "sampler must have a 'properties' property")
        self.assertFalse(callable(sampler.properties),
                         "sampler must have a 'properties' property")

    def test_instantiation_overwrite_sample_ising_and_call_sample(self):
        class Dummy(dimod.Sampler):
            def sample_ising(self, h, J):
                return dimod.Response.from_samples([[-1, 1]], {"energy": [0.05]}, {}, dimod.SPIN)

            @property
            def parameters(self):
                return {}

            @property
            def properties(self):
                return {}

        sampler = Dummy()
        bqm = dimod.BinaryQuadraticModel({0: 0.1, 1: -0.3}, {(0, 1): -1}, 0.0, dimod.BINARY)
        resp = sampler.sample(bqm)
        expected_resp = dimod.Response.from_samples([[0, 1]], {"energy": [-0.3]}, {}, dimod.BINARY)
        np.testing.assert_almost_equal(resp.record.sample, expected_resp.record.sample)
        np.testing.assert_almost_equal(resp.record.energy, expected_resp.record.energy)

    def test_instantiation_overwrite_sample_qubo_and_call_sample(self):
        class Dummy(dimod.Sampler):
            def sample_qubo(self, Q):
                return dimod.Response.from_samples([[0, 1]], {"energy": [1.4]}, {}, dimod.BINARY)

            @property
            def parameters(self):
                return {}

            @property
            def properties(self):
                return {}

        sampler = Dummy()
        bqm = dimod.BinaryQuadraticModel({0: 0.1, 1: -0.3}, {(0, 1): -1}, 0.1, dimod.SPIN)
        resp = sampler.sample(bqm)
        expected_resp = dimod.Response.from_samples([[-1, 1]], {"energy": [0.7]}, {}, dimod.SPIN)
        np.testing.assert_almost_equal(resp.record.sample, expected_resp.record.sample)
        np.testing.assert_almost_equal(resp.record.energy, expected_resp.record.energy)

    def test_sampler_can_return_integer_energy_values(self):
        class Dummy(dimod.Sampler):
            def sample_qubo(self, Q):
                return dimod.Response.from_samples([[1]], {"energy": [-3]}, {}, dimod.BINARY)

            @property
            def parameters(self):
                return {}

            @property
            def properties(self):
                return {}

        sampler = Dummy()
        bqm = dimod.BinaryQuadraticModel({0: -3}, {}, 0, dimod.BINARY)
        resp = sampler.sample(bqm)
        expected_resp = dimod.Response.from_samples([[1]], {"energy": [-3]}, {}, dimod.BINARY)
        np.testing.assert_almost_equal(resp.record.sample, expected_resp.record.sample)
        np.testing.assert_almost_equal(resp.record.energy, expected_resp.record.energy)

    def test_spin_bqm_to_sample_ising(self):

        class CountBQM(dimod.BinaryQuadraticModel):
            # should never have vartype changed
            def change_vartype(self, *args, **kwargs):
                raise RuntimeError

            def to_qubo(self):
                raise RuntimeError

        class Ising(dimod.Sampler):
            parameters = None
            properties = None

            def sample_ising(self, h, J):
                bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
                samples = [1]*len(bqm)
                return dimod.SampleSet.from_samples_bqm(samples, bqm)

        sampler = Ising()
        cbqm = CountBQM.from_ising({0: -3}, {(0, 1): -1.5}, offset=1.3)
        sampleset = sampler.sample(cbqm)
        dimod.testing.assert_response_energies(sampleset, cbqm)

    def test_binary_bqm_to_sample_qubo(self):

        class CountBQM(dimod.BinaryQuadraticModel):
            # should never have vartype changed
            def change_vartype(self, *args, **kwargs):
                raise RuntimeError

            def to_ising(self):
                raise RuntimeError

        class Qubo(dimod.Sampler):
            parameters = None
            properties = None

            def sample_qubo(self, Q):
                bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
                samples = [1]*len(bqm)
                return dimod.SampleSet.from_samples_bqm(samples, bqm)

        sampler = Qubo()
        cbqm = CountBQM.from_qubo({(0, 0): -3, (1, 0): 1.5}, offset=.5)
        sampleset = sampler.sample(cbqm)
        dimod.testing.assert_response_energies(sampleset, cbqm)
