"""Tests for the sampler class"""
import unittest

import dimod

try:
    # py3
    import unittest.mock as mock
except ImportError:
    # py2
    import mock


class TestSamplerClass(unittest.TestCase):
    """Tests for the template Sampler class"""
    def test_sampler_template_error(self):
        """trying to use the base sampler should result in a meaningful error message"""
        with self.assertRaises(dimod.InvalidSampler):
            dimod.Sampler().sample_ising({}, {})

        with self.assertRaises(dimod.InvalidSampler):
            dimod.Sampler().sample_qubo({})

        with self.assertRaises(dimod.InvalidSampler):
            dimod.Sampler().sample(dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN))

    @mock.patch.object(dimod.Sampler, 'sample')
    def test_overwrite_sample(self, mock_method):
        dimod.Sampler().sample_ising({}, {})
        dimod.Sampler().sample_qubo({})

    @mock.patch.object(dimod.Sampler, 'sample_ising')
    def test_overwrite_sample_ising(self, mock_method):
        dimod.Sampler().sample(dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN))
        dimod.Sampler().sample_qubo({})

    @mock.patch.object(dimod.Sampler, 'sample_qubo')
    def test_overwrite(self, mock_method):
        dimod.Sampler().sample(dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN))
        dimod.Sampler().sample_ising({}, {})
