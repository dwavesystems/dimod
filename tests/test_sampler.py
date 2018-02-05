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

    def test_kwarg_overwrite(self):

        kwds_recieved = [None]

        class MockSampler(dimod.Sampler):
            @dimod.decorators.patch_sample_kwargs
            def sample_ising(self, h, J, kwd=1):
                kwds_recieved[0] = kwd
                response = dimod.Response(dimod.SPIN)
                return response

        sampler = MockSampler()

        # should get default
        sampler.sample_ising({}, {})
        self.assertEqual(kwds_recieved, [1])
        sampler.sample_qubo({})
        self.assertEqual(kwds_recieved, [1])
        sampler.sample(dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN))
        self.assertEqual(kwds_recieved, [1])

        # should overwrite everything
        sampler.sample_ising({}, {}, kwd=15)
        self.assertEqual(kwds_recieved, [15])
        sampler.sample_qubo({}, kwd=16)
        self.assertEqual(kwds_recieved, [16])
        sampler.sample(dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN), kwd=17)
        self.assertEqual(kwds_recieved, [17])

        #

        sampler = MockSampler(default_sample_kwargs={'kwd': 4})

        # should get specified default
        sampler.sample_ising({}, {})
        self.assertEqual(kwds_recieved, [4])
        sampler.sample_qubo({})
        self.assertEqual(kwds_recieved, [4])
        sampler.sample(dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN))
        self.assertEqual(kwds_recieved, [4])

        # should overwrite everything
        sampler.sample_ising({}, {}, kwd=15)
        self.assertEqual(kwds_recieved, [15])
        sampler.sample_qubo({}, kwd=16)
        self.assertEqual(kwds_recieved, [16])
        sampler.sample(dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN), kwd=17)
        self.assertEqual(kwds_recieved, [17])
