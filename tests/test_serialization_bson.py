import unittest
import dimod
from dimod.serialization.bson import bqm_bson_decoder, bqm_bson_encoder
import numpy as np


try:
    import bson
    _bson_imported = True
except ImportError:
    _bson_imported = False


class TestBSONSerialization(unittest.TestCase):
    def test_empty_bqm(self):
        bqm = dimod.BinaryQuadraticModel.from_qubo({})
        encoded = bqm_bson_encoder(bqm)
        expected_encoding = {
            'as_complete': False,
            'linear': b'',
            'quadratic_vals': b'',
            'variable_type': 'BINARY',
            'offset': 0.0,
            'variable_order': [],
            'index_dtype': '<u2',
            'quadratic_head': b'',
            'quadratic_tail': b'',
        }
        self.assertDictEqual(encoded, expected_encoding)
        decoded = bqm_bson_decoder(encoded)
        self.assertEqual(bqm, decoded)

    def test_single_variable_bqm(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({"a": -1}, {})
        encoded = bqm_bson_encoder(bqm)
        expected_encoding = {
            'as_complete': False,
            'linear': b'\x00\x00\x80\xbf',
            'quadratic_vals': b'',
            'variable_type': 'SPIN',
            'offset': 0.0,
            'variable_order': ['a'],
            'index_dtype': '<u2',
            'quadratic_head': b'',
            'quadratic_tail': b'',
        }
        self.assertDictEqual(encoded, expected_encoding)
        decoded = bqm_bson_decoder(encoded)
        self.assertEqual(bqm, decoded)

    def test_small_bqm(self):
        bqm = dimod.BinaryQuadraticModel.from_ising(
            {"a": 1, "b": 3, "c": 4.5, "d": 0},
            {"ab": -3, "cd": 3.5, "ad": 2}
        )
        encoded = bqm_bson_encoder(bqm)
        expected_encoding = {
            'as_complete': True,
            'linear': b'\x00\x00\x80?\x00\x00@@\x00\x00\x90@\x00\x00\x00\x00',
            'quadratic_vals': b'\x00\x00@\xc0\x00\x00\x00\x00\x00\x00\x00@'
                              b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`@',
            'variable_type': 'SPIN',
            'offset': 0.0,
            'variable_order': ['a', 'b', 'c', 'd'],
            'index_dtype': '<u2',
        }
        self.assertDictEqual(encoded, expected_encoding)
        decoded = bqm_bson_decoder(encoded)

        # no easy way to directly check if the bqm objects are equal (b/c float
        # precision, missing edges), so for now check if the qubo matrices are
        # the same
        var_order = sorted(bqm)
        np.testing.assert_almost_equal(bqm.to_numpy_matrix(var_order),
                                       decoded.to_numpy_matrix(var_order))

    
    @unittest.skipUnless(_bson_imported, "no pymongo bson installed")
    def test_bsonable(self):
        bqm = dimod.BinaryQuadraticModel.from_ising(
            {"a": 1, "b": 3, "c": 4.5, "d": 0},
            {"ab": -3, "cd": 3.5, "ad": 2}
        )
        encoded = bqm_bson_encoder(bqm)
        bson.BSON.encode(encoded)
