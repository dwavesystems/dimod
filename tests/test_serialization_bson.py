# Copyright 2019 D-Wave Systems Inc.
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
import dimod
from dimod.serialization.bson import bqm_bson_decoder, bqm_bson_encoder
import numpy as np
from six import PY2


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
            'bias_dtype': '<f4',
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
            'bias_dtype': '<f4',
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
        decoded = bqm_bson_decoder(encoded)

        # no easy way to directly check if the bqm objects are equal (b/c float
        # precision, missing edges), so for now check if the qubo matrices are
        # the same
        var_order = sorted(bqm)
        np.testing.assert_almost_equal(bqm.to_numpy_matrix(var_order),
                                       decoded.to_numpy_matrix(var_order))

    def test_complex_variable_names(self):
        linear = {'a': -1, 4: 1, ('a', "complex key"): 3}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3, ('a', 3): -1}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 3, dimod.SPIN,
                                         tag=5)
        encoded = bqm_bson_encoder(bqm)
        decoded = bqm_bson_decoder(encoded)

        # no easy way to directly check if the bqm objects are equal (b/c float
        # precision, missing edges), so for now check if the qubo matrices are
        # the same
        var_order = list(bqm.variables)
        np.testing.assert_almost_equal(bqm.to_numpy_matrix(var_order),
                                       decoded.to_numpy_matrix(var_order),
                                       decimal=6)

    @unittest.skipUnless(_bson_imported, "no pymongo bson installed")
    def test_bsonable(self):
        bqm = dimod.BinaryQuadraticModel.from_ising(
            {"a": 1, "b": 3, "c": 4.5, "d": 0},
            {"ab": -3, "cd": 3.5, "ad": 2}
        )
        encoded = bqm_bson_encoder(bqm,
                                   bytes_type=(bson.Binary if PY2 else bytes))
        bson.BSON.encode(encoded)

    def test_bias_dtype(self):
        bqm = dimod.BinaryQuadraticModel.from_ising(
            {"a": 1, "b": 3, "c": 4.5, "d": 0},
            {"ab": -3, "cd": 3.5, "ad": 2},
        )
        encoded = bqm_bson_encoder(bqm, bias_dtype=np.float16)
        expected_encoding = {
            'as_complete': True,
            'linear': b'\x00<\x00B\x80D\x00\x00',
            'quadratic_vals': b'\x00\xc2\x00\x00\x00@\x00\x00\x00\x00\x00C',
            'variable_type': 'SPIN',
            'offset': 0.0,
            'variable_order': ['a', 'b', 'c', 'd'],
            'index_dtype': '<u2',
            'bias_dtype': '<f2'
        }
        self.assertDictEqual(encoded, expected_encoding)
        decoded = bqm_bson_decoder(encoded)
        var_order = list(bqm.variables)
        np.testing.assert_almost_equal(bqm.to_numpy_matrix(var_order),
                                       decoded.to_numpy_matrix(var_order),
                                       decimal=6)
