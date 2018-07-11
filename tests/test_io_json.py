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
import json

import jsonschema

import dimod
import dimod.io.json as dson


class TestDimodEncoder(unittest.TestCase):
    def test_empty_bqm(self):
        bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)

        dct = dson.DimodEncoder().default(bqm)

        self.assertEqual(dct,
                         {'linear_terms': [],
                          'quadratic_terms': [],
                          'offset': 0.0,
                          'variable_type': 'SPIN',
                          'version': {'dimod': dimod.__version__, 'bqm_schema': '1.0.0'},
                          'variable_labels': [], 'info': {}})

        jsonschema.validate(dct, dson.bqm_json_schema)

    def test_bqm_labels(self):
        bqm = dimod.BinaryQuadraticModel.from_ising({'a': -1, 0: 1, (0,): -1, ((0,),): 1.5},
                                                    {('a', 0): -1, ('a', (0,)): 1, (0, (0,)): .5})

        dct = dson.DimodEncoder().default(bqm)

        jsonschema.validate(dct, dson.bqm_json_schema)

    def test_bqm_complex(self):
        linear = {'a': -1, 4: 1, ('a', "complex key"): 3}
        quadratic = {('a', 'c'): 1.2, ('b', 'c'): .3, ('a', 3): -1}
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 3, dimod.SPIN)

        dct = dson.DimodEncoder().default(bqm)

        jsonschema.validate(dct, dson.bqm_json_schema)
