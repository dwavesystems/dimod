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
import json

import numpy as np

try:
    import simplejson
except ImportError:
    _simplejson = False
else:
    _simplejson = True

import dimod

from dimod.serialization.json import DimodEncoder, DimodDecoder, dimod_object_hook


class TestEncode(unittest.TestCase):
    def test_builtin(self):
        # non-dimod objects
        obj = [0, 'a', [0, 'a']]

        # shouldn't be any different than without the encoder
        self.assertEqual(json.dumps(obj), json.dumps(obj, cls=DimodEncoder))


class TestDecoder(unittest.TestCase):
    def test_builtin(self):
        # non-dimod objects
        s = '[0, "a", [0, "a"]]'

        # shouldn't be any different than without the encoder
        self.assertEqual(json.loads(s), json.loads(s, cls=DimodDecoder))


class TestFunctional(unittest.TestCase):
    def test_builtin(self):
        # non-dimod objects
        obj = [0, 'a', [0, 'a']]

        new = json.loads(json.dumps(obj, cls=DimodEncoder), cls=DimodDecoder)
        self.assertEqual(obj, new)

    def test_sampleset_empty(self):
        obj = dimod.SampleSet.from_samples([], dimod.SPIN, energy=[])

        new = json.loads(json.dumps(obj, cls=DimodEncoder), cls=DimodDecoder)
        self.assertEqual(obj, new)

    def test_sampleset_triu(self):
        num_variables = 100
        num_samples = 100
        samples = 2*np.triu(np.ones((num_samples, num_variables)), -4) - 1
        bqm = dimod.BinaryQuadraticModel.from_ising({v: .1*v for v in range(num_variables)}, {})
        obj = dimod.SampleSet.from_samples_bqm(samples, bqm)

        new = json.loads(json.dumps(obj, cls=DimodEncoder), cls=DimodDecoder)
        self.assertEqual(obj, new)

    def test_bqm(self):
        num_variables = 100
        bqm = dimod.BinaryQuadraticModel.from_ising({v: .1*v for v in range(num_variables)}, {})

        new = json.loads(json.dumps(bqm.to_serializable(), cls=DimodEncoder), cls=DimodDecoder)
        self.assertEqual(bqm, new)
