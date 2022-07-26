# Copyright 2021 D-Wave Systems Inc.
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

import pickle
import unittest

import dimod


class TestCopy(unittest.TestCase):
    def test_integer(self):
        from copy import deepcopy

        self.assertIs(dimod.BINARY, deepcopy(dimod.BINARY))
        self.assertIs(dimod.INTEGER, deepcopy(dimod.INTEGER))
        self.assertIs(dimod.SPIN, deepcopy(dimod.SPIN))
        self.assertIs(dimod.REAL, deepcopy(dimod.REAL))


class TestPickle(unittest.TestCase):
    def test_vartypes(self):
        vartypes = list(dimod.Vartype)
        self.assertEqual(vartypes, pickle.loads(pickle.dumps(vartypes)))
