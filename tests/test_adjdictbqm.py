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

import unittest

import dimod


class TestConstruction(unittest.TestCase):
    def test_new_bqm(self):
        bqm = dimod.BQM({'a': 1}, {'ab': 3}, 6, 'SPIN')
        with self.assertWarns(DeprecationWarning):
            new = dimod.AdjDictBQM(bqm)
        self.assertEqual(bqm.linear, new.linear)
        self.assertEqual(bqm.quadratic, new.quadratic)
        self.assertEqual(bqm.offset, new.offset)
        self.assertEqual(bqm.vartype, new.vartype)
