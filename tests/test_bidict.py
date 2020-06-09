# Copyright 2020 D-Wave Systems Inc.
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

import sys
import unittest

from dimod.bidict import BiDict


class TestForward(unittest.TestCase):
    def test_construction_empty(self):
        bidict = BiDict()
        self.assertEqual(bidict, {})

    @unittest.skipUnless(sys.version_info[1] >= 7, "dicts only orders in 3.7+")
    def test_construction_duplicate_keys(self):
        d = {}
        d['a'] = 1
        d['b'] = 1
        bidict = BiDict(d)
        self.assertEqual(bidict, dict(b=1))
        self.assertEqual(bidict.inverse, {1: 'b'})

    def test_construction_mixed(self):
        bidict = BiDict({'c': 4}, a=1, b=2)
        self.assertEqual(bidict, dict(a=1, b=2, c=4))
        self.assertEqual(bidict.inverse, {1: 'a', 2: 'b', 4: 'c'})

    def test_overwrite(self):
        bidict = BiDict(a=1)
        bidict['b'] = 1
        self.assertEqual(bidict, dict(b=1))
        self.assertEqual(bidict.inverse, {1: 'b'})


class TestInverse(unittest.TestCase):
    def test_delitem(self):
        bidict = BiDict(a=1, b=2)
        del bidict.inverse[1]
        self.assertEqual(bidict, dict(b=2))
        self.assertEqual(bidict.inverse, {2: 'b'})

    def test_inverse_inverse(self):
        bidict = BiDict(a=1, b=2)
        bidict.inverse.inverse['c'] = 3
        self.assertEqual(bidict, {'a': 1, 'b': 2, 'c': 3})
        self.assertEqual(bidict.inverse, {1: 'a', 2: 'b', 3: 'c'})

    def test_simple(self):
        bidict = BiDict(a=1, b=2)
        self.assertEqual(bidict, dict(a=1, b=2))
        self.assertEqual(bidict.inverse, {1: 'a', 2: 'b'})

    def test_setitem(self):
        bidict = BiDict(a=1, b=2)
        bidict.inverse[3] = 'c'
        self.assertEqual(bidict, dict(a=1, b=2, c=3))
        self.assertEqual(bidict.inverse, {1: 'a', 2: 'b', 3: 'c'})

    def test_setitem_overwrite(self):
        bidict = BiDict(a=1, b=2)
        bidict.inverse[2] = 'c'
        self.assertEqual(bidict, dict(a=1, c=2))
        self.assertEqual(bidict.inverse, {1: 'a', 2: 'c'})
