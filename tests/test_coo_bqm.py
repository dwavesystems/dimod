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
# =============================================================================
import unittest

import numpy as np

import dimod

from dimod.bqm.coo_bqm import CooBQM
from dimod.exceptions import WriteableError


# class TestAggregate(unittest.TestCase):
#     def test_duplicate(self):
#         ldata = [0, 1]
#         irow = [0, 0]
#         icol = [1, 1]
#         qdata = [1, 2]

#         bqm = CooBQM(ldata, (irow, icol, qdata), 0, 'SPIN')

#         np.testing.assert_array_equal(bqm.irow, irow)
#         np.testing.assert_array_equal(bqm.icol, icol)
#         np.testing.assert_array_equal(bqm.qdata, qdata)

#         new = bqm.aggregate()

#         new.testing.assert_array_equal(new.irow, [0])
#         new.testing.assert_array_equal(new.icol, [1])
#         new.testing.assert_array_equal(new.qdata, [3])


class TestDenseLinear(unittest.TestCase):
    def test_set_linear(self):
        bqm = CooBQM.from_ising({'b': 1}, {('a', 'b'): -2})

        bqm.linear['a'] = 5

        self.assertEqual(bqm.linear, {'a': 5, 'b': 1})


class TestFromIsing(unittest.TestCase):
    def test_empty(self):
        bqm = CooBQM.from_ising({}, {})

    def test_integer_labelled(self):
        bqm = CooBQM.from_ising({0: -1}, {(0, 1): 1})

        self.assertEqual(bqm.linear, {0: -1, 1: 0})
        self.assertEqual(bqm.quadratic, {(0, 1): 1})

    def test_str_labelled(self):
        bqm = CooBQM.from_ising({'a': -1}, {('a', 'b'): 1})

        self.assertEqual(bqm.linear, {'a': -1, 'b': 0})
        self.assertEqual(bqm.quadratic, {('a', 'b'): 1})

    def test_self_loop(self):
        bqm = CooBQM.from_ising({}, {('a', 'a'): -1})

        self.assertEqual(bqm.linear, {'a': 0})
        self.assertEqual(bqm.quadratic, {('a', 'a'): -1})  # no reduction


class TestWriteable(unittest.TestCase):
    def test_setflags(self):
        bqm = CooBQM.from_ising({'a': -1}, {('a', 'b'): 1})
        self.assertTrue(bqm.is_writeable)

        bqm.relabel({'a': 0})

        self.assertEqual(bqm.linear, {0: -1, 'b': 0})
        self.assertEqual(bqm.quadratic, {(0, 'b'): 1})

        bqm.setflags(write=False)
        self.assertFalse(bqm.is_writeable)

        with self.assertRaises(ValueError):
            bqm.relabel({'b': 1})

        bqm.setflags(write=True)
        self.assertTrue(bqm.is_writeable)

        bqm.relabel({'b': 1})

        self.assertEqual(bqm.linear, {0: -1, 1: 0})
        self.assertEqual(bqm.quadratic, {(0, 1): 1})

    def test_set_linear(self):
        bqm = CooBQM.from_ising({'b': 1}, {('a', 'b'): -2})
        bqm.setflags(write=False)

        with self.assertRaises(WriteableError) as e:
            bqm.linear['a'] = 5

        # make sure the type and attr are correct in the error message
        msg = e.exception.args[0]
        self.assertEqual(msg,
                         ('Cannot set linear bias when {}.{} is '
                          'False'.format(type(bqm).__name__, 'is_writeable')))
