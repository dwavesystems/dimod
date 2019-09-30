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
"""
Generic dimod/bqm tests.
"""
import itertools

from collections import OrderedDict

import numpy as np

import dimod


class TestBQMAPI:
    """
    Tests for BQMs like AdjArrayBQM (doesn't try to change the shape)
    """

    def test__len__(self):
        bqm = self.BQM(np.ones((107, 107)), dimod.BINARY)
        self.assertEqual(len(bqm), 107)

    def test_construction_args(self):
        # make sure all the orderes are ok

        with self.assertRaises(TypeError):
            self.BQM()

        bqm = self.BQM('SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (0, 0))

        bqm = self.BQM(vartype='SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (0, 0))

        bqm = self.BQM(5, 'SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (5, 0))

        bqm = self.BQM(5, vartype='SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (5, 0))

        bqm = self.BQM(vartype='SPIN', obj=5)
        self.assertEqual(bqm.vartype, dimod.SPIN)
        self.assertEqual(bqm.shape, (5, 0))

    def test_construction_symmetric(self):
        bqm = self.BQM(np.ones((5, 5)), 'BINARY')
        for u, v in itertools.combinations(range(5), 2):
            self.assertEqual(bqm.get_quadratic(u, v), 2)  # added
        for u in range(5):
            self.assertEqual(bqm.get_linear(u), 1)

    # def test_construction_nans(self):
    #     bqm = self.BQM(np.nan((5, 5)))
    #     self.assertEqual(bqm.shape, (5, 0))
    #     for u in range(5):
    #         self.assertEqual(bqm.get_linear(u), 0)

    def test_get_linear_disconnected_string_labels(self):
        bqm = self.BQM(({'a': -1, 'b': 1}, {}), dimod.BINARY)
        self.assertEqual(bqm.get_linear('a'), -1)
        self.assertEqual(bqm.get_linear('b'), 1)
        with self.assertRaises(ValueError):
            bqm.get_linear('c')

    def test_get_linear_disconnected(self):
        bqm = self.BQM(5, dimod.SPIN)

        for v in range(5):
            self.assertEqual(bqm.get_linear(v), 0)

        with self.assertRaises(ValueError):
            bqm.get_linear(-1)

        with self.assertRaises(ValueError):
            bqm.get_linear(5)

    def test_get_quadratic_3x3array(self):
        bqm = self.BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]], dimod.SPIN)

        self.assertEqual(bqm.get_quadratic(0, 1), 1)
        self.assertEqual(bqm.get_quadratic(1, 0), 1)

        self.assertEqual(bqm.get_quadratic(0, 2), 2)
        self.assertEqual(bqm.get_quadratic(2, 0), 2)

        with self.assertRaises(ValueError):
            bqm.get_quadratic(2, 1)
        with self.assertRaises(ValueError):
            bqm.get_quadratic(1, 2)

        with self.assertRaises(ValueError):
            bqm.get_quadratic(0, 0)

    def test_has_variable(self):
        h = OrderedDict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = self.BQM((h, J), dimod.SPIN)

        self.assertTrue(bqm.has_variable('a'))
        self.assertTrue(bqm.has_variable(1))
        self.assertTrue(bqm.has_variable(3))

        # no false positives
        self.assertFalse(bqm.has_variable(0))
        self.assertFalse(bqm.has_variable(2))

    def test_iter_variables(self):
        h = OrderedDict([('a', -1), (1, -1), (3, -1)])
        J = {}
        bqm = self.BQM((h, J), dimod.SPIN)

        self.assertEqual(list(bqm.iter_variables()), ['a', 1, 3])

    def test_set_linear(self):
        # does not change shape
        bqm = self.BQM(np.triu(np.ones((3, 3))), dimod.BINARY)

        self.assertEqual(bqm.get_linear(0), 1)
        bqm.set_linear(0, .5)
        self.assertEqual(bqm.get_linear(0), .5)

    def test_set_quadratic(self):
        # does not change shape
        bqm = self.BQM(np.triu(np.ones((3, 3))), dimod.BINARY)

        self.assertEqual(bqm.get_quadratic(0, 1), 1)
        bqm.set_quadratic(0, 1, .5)
        self.assertEqual(bqm.get_quadratic(0, 1), .5)
        self.assertEqual(bqm.get_quadratic(1, 0), .5)
        bqm.set_quadratic(0, 1, -.5)
        self.assertEqual(bqm.get_quadratic(0, 1), -.5)
        self.assertEqual(bqm.get_quadratic(1, 0), -.5)

    def test_shape_3x3array(self):
        bqm = self.BQM([[0, 1, 2], [0, 0.5, 0], [0, 0, 1]], dimod.BINARY)

        self.assertEqual(bqm.shape, (3, 2))
        self.assertEqual(bqm.num_variables, 3)
        self.assertEqual(bqm.num_interactions, 2)

    def test_shape_disconnected(self):
        bqm = self.BQM(5, dimod.BINARY)

        self.assertEqual(bqm.shape, (5, 0))
        self.assertEqual(bqm.num_variables, 5)
        self.assertEqual(bqm.num_interactions, 0)

    def test_shape_empty(self):
        self.assertEqual(self.BQM(dimod.SPIN).shape, (0, 0))
        self.assertEqual(self.BQM(0, dimod.SPIN).shape, (0, 0))

        self.assertEqual(self.BQM(dimod.SPIN).num_variables, 0)
        self.assertEqual(self.BQM(0, dimod.SPIN).num_variables, 0)

        self.assertEqual(self.BQM(dimod.SPIN).num_interactions, 0)
        self.assertEqual(self.BQM(0, dimod.SPIN).num_interactions, 0)

    def test_vartype(self):
        bqm = self.BQM(vartype='SPIN')
        self.assertEqual(bqm.vartype, dimod.SPIN)

        bqm = self.BQM(vartype=dimod.SPIN)
        self.assertEqual(bqm.vartype, dimod.SPIN)

        bqm = self.BQM(vartype=(-1, 1))
        self.assertEqual(bqm.vartype, dimod.SPIN)

        bqm = self.BQM(vartype='BINARY')
        self.assertEqual(bqm.vartype, dimod.BINARY)

        bqm = self.BQM(vartype=dimod.BINARY)
        self.assertEqual(bqm.vartype, dimod.BINARY)

        bqm = self.BQM(vartype=(0, 1))
        self.assertEqual(bqm.vartype, dimod.BINARY)

    def test_vartype_readonly(self):
        bqm = self.BQM(vartype='SPIN')
        with self.assertRaises(AttributeError):
            bqm.vartype = dimod.BINARY


class TestShapeableBQMAPI(TestBQMAPI):
    def test_add_variable_exception(self):
        bqm = self.BQM(dimod.SPIN)
        with self.assertRaises(TypeError):
            bqm.add_variable([])

    def test_add_variable_labelled(self):
        bqm = self.BQM(dimod.SPIN)
        bqm.add_variable('a')
        bqm.add_variable(1)
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(list(bqm.iter_variables()), ['a', 1])
        bqm.add_variable()
        self.assertEqual(bqm.shape, (3, 0))
        self.assertEqual(list(bqm.iter_variables()), ['a', 1, 2])

    def test_add_variable_int_labelled(self):
        bqm = self.BQM(dimod.SPIN)
        self.assertEqual(bqm.add_variable(1), 1)
        self.assertEqual(bqm.add_variable(), 0)  # 1 is already taken
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(bqm.add_variable(), 2)
        self.assertEqual(bqm.shape, (3, 0))

    def test_add_variable_unlabelled(self):
        bqm = self.BQM(dimod.SPIN)
        bqm.add_variable()
        bqm.add_variable()
        self.assertEqual(bqm.shape, (2, 0))
        self.assertEqual(list(bqm.iter_variables()), [0, 1])

    def test_pop_variable_labelled(self):
        bqm = self.BQM(dimod.SPIN)
        bqm.add_variable('a')
        bqm.add_variable(1)
        bqm.add_variable(0)
        self.assertEqual(bqm.pop_variable(), 0)
        self.assertEqual(bqm.pop_variable(), 1)
        self.assertEqual(bqm.pop_variable(), 'a')
        with self.assertRaises(ValueError):
            bqm.pop_variable()

    def test_pop_variable_unlabelled(self):
        bqm = self.BQM(2, dimod.BINARY)
        self.assertEqual(bqm.pop_variable(), 1)
        self.assertEqual(bqm.pop_variable(), 0)
        with self.assertRaises(ValueError):
            bqm.pop_variable()

    def test_remove_interaction(self):
        bqm = self.BQM(np.triu(np.ones((3, 3))), dimod.BINARY)
        self.assertTrue(bqm.remove_interaction(0, 1))
        self.assertFalse(bqm.remove_interaction(0, 1))
        self.assertEqual(bqm.shape, (3, 2))
        with self.assertRaises(ValueError):
            bqm.get_quadratic(0, 1)

    def test_set_quadratic_exception(self):
        bqm = self.BQM(dimod.SPIN)
        with self.assertRaises(TypeError):
            bqm.set_quadratic([], 1, .5)
        with self.assertRaises(TypeError):
            bqm.set_quadratic(1, [], .5)
