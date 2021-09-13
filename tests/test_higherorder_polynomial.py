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

import itertools
import unittest

import numpy as np

import dimod
from dimod.higherorder import BinaryPolynomial


class TestConstruction(unittest.TestCase):
    # just that things don't fall down, we'll test correctness when
    # testing other attributes
    def test_from_dict(self):
        BinaryPolynomial({'a': -1, tuple(): 1.3, 'bc': -1, ('a', 'b'): 1}, 'SPIN')

    def test_from_iterator(self):
        BinaryPolynomial(((term, -1) for term in itertools.combinations(range(100), 2)), 'SPIN')

    def test_aggregation(self):
        poly = BinaryPolynomial({'ab': 1, 'ba': 1, ('a', 'b'): 1, ('b', 'a'): 1}, 'SPIN')
        self.assertEqual(poly, BinaryPolynomial({'ab': 4}, 'SPIN'))

    def test_squares_binary(self):
        poly = BinaryPolynomial({'aa': -1}, dimod.BINARY)
        self.assertEqual(poly['a'], -1)

    def test_squares_spin(self):
        poly = BinaryPolynomial({'aa': -1}, dimod.SPIN)
        self.assertEqual(poly[()], -1)

    def test_cubes_spin(self):
        poly = BinaryPolynomial({'aaa': -1}, dimod.SPIN)
        self.assertEqual(poly['a'], -1)


class Test__contains__(unittest.TestCase):
    def test_single_term(self):
        poly = BinaryPolynomial({('a', 'b'): 1}, 'SPIN')
        self.assertIn('ab', poly)
        self.assertIn('ba', poly)
        self.assertIn(('a', 'b'), poly)
        self.assertIn(('b', 'a'), poly)


class Test__del__(unittest.TestCase):
    def test_typical(self):
        poly = BinaryPolynomial({'ab': -1, 'a': 1, 'b': 1}, 'BINARY')
        self.assertEqual(len(poly), 3)
        del poly[('a', 'b')]
        self.assertEqual(len(poly), 2)


class Test__eq__(unittest.TestCase):
    def test_unlike_types(self):
        polydict = {'ab': -1, 'a': 1, 'b': 1}
        self.assertEqual(BinaryPolynomial(polydict, 'SPIN'), polydict)
        self.assertNotEqual(BinaryPolynomial(polydict, 'SPIN'), 1)

    def test_different_terms(self):
        poly1 = BinaryPolynomial({'a': 1, 'b': 1}, 'BINARY')
        poly2 = BinaryPolynomial({'ab': -1, 'a': 1, 'b': 1}, 'BINARY')
        self.assertNotEqual(poly1, poly2)


class Test__len__(unittest.TestCase):
    def test_single_term(self):
        poly = BinaryPolynomial({('a', 'b'): 1}, 'SPIN')
        self.assertEqual(len(poly), 1)

    def test_repeated_term(self):
        poly = BinaryPolynomial({('a', 'b'): 1, 'ba': 1}, 'BINARY')
        self.assertEqual(len(poly), 1)


class Test__getitems__(unittest.TestCase):
    def test_repeated_term(self):
        poly = BinaryPolynomial({'ab': 1, 'ba': 1, ('a', 'b'): 1, ('b', 'a'): 1}, 'BINARY')
        self.assertEqual(poly['ab'], 4)


class Test_energies(unittest.TestCase):
    def test_single_variable(self):
        poly = BinaryPolynomial({'a': -1}, 'SPIN')

        energies = poly.energies(([[-1], [1]], ['a']))
        np.testing.assert_array_equal(energies, [1, -1])


class TestDegree(unittest.TestCase):
    def test_empty(self):
        poly = BinaryPolynomial([], 'BINARY')
        self.assertEqual(poly.degree, 0)

    def test_degree0(self):
        poly = BinaryPolynomial.from_hising({}, {}, 0)
        self.assertEqual(poly.degree, 0)

    def test_degree3(self):
        poly = BinaryPolynomial.from_hubo({'abc': -1}, 0)
        self.assertEqual(poly.degree, 3)


class TestScale(unittest.TestCase):
    def test_single_variable(self):
        poly = BinaryPolynomial({'a': -1}, 'SPIN')
        poly.scale(.5)
        self.assertEqual(poly['a'], -.5)

    def test_typical(self):
        poly = BinaryPolynomial({'a': 1, 'ab': 1, '': 1}, 'BINARY')
        poly.scale(2)
        self.assertEqual(poly['a'], 2)
        self.assertEqual(poly['ba'], 2)
        self.assertEqual(poly[tuple()], 2)

    def test_ignore_terms(self):
        poly = BinaryPolynomial({'a': 1, 'ab': 1, '': 1}, 'BINARY')
        poly.scale(2, ignored_terms=['', 'ba'])
        self.assertEqual(poly['a'], 2)
        self.assertEqual(poly['ba'], 1)
        self.assertEqual(poly[tuple()], 1)

    def test_scale_by_float(self):
        poly = BinaryPolynomial({'a': 4}, 'SPIN')
        poly.scale(.25)
        self.assertEqual(poly['a'], 1)


class TestNormalize(unittest.TestCase):
    def test_empty(self):
        poly = BinaryPolynomial({}, 'SPIN')
        poly.normalize()

    def test_typical(self):
        poly = BinaryPolynomial({'a': 1, 'ab': 1, '': 1}, 'BINARY')
        poly.normalize(.5)
        self.assertEqual(poly['a'], .5)
        self.assertEqual(poly['ba'], .5)
        self.assertEqual(poly[tuple()], .5)

    def test_int_division(self):
        # we use division which previous caused issues in python2
        poly = BinaryPolynomial({'a': 4}, 'SPIN')
        poly.normalize(bias_range=1, poly_range=None, ignored_terms=[])
        self.assertEqual(poly['a'], 1)

    def test_ignored(self):
        poly = BinaryPolynomial({'a': 4, 'b': 2}, 'SPIN')
        poly.normalize(bias_range=1, poly_range=None, ignored_terms=['a'])
        self.assertEqual(poly['a'], 4)  # should not be scaled or used in calcs
        self.assertEqual(poly['b'], 1)  # should be normalized correctly


class TestRelabel(unittest.TestCase):
    def test_swap_partial(self):
        poly = BinaryPolynomial({'abc': 1, 'bc': 1, 'ab': -1}, 'BINARY')
        poly.relabel_variables({'b': 'c', 'c': 'b'})
        self.assertEqual(poly, {'abc': 1, 'bc': 1, 'ac': -1})

    def test_copy(self):
        poly = BinaryPolynomial({'abc': 1, 'bc': 1, 'ab': -1}, 'BINARY')
        new = poly.relabel_variables({'a': 'z'}, inplace=False)
        self.assertEqual(new, {'zbc': 1, 'bc': 1, 'zb': -1})
        self.assertEqual(poly, {'abc': 1, 'bc': 1, 'ab': -1})


class Test_to_binary(unittest.TestCase):
    def test_energy_equivalence_only_offset(self):
        spipoly = BinaryPolynomial({'': 5}, 'SPIN')
        binpoly = spipoly.to_binary()

        variables = list(binpoly.variables)
        for config in itertools.product((0, 1), repeat=len(variables)):
            binary_sample = dict(zip(variables, config))
            spin_sample = {v: 2*x - 1 for v, x in binary_sample.items()}
            self.assertAlmostEqual(spipoly.energy(spin_sample), binpoly.energy(binary_sample))

    def test_energy_equivalence_only_linear(self):
        spipoly = BinaryPolynomial({'a': 5, 'b': -3}, 'SPIN')
        binpoly = spipoly.to_binary()

        variables = list(binpoly.variables)
        for config in itertools.product((0, 1), repeat=len(variables)):
            binary_sample = dict(zip(variables, config))
            spin_sample = {v: 2*x - 1 for v, x in binary_sample.items()}
            self.assertAlmostEqual(spipoly.energy(spin_sample), binpoly.energy(binary_sample))

    def test_energy_equivalence(self):
        spipoly = BinaryPolynomial({'abc': 1, 'bc': 1, 'ab': -1, '': 0}, 'SPIN')
        binpoly = spipoly.to_binary()

        variables = list(binpoly.variables)
        for config in itertools.product((0, 1), repeat=len(variables)):
            binary_sample = dict(zip(variables, config))
            spin_sample = {v: 2*x - 1 for v, x in binary_sample.items()}
            self.assertAlmostEqual(spipoly.energy(spin_sample), binpoly.energy(binary_sample))


class Test_to_spin(unittest.TestCase):
    def test_energy_equivalence_only_offset(self):
        binpoly = BinaryPolynomial({'': 5}, 'BINARY')
        spipoly = binpoly.to_spin()

        variables = list(binpoly.variables)
        for config in itertools.product((0, 1), repeat=len(variables)):
            binary_sample = dict(zip(variables, config))
            spin_sample = {v: 2*x - 1 for v, x in binary_sample.items()}
            self.assertAlmostEqual(spipoly.energy(spin_sample), binpoly.energy(binary_sample))

    def test_energy_equivalence_only_linear(self):
        binpoly = BinaryPolynomial({'a': 5, 'b': -3}, 'BINARY')
        spipoly = binpoly.to_spin()

        variables = list(binpoly.variables)
        for config in itertools.product((0, 1), repeat=len(variables)):
            binary_sample = dict(zip(variables, config))
            spin_sample = {v: 2*x - 1 for v, x in binary_sample.items()}
            self.assertAlmostEqual(spipoly.energy(spin_sample), binpoly.energy(binary_sample))

    def test_energy_equivalence(self):
        binpoly = BinaryPolynomial({'abc': 1, 'bc': 1, 'ab': -1, '': 0}, 'BINARY')
        spipoly = binpoly.to_spin()

        variables = list(binpoly.variables)
        for config in itertools.product((0, 1), repeat=len(variables)):
            binary_sample = dict(zip(variables, config))
            spin_sample = {v: 2*x - 1 for v, x in binary_sample.items()}
            self.assertAlmostEqual(spipoly.energy(spin_sample), binpoly.energy(binary_sample))


class TestHUBO(unittest.TestCase):
    # .to_hubo and .from_hubo
    def test_binary(self):
        poly = BinaryPolynomial({'abc': 1, 'bc': 1, 'ab': -1, '': 0}, 'BINARY')
        H, off = poly.to_hubo()
        new = BinaryPolynomial.from_hubo(H, off)
        self.assertEqual(poly, new)

    def test_spin(self):
        poly = BinaryPolynomial({'abc': 1, 'bc': 1, 'ab': -1, '': 0}, 'SPIN')
        H, off = poly.to_hubo()
        new = BinaryPolynomial.from_hubo(H, off)
        self.assertEqual(poly, new.to_spin())


class TestHising(unittest.TestCase):
    # .to_hising and .from_hising
    def test_binary(self):
        poly = BinaryPolynomial({'abc': 1, 'bc': 1, 'ab': -1}, 'BINARY')
        h, J, off = poly.to_hising()
        new = BinaryPolynomial.from_hising(h, J, off)
        self.assertEqual(poly, new.to_binary())

    def test_spin(self):
        poly = BinaryPolynomial({'abc': 1, 'bc': 1, 'ab': -1}, 'SPIN')
        h, J, off = poly.to_hising()
        new = BinaryPolynomial.from_hising(h, J, off)
        self.assertEqual(poly, new)
