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
import itertools

from itertools import groupby

import numpy as np

import dimod

from dimod import ising_to_qubo, qubo_to_ising, ising_energy, qubo_energy
from dimod.exceptions import WriteableError


class TestIsingEnergy(unittest.TestCase):
    def test_trivial(self):
        en = ising_energy({}, {}, {})
        self.assertEqual(en, 0)

    def test_typical(self):
        # AND gate
        h = {0: -.5, 1: 0, 2: 1, 3: -.5}
        J = {(0, 2): -1, (1, 2): -1, (0, 3): .5, (1, 3): -1}

        en0 = min(ising_energy({0: -1, 1: -1, 2: -1, 3: -1}, h, J),
                  ising_energy({0: -1, 1: -1, 2: -1, 3: +1}, h, J))
        en1 = min(ising_energy({0: +1, 1: -1, 2: -1, 3: -1}, h, J),
                  ising_energy({0: +1, 1: -1, 2: -1, 3: +1}, h, J))
        en2 = min(ising_energy({0: -1, 1: +1, 2: -1, 3: -1}, h, J),
                  ising_energy({0: -1, 1: +1, 2: -1, 3: +1}, h, J))
        en3 = min(ising_energy({0: +1, 1: +1, 2: +1, 3: -1}, h, J),
                  ising_energy({0: +1, 1: +1, 2: +1, 3: +1}, h, J))

        self.assertEqual(en0, en1)
        self.assertEqual(en0, en2)
        self.assertEqual(en0, en3)


class TestQuboEnergy(unittest.TestCase):
    def test_trivial(self):
        en = qubo_energy({}, {})
        self.assertEqual(en, 0)

    def test_typical(self):
        # AND gate
        h = {0: -.5, 1: 0, 2: 1, 3: -.5}
        J = {(0, 2): -1, (1, 2): -1, (0, 3): .5, (1, 3): -1}
        Q, __ = ising_to_qubo(h, J)

        en0 = min(qubo_energy({0: 0, 1: 0, 2: 0, 3: 0}, Q),
                  qubo_energy({0: 0, 1: 0, 2: 0, 3: 1}, Q))
        en1 = min(qubo_energy({0: 1, 1: 0, 2: 0, 3: 0}, Q),
                  qubo_energy({0: 1, 1: 0, 2: 0, 3: 1}, Q))
        en2 = min(qubo_energy({0: 0, 1: 1, 2: 0, 3: 0}, Q),
                  qubo_energy({0: 0, 1: 1, 2: 0, 3: 1}, Q))
        en3 = min(qubo_energy({0: 1, 1: 1, 2: 1, 3: 0}, Q),
                  qubo_energy({0: 1, 1: 1, 2: 1, 3: 1}, Q))

        self.assertEqual(en0, en1)
        self.assertEqual(en0, en2)
        self.assertEqual(en0, en3)


class TestIsingToQubo(unittest.TestCase):
    def test_trivial(self):
        q, offset = ising_to_qubo({}, {})
        self.assertEqual(q, {})
        self.assertEqual(offset, 0)

    def test_no_zeros(self):
        q, offset = ising_to_qubo({0: 0, 0: 0, 0: 0}, {(0, 0): 0, (4, 5): 0})
        self.assertEqual(q, {(0, 0): 0.0})
        self.assertEqual(offset, 0)

    def test_j_diag(self):
        q, offset = ising_to_qubo({}, {(0, 0): 1, (300, 300): 99})
        self.assertEqual(q, {(0, 0): 0.0, (300, 300): 0.0})
        self.assertEqual(offset, 100)

    def test_typical(self):
        h = {i: v for i, v in enumerate([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])}
        j = {(0, 5): 2, (0, 8): 4, (1, 4): -5, (1, 7): 1, (2, 0): 5,
             (2, 1): 4, (3, 0): -1, (3, 6): -3, (3, 8): 3, (4, 0): 2, (4, 7): 3,
             (4, 9): 3, (5, 1): 3, (6, 5): -4, (6, 7): -4, (7, 1): -4,
             (7, 8): 3, (8, 2): -4, (8, 3): -3, (8, 6): -5, (8, 7): -4, (9, 0): 4,
             (9, 1): -1, (9, 4): -5, (9, 7): 3}
        q, offset = ising_to_qubo(h, j)
        # norm_q = normalized_matrix(q)
        ans = {(0, 0): -42, (0, 2): 20, (0, 3): -4, (0, 4): 8,
               (0, 5): 8, (0, 8): 16, (0, 9): 16, (1, 1): -4,
               (1, 2): 16, (1, 4): -20, (1, 5): 12, (1, 7): -12,
               (1, 9): -4, (2, 2): -16, (2, 8): -16, (3, 3): 4,
               (3, 6): -12, (4, 4): 2, (4, 7): 12, (4, 9): -8,
               (5, 5): -2, (5, 6): -16, (6, 6): 34, (6, 7): -16,
               (6, 8): -20, (7, 7): 8, (7, 8): -4, (7, 9): 12,
               (8, 8): 18}
        for (u, v), bias in normalized_matrix(q).items():
            self.assertIn((u, v), ans)
            self.assertEqual(bias, ans[(u, v)])

        self.assertEqual(offset, 2)

    def test_energy(self):
        h = {v: v for v in range(0, 100, 2)}
        h.update({v: -(1 / v) for v in range(1, 100, 2)})
        J = {(u, v): 2 * (u / 3) + v ** .5 for (u, v) in itertools.combinations(range(100), 2)}

        spin_sample = {v: 1 if v % 2 else -1 for v in h}
        bin_sample = {v: 1 if v % 2 else 0 for v in h}

        Q, off = ising_to_qubo(h, J)

        ising_en = ising_energy(spin_sample, h, J)
        qubo_en = qubo_energy(bin_sample, Q)

        self.assertAlmostEqual(ising_en, qubo_en + off)

    def test_offset_propogation(self):
        h = {v: 1 / (v + 1) for v in range(10)}
        J = {(u, v): 2 * (u / 3) + v ** .5 for (u, v) in itertools.combinations(range(10), 2)}

        Q, offset = ising_to_qubo(h, J)

        Q, offset2 = ising_to_qubo(h, J, offset=3)

        self.assertAlmostEqual(offset + 3, offset2)

    def test_underspecified_h(self):
        h = {}
        J = {'ab': -1}

        Q, offset = ising_to_qubo(h, J)

        self.assertEqual(Q, {('a', 'b'): -4, ('a', 'a'): 2, ('b', 'b'): 2})
        self.assertEqual(offset, -1.0)


class TestQuboToIsing(unittest.TestCase):
    def test_trivial(self):
        h, j, offset = qubo_to_ising({})
        self.assertEqual(h, {})
        self.assertEqual(j, {})
        self.assertEqual(offset, 0)

    def test_no_zeros(self):
        h, j, offset = qubo_to_ising({(0, 0): 0, (4, 5): 0})
        self.assertEqual(h, {0: 0, 4: 0, 5: 0})
        self.assertEqual(j, {})
        self.assertEqual(offset, 0)

    def test_typical(self):
        q = {(0, 0): 4, (0, 3): 5, (0, 5): 4, (1, 1): 5, (1, 6): 1, (1, 7): -2,
             (1, 9): -3, (3, 0): -2, (3, 1): 2, (4, 5): 4, (4, 8): 2, (4, 9): -1,
             (5, 1): 2, (5, 6): -5, (5, 8): -4, (6, 0): 1, (6, 5): 2, (6, 6): -4,
             (6, 7): -2, (7, 0): -2, (7, 5): -3, (7, 6): -5, (7, 7): -3, (7, 8): 1,
             (8, 0): 2, (8, 5): 1, (9, 7): -3}
        h, j, offset = qubo_to_ising(q)
        self.assertEqual(h, {0: 4.0, 1: 2.5, 3: 1.25, 4: 1.25, 5: 0.25,
                             6: -4.0, 7: -5.5, 8: 0.5, 9: -1.75})
        norm_j = normalized_matrix(j)
        self.assertEqual(norm_j, {(0, 3): 0.75, (0, 5): 1, (0, 6): 0.25, (0, 7): -0.5,
                                  (0, 8): 0.5, (1, 3): 0.5, (1, 5): 0.5, (1, 6): 0.25,
                                  (1, 7): -0.5, (1, 9): -0.75, (4, 5): 1, (4, 8): 0.5,
                                  (4, 9): -0.25, (5, 6): -0.75, (5, 7): -0.75,
                                  (5, 8): -0.75, (6, 7): -1.75, (7, 8): 0.25,
                                  (7, 9): -0.75})
        self.assertEqual(offset, -0.25)


class TestUtilitiesIntegration(unittest.TestCase):
    def test_start_from_binary(self):
        h = {i: v for i, v in enumerate([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4])}
        j = {(0, 5): 2, (0, 8): 4, (1, 4): -5, (1, 7): 1, (2, 0): 5,
             (2, 1): 4, (3, 0): -1, (3, 6): -3, (3, 8): 3, (4, 0): 2, (4, 7): 3,
             (4, 9): 3, (5, 1): 3, (6, 5): -4, (6, 7): -4, (7, 1): -4,
             (7, 8): 3, (8, 2): -4, (8, 3): -3, (8, 6): -5, (8, 7): -4, (9, 0): 4,
             (9, 1): -1, (9, 4): -5, (9, 7): 3}
        ioff = 1.7

        q, qoff = ising_to_qubo(h, j, ioff)

        bin_sample = {}
        ising_sample = {}
        for v in h:
            bin_sample[v] = 1
            ising_sample[v] = 1

        self.assertAlmostEqual(ising_energy(ising_sample, h, j, ioff),
                               qubo_energy(bin_sample, q, qoff))

    def test_start_from_spin(self):
        Q = {(0, 0): 4, (0, 3): 5, (0, 5): 4, (1, 1): 5, (1, 6): 1, (1, 7): -2,
             (1, 9): -3, (3, 0): -2, (3, 1): 2, (4, 5): 4, (4, 8): 2, (4, 9): -1,
             (5, 1): 2, (5, 6): -5, (5, 8): -4, (6, 0): 1, (6, 5): 2, (6, 6): -4,
             (6, 7): -2, (7, 0): -2, (7, 5): -3, (7, 6): -5, (7, 7): -3, (7, 8): 1,
             (8, 0): 2, (8, 5): 1, (9, 7): -3}
        qoff = 1.3

        h, J, ioff = qubo_to_ising(Q, qoff)

        bin_sample = {}
        ising_sample = {}
        for v in h:
            bin_sample[v] = 0
            ising_sample[v] = -1

        self.assertAlmostEqual(ising_energy(ising_sample, h, J, ioff),
                               qubo_energy(bin_sample, Q, qoff))


def normalized_matrix(mat):
    def key_fn(x):
        return x[0]

    smat = sorted(((sorted(k), v) for k, v in mat.items()), key=key_fn)
    return dict((tuple(k), s) for k, g in groupby(smat, key=key_fn) for s in
                [sum(v for _, v in g)] if s != 0)


class TestChildStructureDFS(unittest.TestCase):
    def test_sampler(self):
        # not a composed sampler

        nodelist = list(range(5))
        edgelist = list(itertools.combinations(nodelist, 2))

        class Dummy(dimod.Structured):
            @property
            def nodelist(self):
                return nodelist

            @property
            def edgelist(self):
                return edgelist

        sampler = Dummy()

        structure = dimod.child_structure_dfs(sampler)
        self.assertEqual(structure.nodelist, nodelist)
        self.assertEqual(structure.edgelist, edgelist)

    def test_composed_sampler(self):
        nodelist = list(range(5))
        edgelist = list(itertools.combinations(nodelist, 2))

        structured_sampler = dimod.StructureComposite(dimod.NullSampler(),
                                                      nodelist, edgelist)

        sampler = dimod.TrackingComposite(structured_sampler)

        structure = dimod.child_structure_dfs(sampler)
        self.assertEqual(structure.nodelist, nodelist)
        self.assertEqual(structure.edgelist, edgelist)

    def test_unstructured_sampler(self):
        with self.assertRaises(ValueError):
            dimod.child_structure_dfs(dimod.NullSampler())

        nested = dimod.TrackingComposite(dimod.NullSampler())

        with self.assertRaises(ValueError):
            dimod.child_structure_dfs(nested)


class TestAsIntegerArrays(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(TypeError):
            dimod.utilities.asintegerarrays()

    def test_min_itemsize(self):
        arrint8 = np.ones(5, dtype=np.int8)
        arrint16 = np.ones(5, dtype=np.int16)

        new0, new1 = dimod.utilities.asintegerarrays(
            arrint8, arrint16, min_itemsize=4)
        self.assertEqual(new0.dtype, np.int32)
        self.assertEqual(new1.dtype, np.int32)

    def test_partial_passthrough(self):
        arrint8 = np.ones(5, dtype=np.int8)
        arrint16 = np.ones(5, dtype=np.int16)

        new0, new1 = dimod.utilities.asintegerarrays(
            arrint8, arrint16, min_itemsize=2)
        self.assertEqual(new0.dtype, np.int16)
        self.assertEqual(new1.dtype, np.int16)
        self.assertIs(arrint16, new1)  # passthrough

        arruint8 = np.ones(5, dtype=np.uint8)
        arruint16 = np.ones(5, dtype=np.uint16)

        new2, new3 = dimod.utilities.asintegerarrays(
            arruint8, arruint16, min_itemsize=2)
        self.assertEqual(new2.dtype, np.uint16)
        self.assertEqual(new3.dtype, np.uint16)
        self.assertIs(arruint16, new3)  # passthrough

    def test_single_passthrough(self):
        arr = np.ones(5, dtype=np.uint8)
        new = dimod.utilities.asintegerarrays(arr)

        self.assertIs(arr, new)  # should not copy

    def test_unsafe(self):
        arr0 = np.ones(5, dtype=np.uint64)
        arr1 = np.ones(5, dtype=np.int64)

        with self.assertRaises(TypeError):
            dimod.utilities.asintegerarrays(arr0, arr1)

        with self.assertRaises(TypeError):
            dimod.utilities.asintegerarrays(np.ones(5, dtype=np.float32))


class TestAsNumericArrays(unittest.TestCase):
    def test_complex(self):
        arr = np.ones(5, dtype=complex)
        with self.assertRaises(TypeError):
            dimod.utilities.asnumericarrays(arr)

    def test_empty(self):
        with self.assertRaises(TypeError):
            dimod.utilities.asnumericarrays()

    def test_min_itemsize(self):
        arrint8 = np.ones(5, dtype=np.int8)
        arrint16 = np.ones(5, dtype=np.int16)

        new0, new1 = dimod.utilities.asnumericarrays(
            arrint8, arrint16, min_itemsize=4)
        self.assertEqual(new0.dtype, np.int32)
        self.assertEqual(new1.dtype, np.int32)

    def test_partial_passthrough(self):
        arrint8 = np.ones(5, dtype=np.int8)
        arrint16 = np.ones(5, dtype=np.int16)

        new0, new1 = dimod.utilities.asnumericarrays(
            arrint8, arrint16, min_itemsize=2)
        self.assertEqual(new0.dtype, np.int16)
        self.assertEqual(new1.dtype, np.int16)
        self.assertIs(arrint16, new1)  # passthrough

        arruint8 = np.ones(5, dtype=np.uint8)
        arruint16 = np.ones(5, dtype=np.uint16)

        new2, new3 = dimod.utilities.asnumericarrays(
            arruint8, arruint16, min_itemsize=2)
        self.assertEqual(new2.dtype, np.uint16)
        self.assertEqual(new3.dtype, np.uint16)
        self.assertIs(arruint16, new3)  # passthrough

    def test_single_passthrough(self):
        arr = np.ones(5, dtype=np.uint8)
        new = dimod.utilities.asnumericarrays(arr)

        self.assertIs(arr, new)  # should not copy
