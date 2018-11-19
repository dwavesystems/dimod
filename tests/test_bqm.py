import itertools
import unittest

import dimod
import numpy as np

try:
    import dimod.bqm._utils
except ImportError:
    cext = False
else:
    cext = True


class TestFastBQM(unittest.TestCase):
    @staticmethod
    def check_consistent_fastbqm(bqm):
        TestVectorBQM.check_consistent_vectorbqm(bqm)

        for v in bqm.linear:
            assert v in bqm.adj
        for v in bqm.adj:
            assert v in bqm.linear

        # adjacency and quadratic are self-consistent
        for u, v in bqm.quadratic:
            assert v in bqm.linear
            assert v in bqm.adj
            assert u in bqm.adj[v]

            assert u in bqm.linear
            assert u in bqm.adj
            assert v in bqm.adj[u]

            assert bqm.adj[u][v] == bqm.quadratic[(u, v)]
            assert bqm.adj[v][u] == bqm.adj[u][v]

        for u in bqm.adj:
            for v in bqm.adj[u]:
                assert (u, v) in bqm.quadratic and (v, u) in bqm.quadratic

        # (u, v) and (v, u) are both in quadratic but iteration should be unique
        pairs = set(bqm.quadratic)
        for u, v in pairs:
            assert (v, u) not in pairs

    def test_construction(self):
        lins = [{0: -.5, 1: 0.0},
                {0: -.5},
                [-.5, 0.0],
                np.array([-.5, 0.0])]

        quads = [{(0, 1): -1},
                 {(1, 0): -1},
                 {(0, 1): -1},
                 {(1, 0): -1},
                 {(0, 1): -.5, (1, 0): -.5},
                 [[0, -1], [0, 0]],
                 [[0, 0], [-1, 0]],
                 [[0, -.5], [-.5, 0]],
                 np.asarray([[0, -1], [0, 0]]),
                 ([0], [1], [-1])]

        bqms = [dimod.FastBQM({0: -.5, 1: 0.0}, {(0, 1): -1}, 1.2, dimod.SPIN),
                dimod.FastBQM([0, -.5], {(0, 1): -1}, 1.2, dimod.SPIN, labels=[1, 0]),
                dimod.FastBQM([0, -.5], [[0, -1], [0, 0]], 1.2, dimod.SPIN, labels=[1, 0])]
        bqms.extend(dimod.FastBQM(l, q, 1.2, dimod.SPIN) for l in lins for q in quads)

        for bqm0, bqm1 in itertools.combinations(bqms, 2):
            self.assertEqual(bqm0, bqm1)

        for bqm in bqms:
            self.check_consistent_fastbqm(bqm)

    def test_construction_labels(self):

        lins = [{'a': -.5, 'b': 0.0},
                {'a': -.5},
                [-.5, 0.0],
                np.array([-.5, 0.0])]

        quads = [{'ab': -1},
                 {'ba': -1},
                 {('a', 'b'): -1},
                 {('b', 'a'): -1},
                 {('a', 'b'): -.5, ('b', 'a'): -.5},
                 [[0, -1], [0, 0]],
                 [[0, 0], [-1, 0]],
                 [[0, -.5], [-.5, 0]],
                 np.asarray([[0, -1], [0, 0]]),
                 ([0], [1], [-1])]

        bqms = [dimod.FastBQM({'a': -.5, 'b': 0.0}, {'ab': -1}, 1.2, dimod.SPIN),
                dimod.FastBQM([0, -.5], {'ab': -1}, 1.2, dimod.SPIN, labels=['b', 'a']),
                dimod.FastBQM([0, -.5], [[0, -1], [0, 0]], 1.2, dimod.SPIN, labels=['b', 'a'])]
        bqms.extend(dimod.FastBQM(l, q, 1.2, dimod.SPIN, labels=['a', 'b']) for l in lins for q in quads)

        for bqm0, bqm1 in itertools.combinations(bqms, 2):
            self.assertEqual(bqm0, bqm1)

        for bqm in bqms:
            self.check_consistent_fastbqm(bqm)

    def test_construction_empty(self):
        lins = [{}, [], np.array([])]

        quads = [{}, [[]], [], np.asarray([]), np.asarray([[]]), ([], [], [])]

        bqms = [dimod.FastBQM(l, q, 1.2, dimod.SPIN) for l in lins for q in quads]

        for bqm0, bqm1 in itertools.combinations(bqms, 2):
            self.assertEqual(bqm0, bqm1)

        for bqm in bqms:
            self.check_consistent_fastbqm(bqm)

    def test__eq__(self):
        bqm = dimod.FastBQM({'a': .5}, {'ab': -1, 'bc': 1}, 1.0, dimod.SPIN)
        self.check_consistent_fastbqm(bqm)

        # should test true for self
        self.assertEqual(bqm, bqm)

    def test_energies(self):
        bqm = dimod.FastBQM({'a': .5}, {'ab': -1}, 1.0, dimod.SPIN)
        self.check_consistent_fastbqm(bqm)

        self.assertTrue(all(bqm.energies([{'a': -1, 'b': -1}, {'a': -1, 'b': +1}]) == [-.5, 1.5]))

        self.assertTrue(all(bqm.energies(([[-1, -1], [+1, -1]], ['b', 'a'])) == [-.5, 1.5]))

    def test_energy(self):
        samples = [{'a': -1, 'b': +1},
                   ([-1, +1], 'ab'),
                   ([+1, -1], 'ba'),
                   ]

        bqm = dimod.FastBQM({'b': .12}, {'ab': 1.3}, 8.9, dimod.SPIN)

        energies = list(map(bqm.energy, samples))
        for e, f in zip(energies, energies[1:]):
            self.assertAlmostEqual(e, f)

        with self.assertRaises(ValueError):
            bqm.energy(([[-1, 1], [1, -1]], 'ab'))

    def test_scale(self):
        bqm = dimod.FastBQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.scale(.5)
        self.assertAlmostEqual(bqm.linear, {0: -1., 1: 1.})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -.5})
        self.assertAlmostEqual(bqm.offset, .5)
        self.check_consistent_fastbqm(bqm)

        self.assertAlmostEqual(bqm.to_binary().energy({v: v % 2 for v in bqm.linear}),
                               bqm.to_spin().energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

        with self.assertRaises(TypeError):
            bqm.scale('a')

    def test_scale_exclusions(self):
        bqm = dimod.FastBQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.scale(.5, ignored_variables=[0])
        self.check_consistent_fastbqm(bqm)
        self.assertEqual(bqm, dimod.FastBQM({0: -2, 1: 1}, {(0, 1): -.5}, .5, dimod.SPIN))

        bqm = dimod.FastBQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.scale(.5, ignored_interactions=[(1, 0)])
        self.check_consistent_fastbqm(bqm)
        self.assertEqual(bqm, dimod.FastBQM({0: -1, 1: 1}, {(0, 1): -1.}, .5, dimod.SPIN))

    def test_normalize(self):
        bqm = dimod.FastBQM({0: -2, 1: 2}, {(0, 1): -1}, 1., dimod.SPIN)
        bqm.normalize(.5)
        self.assertAlmostEqual(bqm.linear, {0: -.5, 1: .5})
        self.assertAlmostEqual(bqm.quadratic, {(0, 1): -.25})
        self.assertAlmostEqual(bqm.offset, .25)
        self.check_consistent_fastbqm(bqm)

        self.assertAlmostEqual(bqm.to_binary().energy({v: v % 2 for v in bqm.linear}),
                               bqm.to_spin().energy({v: 2 * (v % 2) - 1 for v in bqm.linear}))

        #

        with self.assertRaises(TypeError):
            bqm.scale('a')

    def test_flip_variable(self):

        # single spin variable, trivial
        bqm = dimod.FastBQM({'a': -1}, {}, 0.0, dimod.SPIN)
        original_bqm = dimod.FastBQM({'a': -1}, {}, 0.0, dimod.SPIN)
        bqm.flip_variable('a')
        self.assertAlmostEqual(bqm.energy({'a': +1}), original_bqm.energy({'a': -1}))
        self.assertAlmostEqual(bqm.energy({'a': -1}), original_bqm.energy({'a': +1}))
        self.check_consistent_fastbqm(bqm)

        bqm.flip_variable('a')  # should return to original
        self.assertEqual(bqm, original_bqm)

        #

        # more complicated spin model
        linear = {v: v * -.43 for v in range(10)}
        quadratic = {(u, v): u * v * -.021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.SPIN
        bqm = dimod.FastBQM(linear, quadratic, offset, vartype)
        original_bqm = dimod.FastBQM(linear, quadratic, offset, vartype)

        bqm.flip_variable(4)
        self.check_consistent_fastbqm(bqm)
        self.assertEqual(linear[4], bqm.linear[4] * -1)
        self.assertNotEqual(bqm, original_bqm)

        sample = {v: 1 for v in linear}
        flipped_sample = sample.copy()
        flipped_sample[4] = -1
        self.assertAlmostEqual(bqm.energy(flipped_sample), original_bqm.energy(sample))

        bqm.flip_variable(4)  # should return to original
        self.assertEqual(bqm, original_bqm)

        #

        # single binary variable
        bqm = dimod.FastBQM({'a': -1}, {}, 0.0, dimod.BINARY)
        original_bqm = dimod.FastBQM({'a': -1}, {}, 0.0, dimod.BINARY)
        bqm.flip_variable('a')
        self.assertAlmostEqual(bqm.energy({'a': 1}), original_bqm.energy({'a': 0}))
        self.assertAlmostEqual(bqm.energy({'a': 0}), original_bqm.energy({'a': 1}))
        self.check_consistent_fastbqm(bqm)

        bqm.flip_variable('a')  # should return to original
        self.assertEqual(bqm, original_bqm)

        #

        linear = {v: v * -.43 for v in range(10)}
        quadratic = {(u, v): u * v * -.021 for u, v in itertools.combinations(linear, 2)}
        offset = -1.2
        vartype = dimod.BINARY
        bqm = dimod.FastBQM(linear, quadratic, offset, vartype)
        original_bqm = dimod.FastBQM(linear, quadratic, offset, vartype)

        bqm.flip_variable(4)
        self.check_consistent_fastbqm(bqm)
        self.assertNotEqual(bqm, original_bqm)

        sample = {v: 1 for v in linear}
        flipped_sample = sample.copy()
        flipped_sample[4] = 0
        self.assertAlmostEqual(bqm.energy(flipped_sample), original_bqm.energy(sample))

        bqm.flip_variable(4)  # should return to original
        self.assertEqual(bqm, original_bqm)

        #

        bqm.flip_variable(100000)  # silent fail

    def test_fix_variable(self):
        # spin model, fix variable to +1
        original_bqm = dimod.FastBQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        bqm = original_bqm.fix_variable('a', +1)

        self.assertAlmostEqual(bqm.energy({'b': +1}), original_bqm.energy({'b': +1, 'a': +1}))
        self.assertAlmostEqual(bqm.energy({'b': -1}), original_bqm.energy({'b': -1, 'a': +1}))
        self.assertAlmostEqual(bqm.to_binary().energy({'b': 1}), original_bqm.to_binary().energy({'b': 1, 'a': 1}))
        self.assertAlmostEqual(bqm.to_binary().energy({'b': 0}), original_bqm.to_binary().energy({'b': 0, 'a': 1}))

        #

        # spin model with binary built, fix variable to +1
        original_bqm = dimod.FastBQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        bqm = original_bqm.fix_variable('a', +1)

        self.assertAlmostEqual(bqm.energy({'b': +1}), original_bqm.energy({'b': +1, 'a': +1}))
        self.assertAlmostEqual(bqm.energy({'b': -1}), original_bqm.energy({'b': -1, 'a': +1}))
        self.assertAlmostEqual(bqm.to_binary().energy({'b': 1}), original_bqm.to_binary().energy({'b': 1, 'a': 1}))
        self.assertAlmostEqual(bqm.to_binary().energy({'b': 0}), original_bqm.to_binary().energy({'b': 0, 'a': 1}))

        #

        # spin model, fix variable to -1
        original_bqm = dimod.FastBQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        bqm = original_bqm.fix_variable('a', -1)

        self.assertAlmostEqual(bqm.energy({'b': +1}), original_bqm.energy({'b': +1, 'a': -1}))
        self.assertAlmostEqual(bqm.energy({'b': -1}), original_bqm.energy({'b': -1, 'a': -1}))
        self.assertAlmostEqual(bqm.to_binary().energy({'b': 1}), original_bqm.to_binary().energy({'b': 1, 'a': 0}))
        self.assertAlmostEqual(bqm.to_binary().energy({'b': 0}), original_bqm.to_binary().energy({'b': 0, 'a': 0}))

        #

        # spin model with binary built, fix variable to -1
        original_bqm = dimod.FastBQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.SPIN)
        bqm = original_bqm.fix_variable('a', -1)

        self.assertAlmostEqual(bqm.energy({'b': +1}), original_bqm.energy({'b': +1, 'a': -1}))
        self.assertAlmostEqual(bqm.energy({'b': -1}), original_bqm.energy({'b': -1, 'a': -1}))
        self.assertAlmostEqual(bqm.to_binary().energy({'b': 1}), original_bqm.to_binary().energy({'b': 1, 'a': 0}))
        self.assertAlmostEqual(bqm.to_binary().energy({'b': 0}), original_bqm.to_binary().energy({'b': 0, 'a': 0}))

        #

        # binary model, fix variable to +1
        original_bqm = dimod.FastBQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        bqm = original_bqm.fix_variable('a', 1)

        self.assertAlmostEqual(bqm.to_spin().energy({'b': +1}), original_bqm.energy({'b': 1, 'a': 1}))
        self.assertAlmostEqual(bqm.to_spin().energy({'b': -1}), original_bqm.energy({'b': 0, 'a': 1}))
        self.assertAlmostEqual(bqm.to_spin().energy({'b': +1}), original_bqm.to_spin().energy({'b': +1, 'a': +1}))
        self.assertAlmostEqual(bqm.to_spin().energy({'b': -1}), original_bqm.to_spin().energy({'b': -1, 'a': +1}))

        #

        original_bqm = dimod.FastBQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        bqm = original_bqm.fix_variable('a', 1)

        self.assertAlmostEqual(bqm.to_spin().energy({'b': +1}), original_bqm.energy({'b': 1, 'a': 1}))
        self.assertAlmostEqual(bqm.to_spin().energy({'b': -1}), original_bqm.energy({'b': 0, 'a': 1}))
        self.assertAlmostEqual(bqm.to_spin().energy({'b': +1}), original_bqm.to_spin().energy({'b': +1, 'a': +1}))
        self.assertAlmostEqual(bqm.to_spin().energy({'b': -1}), original_bqm.to_spin().energy({'b': -1, 'a': +1}))

        #

        # binary model, fix variable to 0
        original_bqm = dimod.FastBQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        bqm = original_bqm.fix_variable('a', 0)

        self.assertAlmostEqual(bqm.to_spin().energy({'b': +1}), original_bqm.energy({'b': 1, 'a': 0}))
        self.assertAlmostEqual(bqm.to_spin().energy({'b': -1}), original_bqm.energy({'b': 0, 'a': 0}))
        self.assertAlmostEqual(bqm.to_spin().energy({'b': +1}), original_bqm.to_spin().energy({'b': +1, 'a': -1}))
        self.assertAlmostEqual(bqm.to_spin().energy({'b': -1}), original_bqm.to_spin().energy({'b': -1, 'a': -1}))

        #

        original_bqm = dimod.FastBQM({'a': .3}, {('a', 'b'): -1}, 1.2, dimod.BINARY)
        bqm = original_bqm.fix_variable('a', 0)

        self.assertAlmostEqual(bqm.to_spin().energy({'b': +1}), original_bqm.energy({'b': 1, 'a': 0}))
        self.assertAlmostEqual(bqm.to_spin().energy({'b': -1}), original_bqm.energy({'b': 0, 'a': 0}))
        self.assertAlmostEqual(bqm.to_spin().energy({'b': +1}), original_bqm.to_spin().energy({'b': +1, 'a': -1}))
        self.assertAlmostEqual(bqm.to_spin().energy({'b': -1}), original_bqm.to_spin().energy({'b': -1, 'a': -1}))

        #

        with self.assertRaises(ValueError):
            bqm.fix_variable('b', -1)  # spin for binary
        with self.assertRaises(ValueError):
            bqm.fix_variable('c', 0)  # 'c' is not a variable

    def test_fix_variables(self):

        num_variables = 100

        def bias(v):
            return v - (num_variables // 2)

        linear = {v: bias(v) for v in range(num_variables)}
        quadratic = {(u, v): bias(u)*bias(v) for u, v in itertools.combinations(range(num_variables), 2)}

        original_bqm = dimod.FastBQM(linear, quadratic, 1.2, dimod.SPIN)

        assignments = {1: -1, 2: 1, 56: -1, 53: 1}

        bqm = original_bqm.fix_variables(assignments)

        original_sample = {v: 1 for v in range(num_variables)}
        original_sample.update(assignments)

        sample = {v: val for v, val in original_sample.items() if v not in assignments}

        self.assertEqual(bqm.energy(sample), original_bqm.energy(original_sample))



class TestVectorBQM(unittest.TestCase):
    @staticmethod
    def check_consistent_vectorbqm(vbqm):
        num_variables = len(vbqm)

        for attr in ['vartype', 'dtype', 'index_dtype', 'offset', 'ldata', 'irow', 'icol', 'qdata', 'iadj']:
            assert hasattr(vbqm, attr)

        assert num_variables == len(vbqm.ldata)

        assert len(vbqm.iadj) == len(vbqm.ldata)
        assert all(v in vbqm.iadj for v in range(num_variables))

        assert len(vbqm.irow) == len(vbqm.icol)
        assert len(vbqm.icol) == len(vbqm.qdata)

        for idx, (i, j) in enumerate(zip(vbqm.irow, vbqm.icol)):
            assert vbqm.iadj[i][j] == idx

        assert vbqm.ldata.dtype is vbqm.dtype
        assert vbqm.qdata.dtype is vbqm.dtype
        assert isinstance(vbqm.offset, vbqm.dtype.type)  # cover object case
        assert vbqm.irow.dtype is vbqm.index_dtype
        assert vbqm.icol.dtype is vbqm.index_dtype

    def test_construction_empty(self):
        vbqm = dimod.VectorBQM([], [], 0.0, dimod.SPIN)

        self.check_consistent_vectorbqm(vbqm)

        np.testing.assert_array_equal(vbqm.ldata, [])
        np.testing.assert_array_equal(vbqm.qdata, [])
        self.assertEqual(vbqm.offset, 0.0)

    def test_construction_dense(self):
        vbqm = dimod.VectorBQM([-1, 1], [[0, 1], [.5, 0]], 1.5, dimod.BINARY)
        self.check_consistent_vectorbqm(vbqm)

        np.testing.assert_array_equal(vbqm.ldata, [-1, 1])
        np.testing.assert_array_equal(vbqm.qdata, [1.5])  # should aggregate
        self.assertEqual(vbqm.offset, 1.5)

    def test_construction_sparse(self):
        lin = np.zeros(6)
        quad = ([5, 4, 3, 2, 1, 0], [4, 3, 2, 1, 0, 1], [-1, .5, .5, 1.3, 1.2, -1])
        vbqm = dimod.VectorBQM(lin, quad, .2, dimod.SPIN)

        self.check_consistent_vectorbqm(vbqm)
        self.assertAlmostEqual(vbqm.qdata[vbqm.iadj[0][1]], .2)

    def test_multiple_constructions(self):
        lins = [[-.5, 0.0],
                np.array([-.5, 0.0])]

        quads = [[[0, -1], [0, 0]],
                 [[0, 0], [-1, 0]],
                 [[0, -.5], [-.5, 0]],
                 np.asarray([[0, -.5], [-.5, 0]]),
                 np.asarray([[0, -1], [0, 0]]),
                 ([0], [1], [-1])]

        bqms = [dimod.VectorBQM(l, q, 1.2, dimod.SPIN) for l in lins for q in quads]

        for b0, b1 in itertools.combinations(bqms, 2):
            self.assertEqual(b0, b1)

    def test_energy(self):
        bqm = dimod.VectorBQM([-1, +1], ([0], [1], [-1]), 1.5, dimod.SPIN)

        self.assertEqual(bqm.energy([-1, -1]), .5)
        self.assertEqual(bqm.energy([-1, +1]), 4.5)
        self.assertEqual(bqm.energy([+1, -1]), .5)
        self.assertEqual(bqm.energy([+1, +1]), .5)

    def test_energies(self):
        bqm = dimod.VectorBQM([-1, +1], ([0], [1], [-1]), 1.5, dimod.SPIN)

        samples = [[-1, -1], [-1, +1], [+1, -1], [+1, +1]]

        np.testing.assert_array_equal(bqm.energies(samples), [.5, 4.5, .5, .5])

    def test_energies_order_linear(self):

        samples = list(itertools.product((0, 1), repeat=3))

        # only linear + offset
        bqm = dimod.VectorBQM([1, 2, 3], [], 1.5, dimod.BINARY)
        rbqm = dimod.VectorBQM([3, 2, 1], [], 1.5, dimod.BINARY)

        renergies = rbqm.energies(samples)
        energies = bqm.energies(samples, order=[2, 1, 0])

        np.testing.assert_array_almost_equal(renergies, energies)

    def test_energies_order_quadratic(self):

        samples = list(itertools.product((0, 1), repeat=3))

        # only quadratic + offset
        bqm = dimod.VectorBQM([0, 0, 0], ([0, 1], [1, 2], [1, 2]), 1.5, dimod.BINARY)
        rbqm = dimod.VectorBQM([0, 0, 0], ([2, 1], [1, 0], [1, 2]), 1.5, dimod.BINARY)

        renergies = rbqm.energies(samples)
        energies = bqm.energies(samples, order=[2, 1, 0])

        np.testing.assert_array_almost_equal(renergies, energies)

    def test_energies_order(self):
        samples = list(itertools.product((0, 1), repeat=3))

        # all
        bqm = dimod.VectorBQM([1, 2, 3], ([0, 1], [1, 2], [1, 2]), 1.5, dimod.BINARY)
        rbqm = dimod.VectorBQM([3, 2, 1], ([2, 1], [1, 0], [1, 2]), 1.5, dimod.BINARY)

        renergies = rbqm.energies(samples)
        energies = bqm.energies(samples, order=[2, 1, 0])

        np.testing.assert_array_almost_equal(renergies, energies)

    @unittest.skipUnless(cext, "No c extension built")
    def test_energies_cpp(self):
        num_variables = 1000
        p = .1
        num_samples = 100

        linear = np.random.rand(num_variables)
        row, col = zip(*(pair for pair in itertools.combinations(range(num_variables), 2) if np.random.rand() < p))
        quad = np.random.rand(len(row))

        vbqm = dimod.VectorBQM(linear, (row, col, quad), np.random.rand(), dimod.BINARY)

        samples = np.random.randint(2, size=(num_samples, num_variables))

        energies_c = vbqm.energies(samples, _use_cpp_ext=True)
        energies_np = vbqm.energies(samples, _use_cpp_ext=False)

        np.testing.assert_array_almost_equal(energies_np, energies_c)
