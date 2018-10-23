import unittest
import dimod
import itertools

import numpy as np

from dimod.bqm.fast_bqm import FastBQM

try:
    from dimod.bqm._helpers import fast_energy
except ImportError:
    _helpers = False
else:
    _helpers = True


class TestFastBQM(unittest.TestCase):

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

        bqms = [FastBQM({0: -.5, 1: 0.0}, {(0, 1): -1}, 1.2, dimod.SPIN),
                FastBQM([0, -.5], {(0, 1): -1}, 1.2, dimod.SPIN, labels=[1, 0]),
                FastBQM([0, -.5], [[0, -1], [0, 0]], 1.2, dimod.SPIN, labels=[1, 0])]
        bqms.extend(FastBQM(l, q, 1.2, dimod.SPIN) for l in lins for q in quads)

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

        bqms = [FastBQM({'a': -.5, 'b': 0.0}, {'ab': -1}, 1.2, dimod.SPIN),
                FastBQM([0, -.5], {'ab': -1}, 1.2, dimod.SPIN, labels=['b', 'a']),
                FastBQM([0, -.5], [[0, -1], [0, 0]], 1.2, dimod.SPIN, labels=['b', 'a'])]
        bqms.extend(FastBQM(l, q, 1.2, dimod.SPIN, labels=['a', 'b']) for l in lins for q in quads)

        for bqm0, bqm1 in itertools.combinations(bqms, 2):
            self.assertEqual(bqm0, bqm1)

    def test_energies(self):
        bqm = FastBQM({'a': .5}, {'ab': -1}, 1.0, dimod.SPIN)

        self.assertTrue(all(bqm.energies([{'a': -1, 'b': -1}, {'a': -1, 'b': +1}]) == [-.5, 1.5]))

        self.assertTrue(all(bqm.energies(([[-1, -1], [+1, -1]], ['b', 'a'])) == [-.5, 1.5]))


@unittest.skipUnless(_helpers, "c++ extensions not built")
class TestFastBQMHelpers(unittest.TestCase):
    def test_small(self):
        fbqm = FastBQM([0, 0], [[0, -1], [0, 0]], 0, dimod.SPIN)

        samples = [[-1, -1], [-1, +1], [+1, -1], [+1, +1]]

        self.assertTrue((fast_energy(fbqm, samples) == [-1, 1, 1, -1]).all())

    def test_small_with_offset(self):
        fbqm = FastBQM([0, 0], [[0, -1], [0, 0]], 1.5, dimod.SPIN)

        samples = [[-1, -1], [-1, +1], [+1, -1], [+1, +1]]

        self.assertTrue((fast_energy(fbqm, samples) == [.5, 2.5, 2.5, .5]).all())
