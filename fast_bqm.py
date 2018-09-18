from collections.abc import Collection, Mapping
from itertools import chain

import numpy as np

from dimod.views import VariableIndexView, IndexView


class QuadraticView(Mapping):
    __slots__ = 'bqm',

    def __init__(self, bqm):
        self.bqm = bqm

    def __getitem__(self, interaction):
        u, v = interaction
        return self.bqm.adj[u][v]

    def __iter__(self):
        bqm = self.bqm
        variables = bqm.variables
        for r, c in zip(bqm.irow, bqm.icol):
            yield variables[r], variables[c]

    def __len__(self):
        return len(self.bqm.qdata)

    def __str__(self):
        return str(dict(self))


class NeighbourView(Mapping):
    __slots__ = '_index', '_data'

    def __init__(self, index, data):
        self._index = index
        self._data = data

    def __getitem__(self, v):
        return self._data[self._index[v]]

    def __iter__(self):
        return iter(self._index)

    def __len__(self):
        return len(self._index)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self._index, self._data)

    def __str__(self):
        return str(dict(self))


class AdjacencyView(Mapping):
    def __init__(self, iadj, qdata):
        self.iadj = iadj
        self.qdata = qdata

    def __getitem__(self, v):
        return NeighbourView(self.iadj[v], self.qdata)

    def __iter__(self):
        return iter(self.iadj)

    def __len__(self):
        return len(self.adj)

    def __str__(self):
        return str({v: dict(neighbourhood) for v, neighbourhood in self.items()})


def reduce_coo(row, col, data, dtype=None, index_dtype=None):
    # method adapted from scipy's coo_matrix
    #
    # Copyright (c) 2001, 2002 Enthought, Inc.
    # All rights reserved.
    #
    # Copyright (c) 2003-2017 SciPy Developers.
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are met:
    #
    #   a. Redistributions of source code must retain the above copyright notice,
    #      this list of conditions and the following disclaimer.
    #   b. Redistributions in binary form must reproduce the above copyright
    #      notice, this list of conditions and the following disclaimer in the
    #      documentation and/or other materials provided with the distribution.
    #   c. Neither the name of Enthought nor the names of the SciPy Developers
    #      may be used to endorse or promote products derived from this software
    #      without specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
    # BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
    # OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    # THE POSSIBILITY OF SUCH DAMAGE.
    #

    row = np.asarray(row, dtype=index_dtype)
    col = np.asarray(col, dtype=index_dtype)
    data = np.asarray(data, dtype=dtype)

    # row index should be less than col index, this handles upper-triangular vs lower-triangular
    swaps = row > col
    row[swaps], col[swaps] = col[swaps], row[swaps]

    # sorted lexigraphically
    order = np.lexsort((row, col))
    row = row[order]
    col = col[order]
    data = data[order]

    unique = np.append(True, ((row[1:] != row[:-1]) | (col[1:] != col[:-1])))

    row = row[unique]
    col = col[unique]
    unique_idxs, = np.nonzero(unique)
    data = np.add.reduceat(data, unique_idxs, dtype=data.dtype)

    return row, col, data


class FastBQM(Collection):
    """

    Args:
        linear (Mapping[variable, bias])

        quadratic (Mapping[tuple[variable, variable], bias])

    """
    def __init__(self, linear, quadratic, offset, vartype, labels=None, dtype=np.float, index_dtype=np.int64):

        #
        # variables
        #

        if labels is None:
            # if labels are none, we derive from linear/quadratic and set the labels last

            if isinstance(linear, Mapping) and isinstance(quadratic, Mapping):
                linear_labels = linear.keys()
            else:
                linear_labels = range(len(linear))

            if isinstance(quadratic, Mapping):
                quadratic_labels = (v for interaction in quadratic.keys() for v in interaction)
            elif isinstance(quadratic, tuple) and len(quadratic) == 3:
                row, col, _ = quadratic
                quadratic_labels = range(max(max(row), max(col)) + 1)
            else:
                quadratic_labels = range(len(quadratic))

            labels = chain(linear_labels, quadratic_labels)

        self.variables = variables = VariableIndexView(labels)

        #
        # linear biases
        #

        if isinstance(linear, Mapping):
            self.ldata = ldata = np.fromiter((linear.get(v, 0.0) for v in variables),
                                             count=len(variables), dtype=dtype)
        else:
            self.ldata = ldata = np.fromiter((linear[variables.index(v)] for v in variables),
                                             count=len(variables), dtype=dtype)

        self.linear = IndexView(variables, ldata)

        #
        # quadratic biases
        #

        self.iadj = iadj = {v: {} for v in variables}
        if isinstance(quadratic, Mapping):

            row, col, data = zip(*((variables.index(u), variables.index(v), bias)
                                   for (u, v), bias in quadratic.items()))

        elif isinstance(quadratic, tuple) and len(quadratic) == 3:
            row, col, data = quadratic
        else:
            # assume dense matrix
            quadratic = np.atleast_2d(np.asarray(quadratic, dtype=dtype))

            if quadratic.ndim > 2:
                raise ValueError

            row, col = quadratic.nonzero()
            data = quadratic[row, col]

        irow, icol, qdata = reduce_coo(row, col, data, dtype=dtype, index_dtype=index_dtype)

        for idx, (ir, ic) in enumerate(zip(irow, icol)):
            u = variables[ir]
            v = variables[ic]
            iadj[u][v] = iadj[v][u] = idx

        self.irow = irow
        self.icol = icol
        self.qdata = qdata

        self.adj = adj = AdjacencyView(iadj, qdata)
        self.quadratic = QuadraticView(self)

        #
        # offset and vartype
        #

        self.offset = offset
        self.vartype = vartype

    def __contains__(self, v):
        return v in self.variables

    def __iter__(self):
        return iter(self.variables)

    def __len__(self):
        return len(self.variables)

    def __repr__(self):
        return '{}({}, {}, {}, {})'.format(self.__class__.__name__, self.linear,
                                           self.quadratic, self.offset, self.vartype)

    def __eq__(self, other):
        return (self.vartype == other.vartype and self.offset == other.offset
                and self.linear == other.linear and self.adj == other.adj)


if __name__ == '__main__':
    import unittest
    import dimod
    import itertools

    import numpy as np

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

    unittest.main()
