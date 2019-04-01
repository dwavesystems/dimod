from collections import OrderedDict

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import numpy as np

from dimod.variables import Variables
from dimod.vartypes import SPIN
from dimod.bqm.cyutils import coo_sort


def reduce_coo(row, col, data):
    """
    """
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

    if row.ndim != 1 or col.ndim != 1 or data.ndim != 1 or len(row) != len(col) or len(col) != len(data):
        raise ValueError("row, col and data should all be vectors of equal length")

    if len(row) == 0:
        # empty arrays are already sorted
        return row, col, data

    coo_sort(row, col, data)

    # reduce unique
    unique = ((row[1:] != row[:-1]) | (col[1:] != col[:-1]))
    if not unique.all():
        # copy
        unique = np.append(True, unique)

        row = row[unique]
        col = col[unique]

        unique_idxs, = np.nonzero(unique)
        data = np.add.reduceat(data, unique_idxs, dtype=data.dtype)

    return row, col, data


class LinearArrayView(abc.Mapping):
    def __init__(self, ldata):
        self.ldata = ldata

    def __getitem__(self, vi):
        # todo, should return a KeyError, not an IndexError
        return self.ldata[vi]

    def __iter__(self):
        return iter(range(len(self)))

    def __len__(self):
        return self.ldata.shape[0]

    def __repr__(self):
        # note: this does not actually construct the object but under the
        # assumption that users won't actually want to construct this directly
        # it does make for much more readable printing
        # todo: do this without casting to a dict first
        return str(dict(self))


class LabelledLinearArrayView(LinearArrayView):
    def __init__(self, ldata, variable_to_idx):
        super(LabelledLinearArrayView, self).__init__(ldata)
        self.variable_to_idx = variable_to_idx

    def __getitem__(self, v):
        return self.ldata[self.variable_to_idx[v]]

    def __iter__(self):
        return iter(self.variable_to_idx)


class QuadraticCooView(abc.Mapping):
    # todo: overwrite iteritems and iterkeys to make them much much faster.
    # also that will deal with duplicates for printing the dict for instance

    def __init__(self, bqm):
        self.bqm = bqm

    def __getitem__(self, tup):
        iu, iv = tup

        bqm = self.bqm
        irow = bqm.irow
        icol = bqm.icol
        qdata = bqm.qdata

        if bqm.is_sorted:
            # O(log(|E|))
            if iu > iv:
                iu, iv = iv, iu

            left = np.searchsorted(irow, iu, side='left')

            # we could search a smaller window for the right bound by using the
            # number of variables in the BQM but I am not sure if we want to
            # assume that length of bqm exists
            off = np.searchsorted(irow[left:], iu, side='right')

            idx = left + np.searchsorted(icol[left:left+off], iv, side='left')

            if irow[idx] == iu and icol[idx] == iv:
                return qdata[idx]
        else:
            # O(|E|)
            # note this is ~100x faster than using a loop for 25000000 biases
            # at 225000000 biases the mask version is still faster than the
            # loop version was at 25000000
            # if we cythonized it we could avoid making the mask which would
            # save on memory
            mask = ((irow == iu) & (icol == iv)) ^ ((irow == iv) & (icol == iu))
            if np.any(mask):
                return np.sum(qdata[mask])

        raise KeyError

    def __iter__(self):
        # note: duplicates will appear if not de-duplicated
        bqm = self.bqm
        return iter(zip(bqm.irow, bqm.icol))

    def __len__(self):
        return len(self.bqm.irow)

    def __repr__(self):
        # note: this does not actually construct the object but under the
        # assumption that users won't actually want to construct this directly
        # it does make for much more readable printing
        # todo: do this without casting to a dict first
        return str(dict(self))


class LabelledQuadraticCooView(abc.Mapping):
    pass


class FrozenBinaryQuadraticModelCoo():
    __slots__ = ['ldata',
                 'irow', 'icol', 'qdata',
                 'vartype',
                 'is_sorted',
                 'linear', 'quadratic', 'offset',
                 ]

    def __init__(self, linear, quadratic, offset, vartype,
                 variable_order=None,
                 bias_dtype=np.float, index_dtype=np.int,
                 copy=False,
                 sort_and_reduce=False
                 ):

        # linear
        self.ldata = ldata = np.array(linear, dtype=bias_dtype, copy=copy)\

        # quadratic
        irow, icol, qdata = quadratic
        irow = np.array(irow, dtype=index_dtype, copy=copy)
        icol = np.array(icol, dtype=index_dtype, copy=copy)
        qdata = np.array(qdata, dtype=bias_dtype, copy=copy)

        # todo: check that they are all 1dim, same length etc

        if sort_and_reduce:
            self.is_sorted = is_sorted = True
            irow, icol, qdata = reduce_coo(irow, icol, qdata)
        else:
            # we're not sure if it's sorted or not, but we want bqm.is_sorted
            # to evaluate as False
            self.is_sorted = is_sorted = None

        self.irow = irow
        self.icol = icol
        self.qdata = qdata

        # offset
        self.offset = np.dtype(bias_dtype).type(offset)

        # views
        if variable_order is None:
            self.linear = LinearArrayView(ldata)

            self.quadratic = QuadraticCooView(self)
        else:
            # assume that it's a sequence of hashables for now
            variable_to_idx = OrderedDict((v, idx) for idx, v in enumerate(variable_order))

            # todo: check length

            raise NotImplementedError

            self.linear = LabelledLinearArrayView(ldata, variable_to_idx)
            self.quadratic = LabelledQuadraticCooView(irow, icol, qdata,
                                                      variable_to_idx)

    @classmethod
    def from_ising(cls, h, J, offset=0,
                   bias_dtype=np.float, index_dtype=np.int,
                   sort_and_reduce=False):
        # assume everything is integer-labeled for now
        num_variables = len(h)
        num_interactions = len(J)

        ldata = np.fromiter((h[v] for v in range(num_variables)),
                            count=num_variables, dtype=bias_dtype)

        irow = np.empty(num_interactions, dtype=index_dtype)
        icol = np.empty(num_interactions, dtype=index_dtype)
        qdata = np.empty(num_interactions, dtype=bias_dtype)

        # we could probably speed this up with cython
        for idx, ((u, v), bias) in enumerate(J.items()):
            irow[idx] = u
            icol[idx] = v
            qdata[idx] = bias

        return cls(ldata, (irow, icol, qdata), offset, SPIN,
                   bias_dtype=bias_dtype, index_dtype=index_dtype,
                   sort_and_reduce=sort_and_reduce, copy=False)
