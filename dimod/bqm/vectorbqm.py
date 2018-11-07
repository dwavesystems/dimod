try:
    from collections.abc import Sized
except ImportError:
    from collections import Sized

import numpy as np

from dimod.decorators import vartype_argument
from dimod.views import NeighborView


def reduce_coo(row, col, data, dtype=None, index_dtype=None, copy=True):
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

    row = np.array(row, dtype=index_dtype, copy=copy)
    col = np.array(col, dtype=index_dtype, copy=copy)
    data = np.array(data, dtype=dtype, copy=copy)

    if row.ndim != 1 or col.ndim != 1 or data.ndim != 1 or len(row) != len(col) or len(col) != len(data):
        raise ValueError("row, col and data should all be vectors of equal length")

    if len(row) == 0:
        # empty arrays are already sorted
        return row, col, data

    # row index should be less than col index, this handles upper-triangular vs lower-triangular
    swaps = row > col
    if swaps.any():
        # in-place
        row[swaps], col[swaps] = col[swaps], row[swaps]

    # sort lexigraphically
    order = np.lexsort((row, col))
    if not (order == range(len(order))).all():
        # copy
        row = row[order]
        col = col[order]
        data = data[order]

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


class VectorBinaryQuadraticModel(Sized):
    __slots__ = 'vartype', 'offset', 'ldata', 'irow', 'icol', 'qdata', 'iadj'

    @vartype_argument('vartype')
    def __init__(self, linear, quadratic, offset, vartype, dtype=np.float, index_dtype=np.int64):
        """
        Developer note, linear, quadratic might be modified in-place

        """
        self.vartype = vartype  # checked by decorator

        self.offset = np.dtype(dtype).type(offset)

        # linear
        # cast to a numpy array or make a copy, also setting the dtype.
        try:
            self.ldata = ldata = np.array(linear, dtype=dtype)
        except TypeError:
            raise TypeError("linear must be array-like, {} is not allowed".format(type(linear)))
        if ldata.ndim != 1:
            raise ValueError("linear must be a vector")

        num_variables, = ldata.shape

        # quadratic
        if isinstance(quadratic, tuple) and len(quadratic) == 3:
            row, col, data = quadratic
        else:
            # assume dense matrix
            quadratic = np.atleast_2d(np.asarray(quadratic, dtype=dtype))

            if quadratic.ndim > 2:
                raise ValueError("quadratic should be a square matrix")

            if not quadratic.size:
                # if one of the dimensions is 0
                quadratic = np.empty((0, 0), dtype=dtype)

            if quadratic.shape[0] != quadratic.shape[1]:
                raise ValueError("quadratic should be a square matrix, given matrix is {} x {}".format(
                    quadratic.shape[0], quadratic.shape[1]))

            row, col = quadratic.nonzero()
            data = quadratic[row, col]

        irow, icol, qdata = reduce_coo(row, col, data, dtype=dtype, index_dtype=index_dtype, copy=True)

        if np.max(icol, initial=-1) >= num_variables:
            msg = "mismatched linear and quadratic dimensions ({}, {})".format(num_variables, icol[-1])
            raise ValueError(msg)

        self.iadj = iadj = {v: NeighborView({}, qdata) for v in range(num_variables)}
        for idx, (ir, ic) in enumerate(zip(irow, icol)):
            iadj[ir]._index[ic] = iadj[ic]._index[ir] = idx

        self.irow = irow
        self.icol = icol
        self.qdata = qdata

    @property
    def dtype(self):
        return np.result_type(self.ldata, self.qdata, self.offset)

    @property
    def index_dtype(self):
        return np.result_type(self.irow, self.icol)

    def __len__(self):
        return len(self.ldata)

    def __repr__(self):
        return "{}({}, ({}, {}, {}), {}, '{}', dtype='{}', index_dtype='{}')".format(
            self.__class__.__name__,
            self.ldata,
            self.irow, self.icol, self.qdata,
            self.offset,
            self.vartype.name,
            np.dtype(self.dtype).name, np.dtype(self.index_dtype).name)

    def __eq__(self, other):
        return (self.vartype == other.vartype and self.offset == other.offset
                and (self.ldata == other.ldata).all() and self.iadj == other.iadj)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        # copies are made of all of the arrays
        return self.__class__(self.ldata, (self.irow, self.icol, self.qdata), self.offset, self.vartype,
                              dtype=self.dtype, index_dtype=self.index_dtype)

    def to_spin(self):
        if self.vartype is Vartype.SPIN:
            return self.copy()

        ldata = self.ldata
        qdata = self.qdata
        irow = self.irow
        icol = self.icol

        # offset
        offset = .5 * ldata.sum() + .25 * qdata.sum() + self.offset

        # linear
        linear = .5 * ldata
        for qi, bias in np.ndenumerate(qdata):
            linear[irow[qi]] += .25 * bias
            linear[icol[qi]] += .25 * bias

        # quadratic
        quadratic = (irow, icol, .25 * qdata)

        return self.__class__(linear, quadratic, offset, Vartype.SPIN,
                              dtype=self.dtype, index_dtype=self.index_dtype)

    def to_binary(self):
        if self.vartype is Vartype.BINARY:
            return self.copy()

        ldata = self.ldata
        qdata = self.qdata
        irow = self.irow
        icol = self.icol

        # offset
        offset = -ldata.sum() + qdata.sum() + self.offset

        # linear
        linear = 2 * ldata  # makes a new vector of the same dtype
        for qi, bias in np.ndenumerate(qdata):
            linear[irow[qi]] += -2 * bias
            linear[icol[qi]] += -2 * bias

        # quadratic
        quadratic = (irow, icol, 4 * self.qdata)

        return self.__class__(linear, quadratic, offset, Vartype.BINARY,
                              dtype=self.dtype, index_dtype=self.index_dtype)

    def energies(self, samples, _use_cpp_ext=True):

        # use the faster c++ extension if it's available
        if _use_cpp_ext:
            try:
                from dimod.bqm._helpers import fast_energy
                return fast_energy(self.offset, self.ldata, self.irow, self.icol, self.qdata, samples)
            except ImportError:
                # no c++ extension
                pass
            except TypeError:
                # dtype is the wrong type
                pass

        if not hasattr(samples, "dtype"):
            samples = np.asarray(samples, dtype=samples.dtype)  # handle subclasses
        else:
            samples = np.asarray(samples, dtype=np.int8)

        try:
            num_samples, num_variables = samples.shape
        except ValueError:
            raise ValueError("samples should be a square array where each row is a sample")

        energies = np.full(num_samples, self.offset)  # offset

        energies += samples.dot(self.ldata)  # linear

        energies += (samples[:, self.irow]*samples[:, self.icol]).dot(self.qdata)  # quadratic

        return energies


VectorBQM = VectorBinaryQuadraticModel
