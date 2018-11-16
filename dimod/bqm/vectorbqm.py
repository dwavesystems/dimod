from __future__ import absolute_import

try:
    from collections.abc import Sized
except ImportError:
    from collections import Sized

import numpy as np

from dimod.bqm.utils import reduce_coo
from dimod.decorators import vartype_argument
from dimod.vartypes import Vartype


class VectorBinaryQuadraticModel(Sized):
    __slots__ = 'vartype', 'offset', 'ldata', 'irow', 'icol', 'qdata', 'iadj'

    @vartype_argument('vartype')
    def __init__(self, linear, quadratic, offset, vartype,
                 dtype=np.float, index_dtype=np.int64):
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

        irow, icol, qdata = reduce_coo(row, col, data, dtype=dtype, index_dtype=index_dtype,
                                       copy=True)

        if np.max(icol, initial=-1) >= num_variables:
            msg = "mismatched linear and quadratic dimensions ({}, {})".format(num_variables, icol[-1])
            raise ValueError(msg)

        self.iadj = iadj = {v: {} for v in range(num_variables)}
        for idx, (ir, ic) in enumerate(zip(irow, icol)):
            iadj[ir][ic] = iadj[ic][ir] = idx

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

    def energy(self, sample, *args, **kwargs):
        en, = self.energies([sample], *args, **kwargs)  # should only be one
        return en

    def energies(self, samples, order=None, _use_cpp_ext=True):

        if hasattr(samples, "dtype"):
            samples = np.asarray(samples, dtype=samples.dtype)  # handle subclasses
        else:
            samples = np.asarray(samples, dtype=np.int8)

        ldata = self.ldata
        row = self.irow
        col = self.icol
        qdata = self.qdata
        offset = self.offset

        if order is not None:
            order = np.asarray(order)

            ldata = ldata[order]
            row = order[row]
            col = order[col]

        if _use_cpp_ext:
            try:
                from dimod.bqm._utils import fast_energy
                return fast_energy(offset, ldata, row, col, qdata, samples)
            except ImportError:
                # no c++ extension
                pass
            # except TypeError:
            #     # dtype is the wrong type
            #     pass

        try:
            num_samples, num_variables = samples.shape
        except ValueError:
            raise ValueError("samples should be a square array where each row is a sample")

        energy = np.full(num_samples, offset)  # offset

        energy += samples.dot(ldata)  # linear

        energy += (samples[:, row]*samples[:, col]).dot(qdata)  # quadratic

        return energy


VectorBQM = VectorBinaryQuadraticModel
