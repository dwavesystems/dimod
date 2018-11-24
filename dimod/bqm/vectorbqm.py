from __future__ import absolute_import

try:
    from collections.abc import Sized
except ImportError:
    from collections import Sized

import numpy as np

from dimod.bqm.utils import reduce_coo
from dimod.bqm.vectors import vector
from dimod.decorators import vartype_argument
from dimod.vartypes import Vartype


class VectorBinaryQuadraticModel(Sized):
    __slots__ = 'vartype', 'offset', '_ldata', '_irow', '_icol', '_qdata', 'iadj'

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
            self._ldata = ldata = vector(linear, dtype=dtype)
        except TypeError:
            raise TypeError("linear must be array-like, {} is not allowed".format(type(linear)))

        num_variables = len(ldata)

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

        self._irow = vector(irow, dtype=index_dtype)
        self._icol = vector(icol, dtype=index_dtype)
        self._qdata = vector(qdata, dtype=dtype)

    @property
    def ldata(self):
        return np.asarray(self._ldata)

    @property
    def qdata(self):
        return np.asarray(self._qdata)

    @property
    def irow(self):
        return np.asarray(self._irow)

    @property
    def icol(self):
        return np.asarray(self._icol)

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
        other = self.copy()
        if self.vartype is Vartype.SPIN:
            # we're done!
            return other

        ldata = other.ldata
        qdata = other.qdata

        irow = other.irow
        icol = other.icol

        # offset
        other.offset += .5 * ldata.sum() + .25 * qdata.sum()

        # linear
        ldata /= 2
        for qi, bias in np.ndenumerate(qdata):
            ldata[irow[qi]] += .25 * bias
            ldata[icol[qi]] += .25 * bias

        # quadratic
        qdata /= 4

        other.vartype = Vartype.SPIN

        return other

    def to_binary(self):

        other = self.copy()

        if self.vartype is Vartype.BINARY:
            # we're done!
            return other

        ldata = other.ldata
        qdata = other.qdata

        irow = other.irow
        icol = other.icol

        # offset
        other.offset += -ldata.sum() + qdata.sum()  # this one makes a new value

        # linear
        ldata *= 2  # modifies in-place
        for qi, bias in np.ndenumerate(qdata):
            ldata[irow[qi]] += -2 * bias
            ldata[icol[qi]] += -2 * bias

        # quadratic
        qdata *= 4

        other.vartype = Vartype.BINARY

        return other

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
