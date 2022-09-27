# Copyright 2021 D-Wave Systems Inc.
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

import operator

cimport cython

from cython.operator cimport preincrement as inc, dereference as deref

from dimod.libcpp.vartypes cimport Vartype as cppVartype

from dimod.cyutilities cimport as_numpy_float
from dimod.cyutilities cimport ConstNumeric
from dimod.sampleset import as_samples
from dimod.variables import Variables
from dimod.vartypes import Vartype

# super minor performance benefit from constructing these here rather than
# in the __init__
_dtype = np.dtype(BIAS_DTYPE)
_index_dtype = np.dtype(INDEX_DTYPE)


cdef class cyQMBase_template:
    def __cinit__(self):
        # Dev note: we do *not* allocate self.base because it's an
        # abstract virtual class. So we have to rely on subclasses to do so

        self.variables = Variables()

    def __init__(self):
        self.dtype = _dtype
        self.index_dtype = _index_dtype

        # Ensure that this class is not instantiated by itself.
        # It is possible to get segfaults by bypassing this __init__, but that seems
        # like a small enough edge case to be OK.
        if self.base is NULL:
            raise TypeError(f"Can't instantiate abstract class {type(self).__name__}")

    @property
    def offset(self):
        return as_numpy_float(self.base.offset())

    @offset.setter
    def offset(self, bias_type offset):
        self.base.set_offset(offset)

    def clear(self):
        self.base.clear()
        self.variables._clear()

    def degree(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        return self.base.degree(vi)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _energies(self, ConstNumeric[:, ::1] samples, cyVariables labels):
        cdef Py_ssize_t num_samples = samples.shape[0]
        cdef Py_ssize_t num_variables = samples.shape[1]

        if num_variables != labels.size():
            # as_samples should never return inconsistent sizes, but we do this
            # check because the boundscheck is off and we otherwise might get
            # segfaults
            raise RuntimeError("as_samples returned an inconsistent samples/variables")

        # get the indices of the QM variables
        cdef Py_ssize_t[::1] qm_to_sample = np.empty(self.num_variables(), dtype=np.intp)
        cdef Py_ssize_t si
        for si in range(self.num_variables()):
            qm_to_sample[si] = labels.index(self.variables.at(si))

        cdef np.float64_t[::1] energies = np.empty(num_samples, dtype=np.float64)

        # alright, now let's calculate some energies!
        cdef Py_ssize_t ui, vi
        for si in range(num_samples):
            # offset
            energies[si] = self.base.offset()

            for ui in range(self.num_variables()):
                # linear
                energies[si] += self.base.linear(ui) * samples[si, qm_to_sample[ui]];

                it = self.base.cbegin_neighborhood(ui)
                end = self.base.cend_neighborhood(ui)
                while it != end and deref(it).v <= ui:
                    vi = deref(it).v

                    energies[si] += deref(it).bias * samples[si, qm_to_sample[ui]] * samples[si, qm_to_sample[vi]]

                    inc(it)

        return energies

    def energies(self, samples_like, dtype=None):
        # todo: deprecate dtype, it doesn't actually change what dtype
        # the calculation is done in, so it's pretty misleading

        samples, labels = as_samples(samples_like, labels_type=Variables)

        # we need contiguous and unsigned. as_samples actually enforces contiguous
        # but no harm in double checking for some future-proofness
        samples = np.ascontiguousarray(
                samples,
                dtype=f'i{samples.dtype.itemsize}' if np.issubdtype(samples.dtype, np.unsignedinteger) else None,
                )

        try:
            return np.asarray(self._energies(samples, labels), dtype=dtype)
        except TypeError as err:
            if np.issubdtype(samples.dtype, np.floating) or np.issubdtype(samples.dtype, np.signedinteger):
                raise err
            raise ValueError(f"unsupported sample dtype: {samples.dtype.name}")

    def get_linear(self, v):
        return as_numpy_float(self.base.linear(self.variables.index(v)))

    def get_quadratic(self, u, v, default=None):
        cdef Py_ssize_t ui = self.variables.index(u)
        cdef Py_ssize_t vi = self.variables.index(v)

        if ui == vi:
            if self.base.vartype(ui) == cppVartype.SPIN:
                raise ValueError(f"SPIN variables (e.g. {u!r}) "
                                 "cannot have interactions with themselves"
                                 )
            if self.base.vartype(ui) == cppVartype.BINARY:
                raise ValueError(f"BINARY variables (e.g. {v!r}) "
                                 "cannot have interactions with themselves"
                                 )
        cdef bias_type bias
        try:
            bias = self.base.quadratic_at(ui, vi)
        except IndexError:
            # out of range error is automatically converted to IndexError
            if default is None:
                raise ValueError(f"{u!r} and {v!r} have no interaction") from None
            bias = default
        return as_numpy_float(bias)

    cpdef bint is_linear(self):
        return self.base.is_linear()

    def iter_neighborhood(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)

        it = self.base.cbegin_neighborhood(vi)
        while it != self.base.cend_neighborhood(vi):
            yield self.variables.at(deref(it).v), as_numpy_float(deref(it).bias)
            inc(it)

    def iter_quadratic(self):
        it = self.base.cbegin_quadratic()
        while it != self.base.cend_quadratic():
            u = self.variables.at(deref(it).u)
            v = self.variables.at(deref(it).v)
            yield u, v, as_numpy_float(deref(it).bias)
            inc(it)

    def lower_bound(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        cdef bias_type lb = self.base.lower_bound(vi)
        return as_numpy_float(lb)

    def nbytes(self, bint capacity = False):
        return self.base.nbytes(capacity)

    cpdef Py_ssize_t num_interactions(self):
        return self.base.num_interactions()

    cpdef Py_ssize_t num_variables(self):
        return self.base.num_variables()

    def reduce_linear(self, function, initializer=None):
        if self.num_variables() == 0 and initializer is None:
            # feels like this should be a ValueError but python raises
            # TypeError so...
            raise TypeError("reduce_linear() on an empty model")

        cdef Py_ssize_t start, vi
        cdef bias_type value, tmp

        if initializer is None:
            start = 1
            value = self.base.linear(0)
        else:
            start = 0
            value = initializer

        # speed up a few common cases
        if function is operator.add:
            for vi in range(start, self.num_variables()):
                value += self.base.linear(vi)
        elif function is max:
            for vi in range(start, self.num_variables()):
                tmp = self.base.linear(vi)
                if tmp > value:
                    value = tmp
        elif function is min:
            for vi in range(start, self.num_variables()):
                tmp = self.base.linear(vi)
                if tmp < value:
                    value = tmp
        else:
            for vi in range(start, self.num_variables()):
                value = function(value, self.base.linear(vi))

        return as_numpy_float(value)

    def reduce_neighborhood(self, u, function, initializer=None):
        cdef Py_ssize_t ui = self.variables.index(u)

        if self.base.degree(ui) == 0 and initializer is None:
            # feels like this should be a ValueError but python raises
            # TypeError so...
            raise TypeError("reduce_neighborhood() on an empty neighbhorhood")

        cdef bias_type value, tmp

        # span = self.base.neighborhood(ui)
        it = self.base.cbegin_neighborhood(ui)

        if initializer is None:
            value = deref(it).bias
            inc(it)
        else:
            value = initializer

        # speed up a few common cases
        if function is operator.add:
            while it != self.base.cend_neighborhood(ui):
                value += deref(it).bias
                inc(it)
        elif function is max:
            while it != self.base.cend_neighborhood(ui):
                tmp = deref(it).bias
                if tmp > value:
                    value = tmp
                inc(it)
        elif function is min:
            while it != self.base.cend_neighborhood(ui):
                tmp = deref(it).bias
                if tmp < value:
                    value = tmp
                inc(it)
        else:
            while it != self.base.cend_neighborhood(ui):
                value = function(value, deref(it).second)
                inc(it)

        return as_numpy_float(value)

    def reduce_quadratic(self, function, initializer=None):

        if self.base.is_linear() and initializer is None:
            # feels like this should be a ValueError but python raises
            # TypeError so...
            raise TypeError("reduce_quadratic() on a linear model")

        cdef bias_type value, tmp

        start = self.base.cbegin_quadratic()

        if initializer is None:
            value = deref(start).bias
            inc(start)
        else:
            value = initializer

        # handle a few common cases
        if function is operator.add:
            while start != self.base.cend_quadratic():
                value += deref(start).bias
                inc(start)
        elif function is max:
            while start != self.base.cend_quadratic():
                tmp = deref(start).bias
                if tmp > value:
                    value = tmp
                inc(start)
        elif function is min:
            while start != self.base.cend_quadratic():
                tmp = deref(start).bias
                if tmp < value:
                    value = tmp
                inc(start)
        else:
            while start != self.base.cend_quadratic():
                value = function(value, deref(start).bias)
                inc(start)

        return as_numpy_float(value)

    def relabel_variables(self, mapping):
        self.variables._relabel(mapping)

    def relabel_variables_as_integers(self):
        return self.variables._relabel_as_integers()

    def remove_interaction(self, u, v):
        cdef Py_ssize_t ui = self.variables.index(u)
        cdef Py_ssize_t vi = self.variables.index(v)
        
        if not self.base.remove_interaction(ui, vi):
            raise ValueError(f"{u!r} and {v!r} have no interaction")

    def remove_variable(self, v=None):
        if v is None:
            try:
                v = self.variables[-1]
            except IndexError:
                raise ValueError("cannot pop from an empty model")

        self.base.remove_variable(self.variables.index(v))
        self.variables._remove(v)

        return v

    def scale(self, bias_type scalar):
        self.base.scale(scalar)

    def upper_bound(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        return as_numpy_float(self.base.upper_bound(vi))

    def vartype(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        cdef cppVartype cppvartype = self.base.vartype(vi)

        if cppvartype == cppVartype.BINARY:
            return Vartype.BINARY
        elif cppvartype == cppVartype.SPIN:
            return Vartype.SPIN
        elif cppvartype == cppVartype.INTEGER:
            return Vartype.INTEGER
        elif cppvartype == cppVartype.REAL:
            return Vartype.REAL
        else:
            raise RuntimeError("unexpected vartype")
