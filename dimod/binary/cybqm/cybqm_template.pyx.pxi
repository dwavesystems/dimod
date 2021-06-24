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

# from collections.abc import Mapping

import copy
import operator

from collections.abc import Sized

# cimport cpython
cimport cython

import numpy as np

from cython.operator cimport postincrement as inc, dereference as deref
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set

from dimod.binary.cybqm cimport cyBQM
from dimod.cyutilities cimport as_numpy_float, ConstInteger
from dimod.cyutilities import coo_sort
from dimod.sampleset import as_samples
from dimod.utilities import asintegerarrays, asnumericarrays
from dimod.variables import Variables
from dimod.vartypes import Vartype, as_vartype


# Design principles/conventions:
# - Self-loops must raise exceptions
# - All def/cpdef functions should work with variable labels
# - Access by index should be exposed via cdef
# - All def/cpdef functions should be "safe" - no segfaults


cdef class cyBQM_template(cyBQMBase):
    def __init__(self, vartype):
        self.dtype = np.dtype(BIAS_DTYPE)
        self.index_dtype = np.dtype(INDEX_DTYPE)
        self.change_vartype(vartype)
        self.variables = Variables()

    def __deepcopy__(self, memo):
        cdef cyBQM_template new = type(self)(self.vartype)
        new.cppbqm = self.cppbqm
        new.variables = copy.deepcopy(self.variables, memo)
        memo[id(self)] = new
        return new

    def __reduce__(self):
        ldata, qdata, off, labels = self.to_numpy_vectors(return_labels=True)
        return (
            type(self).from_numpy_vectors,
            (ldata, qdata, off, self.vartype, labels))

    @property
    def offset(self):
        """Constant energy offset associated with the model."""
        return as_numpy_float(self.cppbqm.offset())

    @offset.setter
    def offset(self, bias_type offset):
        self._set_offset(offset)

    @property
    def vartype(self):
        """The model's variable type.

        One of :class:`.Vartype.SPIN` or :class:`.Vartype.BINARY`.
        """
        if self.cppbqm.vartype() == cppVartype.BINARY:
            return Vartype.BINARY
        elif self.cppbqm.vartype() == cppVartype.SPIN:
            return Vartype.SPIN
        else:
            raise RuntimeError("unknown vartype")

    cdef void _add_linear(self, Py_ssize_t vi, bias_type bias):
        # unsafe version of .add_linear
        cdef bias_type *b = &(self.cppbqm.linear(vi))
        b[0] += bias

    cdef void _add_offset(self, bias_type bias):
        cdef bias_type *b = &(self.cppbqm.offset())
        b[0] += bias

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ilinear(self):
        """Return a numpy struct array with the linear biases and the
        indices of the neighborhoods. This method is used for serialization.
        """
        cdef Py_ssize_t num_variables = self.num_variables()

        dtype = np.dtype([('ni', self.index_dtype), ('b', self.dtype)],
                          align=False)
        ldata = np.empty(num_variables, dtype=dtype)

        cdef index_type[:] neighbors_view = ldata['ni']
        cdef bias_type[:] bias_view = ldata['b']

        cdef Py_ssize_t vi
        if num_variables:
            neighbors_view[0] = 0
            bias_view[0] = self.cppbqm.linear(0)

            for vi in range(1, num_variables):
                neighbors_view[vi] = (neighbors_view[vi - 1]
                                      + self.cppbqm.num_interactions(vi - 1))
                bias_view[vi] = self.cppbqm.linear(vi)

        return ldata

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ineighborhood(self, Py_ssize_t ui):
        """
        """
        if not 0 <= ui < self.num_variables():
            raise ValueError(f"out of range variable, {ui!r}")

        cdef Py_ssize_t degree = self.cppbqm.num_interactions(ui)

        dtype = np.dtype([('ui', self.index_dtype), ('b', self.dtype)],
                         align=False)
        neighbors = np.empty(degree, dtype=dtype)
        
        cdef index_type[:] index_view = neighbors['ui']
        cdef bias_type[:] bias_view = neighbors['b']

        span = self.cppbqm.neighborhood(ui)
        cdef Py_ssize_t i = 0
        while span.first != span.second:
            index_view[i] = deref(span.first).first
            bias_view[i] = deref(span.first).second

            i += 1
            inc(span.first)

        return neighbors

    cdef Py_ssize_t _index(self, v, bint permissive=False) except -1:
        """Return the index of variable `v`.

        If `permissive` is True, the variable will be added to the binary
        quadratic model and the size increased accordingly.
        """
        # return the index of variable v
        cdef Py_ssize_t vi = self.variables.index(v, permissive=permissive)

        # we might have added a variable
        if permissive and vi == self.cppbqm.num_variables():
            self.cppbqm.resize(vi + 1)

        return vi

    cdef void _set_linear(self, Py_ssize_t vi, bias_type bias):
        # unsafe version of .set_linear
        cdef bias_type *b = &(self.cppbqm.linear(vi))
        b[0] = bias

    cdef void _set_offset(self, bias_type bias):
        cdef bias_type *b = &(self.cppbqm.offset())
        b[0] = bias

    def add_linear(self, v, bias_type bias):
        cdef Py_ssize_t vi = self._index(v, permissive=True)
        self._add_linear(vi, bias)

    def add_linear_equality_constraint(self, object terms,
                                       bias_type lagrange_multiplier,
                                       bias_type constant):
        cdef vector[index_type] variables
        cdef vector[bias_type] biases

        # can allocate them if we already know the size
        if isinstance(terms, Sized):
            biases.reserve(len(terms))
            variables.reserve(len(terms))

        cdef bias_type bias
        for v, bias in terms:
            variables.push_back(self._index(v, permissive=True))
            biases.push_back(bias)

        cdef Py_ssize_t i, j
        cdef Py_ssize_t num_terms = biases.size()

        # offset part
        self._add_offset(lagrange_multiplier * constant * constant)

        # linear part
        if self.cppbqm.vartype() == cppVartype.BINARY:
            for i in range(num_terms):
                self._add_linear(
                    variables[i],
                    lagrange_multiplier * biases[i] * (2 * constant + biases[i]))
        elif self.cppbqm.vartype() == cppVartype.SPIN:
            for i in range(num_terms):
                self._add_linear(
                    variables[i],
                    lagrange_multiplier * biases[i] * 2 * constant)
                self._add_offset(lagrange_multiplier * biases[i] * biases[i])
        else:
            raise RuntimeError("unexpected vartype")

        # quadratic part
        for i in range(num_terms):
            for j in range(i + 1, num_terms):
                self.cppbqm.add_quadratic(
                    variables[i], variables[j],
                    2 * lagrange_multiplier * biases[i] * biases[j]
                    )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Py_ssize_t add_linear_from_array(self, ConstNumeric[:] linear) except -1:
        cdef Py_ssize_t vi
        cdef Py_ssize_t length = linear.shape[0]

        if self.variables._is_range():
            # we don't need to check the labels so can skip that
            if length > self.num_variables():
                self.resize(length)

            for vi in range(length):
                self._add_linear(vi, linear[vi])
        else:
            # need to add them "one by one"
            for vi in range(length):
                self.add_linear(vi, linear[vi])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_offset_from_array(self, ConstNumeric[::1] offset):
        if offset.shape[0] != 1:
            raise ValueError("array should be of length 1")
        self._add_offset(offset[0])

    def add_quadratic(self, u, v, bias_type bias):
        if u == v:
            raise ValueError(f"{u!r} cannot have an interaction with itself")

        cdef Py_ssize_t ui = self._index(u, permissive=True)
        cdef Py_ssize_t vi = self._index(v, permissive=True)
        self.cppbqm.add_quadratic(ui, vi, bias)

    def add_quadratic_from_arrays(self,
                                  ConstInteger[::1] irow,
                                  ConstInteger[::1] icol,
                                  ConstNumeric[::1] qdata):

        if not irow.shape[0] == icol.shape[0] == qdata.shape[0]:
            raise ValueError("quadratic vectors should be equal length")
        cdef Py_ssize_t length = irow.shape[0]

        if length:
            if self.variables._is_range():
                self.cppbqm.add_quadratic(&irow[0], &icol[0], &qdata[0], length)
                self.variables._stop = self.cppbqm.num_variables()
            else:
                raise NotImplementedError


    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef Py_ssize_t add_quadratic_from_dense(self, ConstNumeric[:, ::1] quadratic) except -1:
        if quadratic.shape[0] != quadratic.shape[1]:
            raise ValueError("quadratic must be a square matrix")

        cdef Py_ssize_t num_variables = quadratic.shape[0]
    
        cdef Py_ssize_t ui
        for ui in range(num_variables):
            if quadratic[ui, ui]:
                raise ValueError(f"{ui!r} cannot have an interaction with itself")

        if self.variables._is_range():
            if num_variables > self.num_variables():
                self.resize(num_variables)
            self.cppbqm.add_quadratic(&quadratic[0, 0], num_variables)
        else:
            raise NotImplementedError

    def add_variable(self, v=None, bias_type bias=0):
        v = self.variables._append(v, permissive=True)

        if self.variables.size() > self.cppbqm.num_variables():
            self.cppbqm.resize(self.variables.size())

        self.add_linear(v, bias)

        return v

    cpdef Py_ssize_t change_vartype(self, object vartype) except -1:
        vartype = as_vartype(vartype)
        if vartype == Vartype.BINARY:
            self.cppbqm.change_vartype(cppVartype.BINARY)
        elif vartype == Vartype.SPIN:
            self.cppbqm.change_vartype(cppVartype.SPIN)
        else:
            raise RuntimeError("unknown vartype", vartype)

    def degree(self, v):
        cdef Py_ssize_t vi = self._index(v)
        return self.cppbqm.num_interactions(vi)

    # need signal to let cython handle the return type
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef cython.floating[::1] _energies(self,
                                        object samples_like,
                                        cython.floating signal=0):
        # internal implementation with energy types
        samples, labels = as_samples(samples_like, dtype=np.int8)

        cdef np.int8_t[:, :] samples_view = samples

        cdef Py_ssize_t num_samples = samples_view.shape[0]
        cdef Py_ssize_t num_variables = samples_view.shape[1]

        if num_variables != self.num_variables():
            raise ValueError("inconsistent number of variables")
        if num_variables != len(labels):
            # an internal error to as_samples. We do this check because
            # the boundscheck is off
            raise RuntimeError(
                "as_samples returned an inconsistent samples/variables")

        cdef Py_ssize_t[::1] bqm_to_sample = np.empty(num_variables, dtype=np.intp)
        cdef Py_ssize_t si
        for si in range(num_variables):
            v = labels[si]  # python label
            bqm_to_sample[self.variables.index(v)] = si

        cdef cython.floating[::1] energies
        if cython.floating is float:
            energies = np.empty(num_samples, dtype=np.float32)
        else:
            energies = np.empty(num_samples, dtype=np.float64)

        # alright, now let's calculate some energies!
        cdef np.int8_t uspin, vspin
        cdef Py_ssize_t ui, vi
        for si in range(num_samples):
            # offset
            energies[si] = self.cppbqm.offset()

            for ui in range(num_variables):
                uspin = samples_view[si, bqm_to_sample[ui]]

                # linear
                energies[si] += self.cppbqm.linear(ui) * uspin;

                span = self.cppbqm.neighborhood(ui)
                while span.first != span.second and deref(span.first).first < ui:
                    vi = deref(span.first).first

                    vspin = samples_view[si, bqm_to_sample[vi]]

                    energies[si] += uspin * vspin * deref(span.first).second

                    inc(span.first)

        return energies

    def energies(self, samples_like, dtype=None):
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        if dtype == np.float64:
            return np.asarray(self._energies[np.float64_t](samples_like))
        elif dtype == np.float32:
            return np.asarray(self._energies[np.float32_t](samples_like))
        else:
            raise ValueError("dtype must be None or a floating type.")

    @classmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _from_numpy_vectors(cls,
                            ConstNumeric[::1] linear,
                            ConstInteger[::1] irow,
                            ConstInteger[::1] icol,
                            ConstNumeric[::1] qdata,
                            bias_type offset,
                            object vartype):
        """Equivalent of from_numpy_vectors with fused types."""

        cdef cyBQM_template bqm = cls(vartype)

        # add the quadratic
        if not irow.shape[0] == icol.shape[0] == qdata.shape[0]:
            raise ValueError("quadratic vectors should be equal length")
        cdef Py_ssize_t length = irow.shape[0]

        if length:
            bqm.cppbqm.add_quadratic(&irow[0], &icol[0], &qdata[0], length)

        bqm.variables._stop = bqm.cppbqm.num_variables()

        # add the linear
        if bqm.num_variables() < linear.shape[0]:
            bqm.resize(linear.shape[0])
        cdef Py_ssize_t vi
        for vi in range(linear.shape[0]):
            bqm._add_linear(vi, linear[vi])

        # add the offset
        bqm._add_offset(offset)

        return bqm

    @classmethod
    def from_numpy_vectors(cls, linear, quadratic, offset, vartype,
                           variable_order=None):
        """Create a binary quadratic model from vectors.

        Args:
            linear (array_like):
                A 1D array-like iterable of linear biases.

            quadratic (tuple[array_like, array_like, array_like]):
                A 3-tuple of 1D array_like vectors of the form (row, col, bias).

            offset (numeric, optional):
                Constant offset for the binary quadratic model.

            vartype (:class:`.Vartype`/str/set):
                Variable type for the binary quadratic model. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            variable_order (iterable, optional):
                If provided, labels the variables; otherwise, indices are used.

        Returns:
            A binary quadratic model

        """
        try:
            irow, icol, qdata = quadratic
        except ValueError:
            raise ValueError("quadratic should be a 3-tuple")

        # We need:
        # * numpy ndarrays
        # * contiguous memory
        # * ldata.dtype == qdata.dtype and irow.dtype == icol.dtype
        # * 32 or 64 bit dtypes
        icol, irow = asintegerarrays(
            icol, irow, min_itemsize=4, requirements='C')
        ldata, qdata = asnumericarrays(
            linear, qdata, min_itemsize=4, requirements='C')

        bqm = cls._from_numpy_vectors(ldata, irow, icol, qdata, offset, vartype)

        if variable_order is not None:
            if len(variable_order) != bqm.num_variables():
                raise ValueError(
                    "variable_order must be the same length as the BQM")

            bqm.relabel_variables(dict(enumerate(variable_order)))

        return bqm

    def get_linear(self, u):
        cdef Py_ssize_t ui = self.variables.index(u)
        cdef bias_type bias = self.cppbqm.linear(ui)
        return as_numpy_float(bias)

    def get_quadratic(self, u, v, default=None):
        cdef Py_ssize_t ui = self.variables.index(u)
        cdef Py_ssize_t vi = self.variables.index(v)

        if ui == vi:
            raise ValueError(f"{u!r} cannot have an interaction with itself")

        # todo: catch error
        cdef bias_type bias
        try:
            bias = self.cppbqm.quadratic_at(ui, vi)
        except IndexError:
            if default is None:
                # out of range error is automatically converted to IndexError
                raise ValueError(f"{u!r} and {v!r} have no interaction")
            bias = default
        return as_numpy_float(bias)

    cpdef bint is_linear(self):
        return self.cppbqm.is_linear()

    def iter_neighborhood(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)

        cdef Py_ssize_t ui
        cdef bias_type bias

        span = self.cppbqm.neighborhood(vi)
        while span.first != span.second:
            ui = deref(span.first).first
            bias = deref(span.first).second

            yield self.variables.at(ui), as_numpy_float(bias)

            inc(span.first)

    def iter_quadratic(self):
        it = self.cppbqm.cbegin_quadratic()
        while it != self.cppbqm.cend_quadratic():
            u = self.variables.at(deref(it).u)
            v = self.variables.at(deref(it).v)
            yield u, v, as_numpy_float(deref(it).bias)
            inc(it)


    cpdef Py_ssize_t num_interactions(self):
        return self.cppbqm.num_interactions()

    cpdef Py_ssize_t num_variables(self):
        return self.cppbqm.num_variables()

    def reduce_linear(self, function, initializer=None):
        if self.num_variables() == 0 and initializer is None:
            # feels like this should be a ValueError but python raises
            # TypeError so...
            raise TypeError("reduce_linear() on an empty BQM")

        cdef Py_ssize_t start, vi
        cdef bias_type value, tmp

        if initializer is None:
            start = 1
            value = self.cppbqm.linear(0)
        else:
            start = 0
            value = initializer

        # speed up a few common cases
        if function is operator.add:
            for vi in range(start, self.num_variables()):
                value += self.cppbqm.linear(vi)
        elif function is max:
            for vi in range(start, self.num_variables()):
                tmp = self.cppbqm.linear(vi)
                if tmp > value:
                    value = tmp
        elif function is min:
            for vi in range(start, self.num_variables()):
                tmp = self.cppbqm.linear(vi)
                if tmp < value:
                    value = tmp
        else:
            for vi in range(start, self.num_variables()):
                value = function(value, self.cppbqm.linear(vi))

        return as_numpy_float(value)

    def reduce_neighborhood(self, u, function, initializer=None):
        cdef Py_ssize_t ui = self.variables.index(u)

        if self.cppbqm.num_interactions(ui) == 0 and initializer is None:
            # feels like this should be a ValueError but python raises
            # TypeError so...
            raise TypeError("reduce_neighborhood() on an empty neighbhorhood")

        cdef bias_type value, tmp

        span = self.cppbqm.neighborhood(ui)

        if initializer is None:
            value = deref(span.first).second
            inc(span.first)
        else:
            value = initializer

        # speed up a few common cases
        if function is operator.add:
            while span.first != span.second:
                value += deref(span.first).second
                inc(span.first)
        elif function is max:
            while span.first != span.second:
                tmp = deref(span.first).second
                if tmp > value:
                    value = tmp
                inc(span.first)
        elif function is min:
            while span.first != span.second:
                tmp = deref(span.first).second
                if tmp < value:
                    value = tmp
                inc(span.first)
        else:
            while span.first != span.second:
                value = function(value, deref(span.first).second)
                inc(span.first)

        return as_numpy_float(value)

    def reduce_quadratic(self, function, initializer=None):

        if self.cppbqm.is_linear() and initializer is None:
            # feels like this should be a ValueError but python raises
            # TypeError so...
            raise TypeError("reduce_quadratic() on a linear model")

        cdef bias_type value, tmp

        start = self.cppbqm.cbegin_quadratic()

        if initializer is None:
            value = deref(start).bias
            inc(start)
        else:
            value = initializer

        # handle a few common cases
        if function is operator.add:
            while start != self.cppbqm.cend_quadratic():
                value += deref(start).bias
                inc(start)
        elif function is max:
            while start != self.cppbqm.cend_quadratic():
                tmp = deref(start).bias
                if tmp > value:
                    value = tmp
                inc(start)
        elif function is min:
            while start != self.cppbqm.cend_quadratic():
                tmp = deref(start).bias
                if tmp < value:
                    value = tmp
                inc(start)
        else:
            while start != self.cppbqm.cend_quadratic():
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
        
        if not self.cppbqm.remove_interaction(ui, vi):
            raise ValueError(f"{u!r} and {v!r} have no interaction")

    def remove_variable(self, v=None):
        """Remove a variable and its associated interactions.

        Args:
            v: The variable to be removed from the binary quadratic model.

        Returns:
            The label of the removed variable.

        Raises:
            ValueError: If the variable does not exist.

        """
        if v is None:
            try:
                v = self.variables[-1]
            except KeyError:
                raise ValueError("cannot pop from an empty model")

        cdef Py_ssize_t vi = self.variables.index(v)
        cdef Py_ssize_t lasti = self.num_variables() - 1

        if vi == lasti:
            # it's the last one!
            self.resize(vi)
            return v

        last = self.variables.at(lasti)

        # in this case we're removing a variable in the middle of the
        # underlying adjacency. We do this by "swapping" the last variable
        # and v, then popping v from the end

        # remove all of the interactions associated with v
        cdef Py_ssize_t ui
        span = self.cppbqm.neighborhood(vi)
        while span.first != span.second:
            ui = deref(span.first).first
            self.cppbqm.remove_interaction(ui, vi)
            span = self.cppbqm.neighborhood(vi)

        # copy all of the interactions from last to v
        span = self.cppbqm.neighborhood(lasti)
        while span.first != span.second:
            ui = deref(span.first).first
            self.cppbqm.set_quadratic(ui, vi, deref(span.first).second)
            inc(span.first)

        # copy the linear bias from last to v
        self._set_linear(vi, self.cppbqm.linear(lasti))

        # remove last from the cppbqm
        self.cppbqm.resize(lasti)

        # now swap the variable labels
        self.variables._relabel({v: last, last: v})

        tmp = self.variables._pop()

        assert tmp == v, f"{tmp} == {v}"

        return v

    cpdef Py_ssize_t resize(self, Py_ssize_t n) except -1:
        if n < 0:
            raise ValueError("n must be non-negative")

        self.cppbqm.resize(n)

        while self.variables.size() < n:
            self.variables._append()
        while self.variables.size() > n:
            self.variables._pop()

    def set_linear(self, v, bias_type bias):
        cdef Py_ssize_t vi = self._index(v, permissive=True)
        self._set_linear(vi, bias)

    def set_quadratic(self, u, v, bias_type bias):
        if u == v:
            raise ValueError(f"{u!r} cannot have an interaction with itself")

        cdef Py_ssize_t ui = self._index(u, permissive=True)
        cdef Py_ssize_t vi = self._index(v, permissive=True)
        self.cppbqm.set_quadratic(ui, vi, bias)

    def to_numpy_vectors(self, variable_order=None, *,
                         sort_indices=False, sort_labels=True,
                         return_labels=False):

        cdef Py_ssize_t num_variables = self.cppbqm.num_variables()
        cdef Py_ssize_t num_interactions = self.cppbqm.num_interactions()

        # numpy arrays, we will return these
        ldata = np.empty(num_variables, dtype=self.dtype)
        irow = np.empty(num_interactions, dtype=self.index_dtype)
        icol = np.empty(num_interactions, dtype=self.index_dtype)
        qdata = np.empty(num_interactions, dtype=self.dtype)

        # views into the numpy arrays for faster cython access
        cdef bias_type[:] ldata_view = ldata
        cdef index_type[:] irow_view = irow
        cdef index_type[:] icol_view = icol
        cdef bias_type[:] qdata_view = qdata

        cdef Py_ssize_t vi
        cdef Py_ssize_t qi = 0  # index in the quadratic arrays
        for vi in range(num_variables):
            span = self.cppbqm.neighborhood(vi)

            while span.first != span.second and deref(span.first).first < vi:
                irow_view[qi] = vi
                icol_view[qi] = deref(span.first).first
                qdata_view[qi] = deref(span.first).second

                qi += 1
                inc(span.first)

        # at this point we have the arrays but they are index-order, NOT the
        # label-order. So we need to do some fiddling
        cdef bint is_range = self.variables._is_range()

        cdef Py_ssize_t[:] reindex
        cdef Py_ssize_t ri
        if variable_order is not None or (sort_labels and not is_range):
            if variable_order is None:
                variable_order = list(self.variables)
                if sort_labels:
                    try:
                        variable_order.sort()
                    except TypeError:
                        # can't sort unlike types in py3
                        pass

            # ok, using the variable_order, calculate the re-index
            reindex = np.full(num_variables, -1, dtype=np.intp)
            for ri, v in enumerate(variable_order):
                vi = self.variables.index(v)
                reindex[vi] = ri

                ldata_view[ri] = self.cppbqm.linear(vi)

            for qi in range(num_interactions):
                irow_view[qi] = reindex[irow_view[qi]]
                icol_view[qi] = reindex[icol_view[qi]]

            labels = variable_order

        else:
            # the fast case! We don't need to do anything except construct the
            # linear
            for vi in range(num_variables):
                ldata_view[vi] = self.cppbqm.linear(vi)

            if return_labels:
                labels = list(self.variables)

        if sort_indices:
            coo_sort(irow, icol, qdata)

        # todo: nametuple
        ret = [ldata, (irow, icol, qdata), self.offset]

        if return_labels:
            ret.append(labels)

        return tuple(ret)

    def _update(self, cyBQM other):
        # get the reindexing
        cdef vector[Py_ssize_t] mapping
        mapping.reserve(other.num_variables())
        for v in other.variables:
            mapping.push_back(self.variables.index(v, permissive=True))

        self.cppbqm.add_bqm(other.cppbqm, mapping)

        assert self.variables.size() == self.cppbqm.num_variables()

    def update(self, other):
        try:
            self._update(other)
        except TypeError:
            # if other is not a cybqm, defer back to the caller
            raise NotImplementedError from None
