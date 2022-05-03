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
import os

from copy import deepcopy

cimport cython

from cython.operator cimport preincrement as inc, dereference as deref
from libc.math cimport ceil, floor
from libcpp.vector cimport vector

import dimod

from dimod.binary.cybqm cimport cyBQM
from dimod.cyutilities cimport as_numpy_float, ConstInteger, ConstNumeric, cppvartype
from dimod.libcpp cimport cppvartype_info
from dimod.quadratic cimport cyQM
from dimod.sampleset import as_samples
from dimod.variables import Variables
from dimod.vartypes import as_vartype, Vartype


ctypedef fused cyBQM_and_QM:
    cyBQM
    cyQM


cdef class cyQM_template(cyQMBase):
    def __init__(self):
        self.dtype = np.dtype(BIAS_DTYPE)
        self.index_dtype = np.dtype(INDEX_DTYPE)
        self.variables = Variables()

        self.REAL_INTERACTIONS = dimod.REAL_INTERACTIONS

    def __deepcopy__(self, memo):
        cdef cyQM_template new = type(self)()
        new.cppqm = self.cppqm
        new.variables = deepcopy(self.variables, memo)

        new.REAL_INTERACTIONS = self.REAL_INTERACTIONS

        memo[id(self)] = new
        return new

    @property
    def offset(self):
        """Constant energy offset associated with the model."""
        return as_numpy_float(self.cppqm.offset())

    @offset.setter
    def offset(self, bias_type offset):
        cdef bias_type *b = &(self.cppqm.offset())
        b[0] = offset

    cdef void _add_linear(self, Py_ssize_t vi, bias_type bias):
        # unsafe version of .add_linear
        cdef bias_type *b = &(self.cppqm.linear(vi))
        b[0] += bias

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ilinear(self):
        """Return the linear biases in a numpy array."""
        cdef bias_type[:] ldata = np.empty(self.num_variables(), dtype=self.dtype)
        cdef Py_ssize_t vi
        for vi in range(self.num_variables()):
            ldata[vi] = self.cppqm.linear(vi)
        return ldata

    def _ilower_triangle(self, Py_ssize_t vi):
        cdef Py_ssize_t degree = self.cppqm.num_interactions(vi)

        dtype = np.dtype([('v', self.index_dtype), ('b', self.dtype)],
                         align=False)
        neighbors = np.empty(degree, dtype=dtype)

        cdef index_type[:] index_view = neighbors['v']
        cdef bias_type[:] bias_view = neighbors['b']

        span = self.cppqm.neighborhood(vi)
        cdef Py_ssize_t i = 0
        while span.first != span.second:
            if deref(span.first).first > vi:
                break
            index_view[i] = deref(span.first).first
            bias_view[i] = deref(span.first).second

            i += 1
            inc(span.first)

        return neighbors[:i]

    def _ilower_triangle_load(self, Py_ssize_t vi, Py_ssize_t num_neighbors, buff):
        dtype = np.dtype([('v', self.index_dtype), ('b', self.dtype)],
                         align=False)

        arr = np.frombuffer(buff[:dtype.itemsize*num_neighbors], dtype=dtype)
        cdef const index_type[:] index_view = arr['v']
        cdef const bias_type[:] bias_view = arr['b']

        cdef Py_ssize_t i
        for i in range(num_neighbors):
            self.cppqm.add_quadratic(vi, index_view[i], bias_view[i])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ivartypes(self):
        cdef Py_ssize_t num_variables = self.num_variables()

        # we could use the bias_type size to determine the vartype dtype to get
        # more alignment, but it complicates the code, so let's keep it simple
        dtype = np.dtype([('vartype', np.int8),
                          ('lb', BIAS_DTYPE), ('ub', BIAS_DTYPE)],
                         align=False)
        arr = np.empty(num_variables, dtype)

        cdef np.int8_t[:] vartype_view = arr['vartype']
        cdef bias_type[:] lb_view = arr['lb']
        cdef bias_type[:] ub_view = arr['ub']

        cdef Py_ssize_t vi
        for vi in range(self.num_variables()):
            vartype_view[vi] = self.cppqm.vartype(vi)
            lb_view[vi] = self.cppqm.lower_bound(vi)
            ub_view[vi] = self.cppqm.upper_bound(vi)

        return memoryview(arr).cast('B')

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ivartypes_load(self, buff, Py_ssize_t num_variables):
        if self.num_variables():
            raise RuntimeError("cannot load vartypes into a model with variables")

        # use the bias size to determine the vartype size since we're not
        # letting numpy handle the alignment
        dtype = np.dtype([('vartype', np.int8),
                          ('lb', BIAS_DTYPE), ('ub', BIAS_DTYPE)],
                         align=False)

        arr = np.frombuffer(buff[:dtype.itemsize*num_variables], dtype=dtype)
        cdef const np.int8_t[:] vartype_view = arr['vartype']
        cdef const bias_type[:] lb_view = arr['lb']
        cdef const bias_type[:] ub_view = arr['ub']

        cdef Py_ssize_t vi
        cdef cppVartype cpp_vartype
        for vi in range(num_variables):
            # I hate that we have to do this manually, but cython doesn't
            # really like static_cast in this context
            if vartype_view[vi] == 0:
                cpp_vartype = cppVartype.BINARY
            elif vartype_view[vi] == 1:
                cpp_vartype = cppVartype.SPIN
            elif vartype_view[vi] == 2:
                cpp_vartype = cppVartype.INTEGER
            elif vartype_view[vi] == 3:
                cpp_vartype = cppVartype.REAL
            else:
                raise RuntimeError
            self.cppqm.add_variable(cpp_vartype, lb_view[vi], ub_view[vi])

        while self.variables.size() < self.cppqm.num_variables():
            self.variables._append()

    cdef void _set_linear(self, Py_ssize_t vi, bias_type bias):
        # unsafe version of .set_linear
        cdef bias_type *b = &(self.cppqm.linear(vi))
        b[0] = bias

    def add_linear(self, v, bias_type bias, *,
                   default_vartype=None,
                   default_lower_bound=None,
                   default_upper_bound=None,
                   ):
        cdef Py_ssize_t vi

        if default_vartype is None or self.variables.count(v):
            # already present
            vi = self.variables.index(v)
        else:
            # we need to add it
            vi = self.num_variables()
            self.add_variable(default_vartype, v,
                              lower_bound=default_lower_bound,
                              upper_bound=default_upper_bound,
                              )

        self._add_linear(vi, bias)

    def add_linear_from_array(self, ConstNumeric[:] linear):
        cdef Py_ssize_t length = linear.shape[0]
        cdef Py_ssize_t vi

        if self.variables._is_range():
            if length > self.num_variables():
                raise ValueError("variables must already exist")
            for vi in range(length):
                self._add_linear(vi, linear[vi])
        else:
            for vi in range(length):
                self.add_linear(vi, linear[vi])

    cdef Py_ssize_t _add_quadratic(self, index_type ui, index_type vi, bias_type bias) except -1:
        # note: does not test that ui, vi are valid

        if ui == vi:
            if self.cppqm.vartype(ui) == cppVartype.SPIN:
                raise ValueError(f"SPIN variables (e.g. {self.variables[ui]!r}) "
                                 "cannot have interactions with themselves"
                                 )
            if self.cppqm.vartype(ui) == cppVartype.BINARY:
                raise ValueError(f"BINARY variables (e.g. {self.variables[ui]!r}) "
                                 "cannot have interactions with themselves"
                                 )

        if not self.REAL_INTERACTIONS:
            if self.cppqm.vartype(ui) == cppVartype.REAL:
                raise ValueError(
                    f"REAL variables (e.g. {self.variables[ui]!r}) "
                    "cannot have interactions"
                    )
            if self.cppqm.vartype(vi) == cppVartype.REAL:
                raise ValueError(
                    f"REAL variables (e.g. {self.variables[vi]!r}) "
                    "cannot have interactions"
                    )

        self.cppqm.add_quadratic(ui, vi, bias)

    def add_quadratic(self, object u, object v, bias_type bias):
        cdef Py_ssize_t ui = self.variables.index(u)
        cdef Py_ssize_t vi = self.variables.index(v)
        self._add_quadratic(ui, vi, bias)

    def add_quadratic_from_arrays(self,
                                  ConstInteger[::1] irow, ConstInteger[::1] icol,
                                  ConstNumeric[::1] qdata):
        if not irow.shape[0] == icol.shape[0] == qdata.shape[0]:
            raise ValueError("quadratic vectors should be equal length")
        cdef Py_ssize_t length = irow.shape[0]

        cdef Py_ssize_t vi
        if self.variables._is_range():
            if length > self.num_variables():
                raise ValueError("variables must already exist")

            for vi in range(length):
                self._add_quadratic(irow[vi], icol[vi], qdata[vi])
        else:
            for vi in range(length):
                self.add_quadratic(irow[vi], icol[vi], qdata[vi])

    def add_quadratic_from_iterable(self, quadratic):
        cdef Py_ssize_t ui, vi
        cdef bias_type bias
        for u, v, bias in quadratic:
            ui = self.variables.index(u)
            vi = self.variables.index(v)
            self._add_quadratic(ui, vi, bias)

    def add_variable(self, vartype, label=None, *, lower_bound=None, upper_bound=None):
        if not isinstance(vartype, Vartype):  # redundant, but provides a bit of a speedup
            vartype = as_vartype(vartype, extended=True)
        cdef cppVartype cppvartype = self.cppvartype(vartype)

        cdef bias_type lb
        cdef bias_type ub

        cdef Py_ssize_t vi
        if label is not None and self.variables.count(label):
            # it already exists, so check that vartype matches
            vi = self.variables.index(label)
            if self.cppqm.vartype(vi) != cppvartype:
                raise TypeError(f"variable {label!r} already exists with a different vartype")
            if cppvartype != cppVartype.BINARY and cppvartype != cppVartype.SPIN:
                if lower_bound is not None:
                    lb = lower_bound
                    if lb != self.cppqm.lower_bound(vi):
                        raise ValueError(
                            f"the specified lower bound, {lower_bound}, for "
                            f"variable {label!r} is different than the existing lower "
                            f"bound, {int(self.cppqm.lower_bound(vi))}")
                if upper_bound is not None:
                    ub = upper_bound
                    if ub != self.cppqm.upper_bound(vi):
                        raise ValueError(
                            f"the specified upper bound, {upper_bound}, for "
                            f"variable {label!r} is different than the existing upper "
                            f"bound, {int(self.cppqm.upper_bound(vi))}")

            return label

        if cppvartype == cppVartype.BINARY or cppvartype == cppVartype.SPIN:
            # in this case we just ignore the provided values
            lb = cppvartype_info[bias_type].default_min(cppvartype)
            ub = cppvartype_info[bias_type].default_max(cppvartype)
        elif cppvartype == cppVartype.INTEGER or cppvartype == cppVartype.REAL:
            if lower_bound is None:
                lb = cppvartype_info[bias_type].default_min(cppvartype)
            else:
                lb = lower_bound
                if lb < cppvartype_info[bias_type].min(cppvartype):
                    raise ValueError(f"lower_bound cannot be less than {cppvartype_info[bias_type].min(cppvartype)}")

            if upper_bound is None:
                ub = cppvartype_info[bias_type].default_max(cppvartype)
            else:
                ub = upper_bound
                if ub > cppvartype_info[bias_type].max(cppvartype):
                    raise ValueError(f"upper_bound cannot be greater than {cppvartype_info[bias_type].max(cppvartype)}")
            
            if lb > ub:
                raise ValueError("lower_bound must be less than or equal to upper_bound")

            if cppvartype == cppVartype.INTEGER and ceil(lb) > floor(ub):
                raise ValueError("there must be at least one valid integer between lower_bound and upper_bound")
        else:
            raise RuntimeError("unknown vartype")

        self.cppqm.add_variable(cppvartype, lb, ub)

        self.variables._append(label)

        assert self.cppqm.num_variables() == self.variables.size()

        return self.variables.at(-1)

    def change_vartype(self, vartype, v):
        vartype = as_vartype(vartype, extended=True)
        cdef Py_ssize_t vi = self.variables.index(v)
        try:
            self.cppqm.change_vartype(self.cppvartype(vartype), vi)
        except RuntimeError:
            # c++ logic_error
            raise TypeError(f"cannot change vartype {self.vartype(v).name!r} "
                            f"to {vartype.name!r}") from None

    cdef cppVartype cppvartype(self, object vartype) except? cppVartype.SPIN:
        return cppvartype(vartype)

    cdef const cppQuadraticModel[bias_type, index_type]* data(self):
        """Return a pointer to the C++ QuadraticModel."""
        return &self.cppqm

    def degree(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        return self.cppqm.num_interactions(vi)

    cdef np.float64_t[::1] _energies(self, ConstNumeric[:, ::1] samples, object labels):
        cdef Py_ssize_t num_samples = samples.shape[0]
        cdef Py_ssize_t num_variables = samples.shape[1]

        if num_variables != len(labels):
            # as_samples should never return inconsistent sizes, but we do this
            # check because the boundscheck is off and we otherwise might get
            # segfaults
            raise RuntimeError("as_samples returned an inconsistent samples/variables")

        # get the indices of the QM variables. Use -1 to signal that a
        # variable's index has not yet been set
        cdef Py_ssize_t[::1] qm_to_sample = np.full(self.num_variables(), -1, dtype=np.intp)
        cdef Py_ssize_t si
        for si in range(num_variables):
            v = labels[si]
            if self.variables.count(v):
                qm_to_sample[self.variables.index(v)] = si

        # make sure that all of the QM variables are accounted for
        for si in range(self.num_variables()):
            if qm_to_sample[si] == -1:
                raise ValueError(f"missing variable {self.variables[si]!r} in sample(s)")

        cdef np.float64_t[::1] energies = np.empty(num_samples, dtype=np.float64)

        # alright, now let's calculate some energies!
        cdef Py_ssize_t ui, vi
        for si in range(num_samples):
            # offset
            energies[si] = self.cppqm.offset()

            for ui in range(self.num_variables()):
                # linear
                energies[si] += self.cppqm.linear(ui) * samples[si, qm_to_sample[ui]];

                span = self.cppqm.neighborhood(ui)
                while span.first != span.second and deref(span.first).first <= ui:
                    vi = deref(span.first).first

                    energies[si] += deref(span.first).second * samples[si, qm_to_sample[ui]] * samples[si, qm_to_sample[vi]]

                    inc(span.first)

        return energies

    def energies(self, samples_like, dtype=None):
        samples, labels = as_samples(samples_like)

        # we need contiguous and unsigned. as_samples actually enforces contiguous
        # but no harm in double checking for some future-proofness
        samples = np.ascontiguousarray(
                samples,
                dtype=f'i{samples.dtype.itemsize}' if np.issubdtype(samples.dtype, np.unsignedinteger) else None,
                )

        # Cython really should be able to figure the type out, but for some reason
        # it fails, so we just dispatch manually
        if samples.dtype == np.float64:
            return np.asarray(self._energies[np.float64_t](samples, labels), dtype=dtype)
        elif samples.dtype == np.float32:
            return np.asarray(self._energies[np.float32_t](samples, labels), dtype=dtype)
        elif samples.dtype == np.int8:
            return np.asarray(self._energies[np.int8_t](samples, labels), dtype=dtype)
        elif samples.dtype == np.int16:
            return np.asarray(self._energies[np.int16_t](samples, labels), dtype=dtype)
        elif samples.dtype == np.int32:
            return np.asarray(self._energies[np.int32_t](samples, labels), dtype=dtype)
        elif samples.dtype == np.int64:
            return np.asarray(self._energies[np.int64_t](samples, labels), dtype=dtype)
        else:
            raise ValueError("unsupported sample dtype")

    @classmethod
    def from_cybqm(cls, cyBQM bqm):

        cdef cyQM_template qm = cls()

        qm.offset = bqm.offset

        # linear
        cdef Py_ssize_t vi
        cdef cppVartype vartype = bqm.cppbqm.vartype()
        for vi in range(bqm.num_variables()):
            qm.cppqm.add_variable(vartype)
            qm._set_linear(vi, bqm.cppbqm.linear(vi))
        qm.variables._extend(bqm.variables)

        # quadratic
        it = bqm.cppbqm.cbegin_quadratic()
        while it != bqm.cppbqm.cend_quadratic():
            qm.cppqm.set_quadratic(deref(it).u, deref(it).v, deref(it).bias)
            inc(it)

        return qm

    def get_linear(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        cdef bias_type bias = self.cppqm.linear(vi)
        return as_numpy_float(bias)

    def get_quadratic(self, u, v, default=None):
        cdef Py_ssize_t ui = self.variables.index(u)
        cdef Py_ssize_t vi = self.variables.index(v)

        if ui == vi:
            if self.cppqm.vartype(ui) == cppVartype.SPIN:
                raise ValueError(f"SPIN variables (e.g. {self.variables[ui]!r}) "
                                 "cannot have interactions with themselves"
                                 )
            if self.cppqm.vartype(ui) == cppVartype.BINARY:
                raise ValueError(f"BINARY variables (e.g. {self.variables[ui]!r}) "
                                 "cannot have interactions with themselves"
                                 )

        cdef bias_type bias
        try:
            bias = self.cppqm.quadratic_at(ui, vi)
        except IndexError:
            if default is None:
                # out of range error is automatically converted to IndexError
                raise ValueError(f"{u!r} and {v!r} have no interaction") from None
            bias = default
        return as_numpy_float(bias)

    cpdef bint is_linear(self):
        return self.cppqm.is_linear()

    def iter_neighborhood(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)

        cdef Py_ssize_t ui
        cdef bias_type bias

        span = self.cppqm.neighborhood(vi)
        while span.first != span.second:
            ui = deref(span.first).first
            bias = deref(span.first).second

            yield self.variables.at(ui), as_numpy_float(bias)

            inc(span.first)

    def iter_quadratic(self):
        it = self.cppqm.cbegin_quadratic()
        while it != self.cppqm.cend_quadratic():
            u = self.variables.at(deref(it).u)
            v = self.variables.at(deref(it).v)
            yield u, v, as_numpy_float(deref(it).bias)
            inc(it)

    def lower_bound(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        return as_numpy_float(self.cppqm.lower_bound(vi))

    cpdef Py_ssize_t nbytes(self, bint capacity = False):
        return self.cppqm.nbytes(capacity)

    cpdef Py_ssize_t num_interactions(self):
        return self.cppqm.num_interactions()

    cpdef Py_ssize_t num_variables(self):
        return self.cppqm.num_variables()

    def relabel_variables(self, mapping):
        self.variables._relabel(mapping)

    def remove_interaction(self, u, v):
        cdef Py_ssize_t ui = self.variables.index(u)
        cdef Py_ssize_t vi = self.variables.index(v)
        
        if not self.cppqm.remove_interaction(ui, vi):
            raise ValueError(f"{u!r} and {v!r} have no interaction")

    def remove_variable(self, v=None):
        if v is None:
            try:
                v = self.variables[-1]
            except KeyError:
                raise ValueError("cannot pop from an empty model")

        cdef Py_ssize_t vi = self.variables.index(v)
        cdef Py_ssize_t lasti = self.num_variables() - 1

        if vi != lasti:
            # we're removing a variable in the middle of the
            # underlying adjacency. We do this by "swapping" the last variable
            # and v, then popping v from the end
            self.cppqm.swap_variables(vi, lasti)

            # now swap the variable labels
            last = self.variables.at(lasti)
            self.variables._relabel({v: last, last: v})

        # remove last from the cppqm and variables
        self.cppqm.resize(lasti)
        tmp = self.variables._pop()

        assert tmp == v, f"{tmp} == {v}"

        return v

    def relabel_variables_as_integers(self):
        return self.variables._relabel_as_integers()

    def reduce_linear(self, function, initializer=None):
        if self.num_variables() == 0 and initializer is None:
            # feels like this should be a ValueError but python raises
            # TypeError so...
            raise TypeError("reduce_linear() on an empty BQM")

        cdef Py_ssize_t start, vi
        cdef bias_type value, tmp

        if initializer is None:
            start = 1
            value = self.cppqm.linear(0)
        else:
            start = 0
            value = initializer

        # speed up a few common cases
        if function is operator.add:
            for vi in range(start, self.num_variables()):
                value += self.cppqm.linear(vi)
        elif function is max:
            for vi in range(start, self.num_variables()):
                tmp = self.cppqm.linear(vi)
                if tmp > value:
                    value = tmp
        elif function is min:
            for vi in range(start, self.num_variables()):
                tmp = self.cppqm.linear(vi)
                if tmp < value:
                    value = tmp
        else:
            for vi in range(start, self.num_variables()):
                value = function(value, self.cppqm.linear(vi))

        return as_numpy_float(value)

    def reduce_neighborhood(self, u, function, initializer=None):
        cdef Py_ssize_t ui = self.variables.index(u)

        if self.cppqm.num_interactions(ui) == 0 and initializer is None:
            # feels like this should be a ValueError but python raises
            # TypeError so...
            raise TypeError("reduce_neighborhood() on an empty neighbhorhood")

        cdef bias_type value, tmp

        span = self.cppqm.neighborhood(ui)

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

        if self.cppqm.is_linear() and initializer is None:
            # feels like this should be a ValueError but python raises
            # TypeError so...
            raise TypeError("reduce_quadratic() on a linear model")

        cdef bias_type value, tmp

        start = self.cppqm.cbegin_quadratic()

        if initializer is None:
            value = deref(start).bias
            inc(start)
        else:
            value = initializer

        # handle a few common cases
        if function is operator.add:
            while start != self.cppqm.cend_quadratic():
                value += deref(start).bias
                inc(start)
        elif function is max:
            while start != self.cppqm.cend_quadratic():
                tmp = deref(start).bias
                if tmp > value:
                    value = tmp
                inc(start)
        elif function is min:
            while start != self.cppqm.cend_quadratic():
                tmp = deref(start).bias
                if tmp < value:
                    value = tmp
                inc(start)
        else:
            while start != self.cppqm.cend_quadratic():
                value = function(value, deref(start).bias)
                inc(start)

        return as_numpy_float(value)

    cpdef void scale(self, bias_type scalar):
        self.cppqm.scale(scalar)

    def set_linear(self, v, bias_type bias):
        cdef Py_ssize_t vi = self.variables.index(v)
        self._set_linear(vi, bias)

    def set_lower_bound(self, v, bias_type lb):
        cdef Py_ssize_t vi = self.variables.index(v)
        cdef cppVartype cppvartype = self.cppqm.vartype(vi)

        if cppvartype == cppVartype.BINARY or cppvartype == cppVartype.SPIN:
            raise ValueError(
                "cannot set the lower bound for BINARY or SPIN variables, "
                f"{v!r} is a {self.vartype(v).name} variable")

        if lb < cppvartype_info[bias_type].min(cppvartype):
            raise ValueError(f"lower_bound cannot be less than {cppvartype_info[bias_type].min(cppvartype)}")
            
        if lb > self.cppqm.upper_bound(vi):
            raise ValueError(
                f"the specified lower bound, {lb}, cannot be set greater than the "
                f"current upper bound, {self.cppqm.upper_bound(vi)}"
                )

        if cppvartype == cppVartype.INTEGER:
            if ceil(lb) > floor(self.cppqm.upper_bound(vi)):
                raise ValueError(
                    "there must be at least one integer value between "
                    f"the specified lower bound, {lb} and the "
                    f"current upper bound, {self.cppqm.upper_bound(vi)}"
                    )

        cdef bias_type *b = &(self.cppqm.lower_bound(vi))
        b[0] = lb

    def set_upper_bound(self, v, bias_type ub):
        cdef Py_ssize_t vi = self.variables.index(v)
        cdef cppVartype cppvartype = self.cppqm.vartype(vi)

        if cppvartype == cppVartype.BINARY or cppvartype == cppVartype.SPIN:
            raise ValueError(
                "cannot set the upper bound for BINARY or SPIN variables, "
                f"{v!r} is a {self.vartype(v).name} variable")

        if ub > cppvartype_info[bias_type].max(cppvartype):
            raise ValueError(f"upper_bound cannot be more than {cppvartype_info[bias_type].max(cppvartype)}")
            
        if ub < self.cppqm.lower_bound(vi):
            raise ValueError(
                f"the specified upper bound, {ub}, cannot be set less than the "
                f"current lower bound, {self.cppqm.lower_bound(vi)}"
                )

        if cppvartype == cppVartype.INTEGER:
            if ceil(self.cppqm.lower_bound(vi)) > floor(ub):
                raise ValueError(
                    "there must be at least one integer value between "
                    f"the specified upper bound, {ub} and the "
                    f"current lower bound, {self.cppqm.lower_bound(vi)}"
                    )

        cdef bias_type *b = &(self.cppqm.upper_bound(vi))
        b[0] = ub

    def set_quadratic(self, u, v, bias_type bias):
        cdef Py_ssize_t ui = self.variables.index(u)
        cdef Py_ssize_t vi = self.variables.index(v)

        if ui == vi:
            if self.cppqm.vartype(ui) == cppVartype.SPIN:
                raise ValueError(f"SPIN variables (e.g. {self.variables[ui]!r}) "
                                 "cannot have interactions with themselves"
                                 )
            if self.cppqm.vartype(ui) == cppVartype.BINARY:
                raise ValueError(f"BINARY variables (e.g. {self.variables[ui]!r}) "
                                 "cannot have interactions with themselves"
                                 )

        if not self.REAL_INTERACTIONS:
            if self.cppqm.vartype(ui) == cppVartype.REAL:
                raise ValueError(
                    f"REAL variables (e.g. {self.variables[ui]!r}) "
                    "cannot have interactions"
                    )
            if self.cppqm.vartype(vi) == cppVartype.REAL:
                raise ValueError(
                    f"REAL variables (e.g. {self.variables[vi]!r}) "
                    "cannot have interactions"
                    )

        self.cppqm.set_quadratic(ui, vi, bias)

    def update(self, cyBQM_and_QM other):
        # we'll need a mapping from the other's variables to ours
        cdef vector[Py_ssize_t] mapping
        mapping.reserve(other.num_variables())

        cdef Py_ssize_t vi

        # first make sure that any variables that overlap match in terms of
        # vartype and bounds
        for vi in range(other.num_variables()):
            v = other.variables.at(vi)
            if self.variables.count(v):
                # there is a variable already
                mapping.push_back(self.variables.index(v))

                if self.cppqm.vartype(mapping[vi]) != other.data().vartype(vi):
                    raise ValueError(f"conflicting vartypes: {v!r}")

                if self.cppqm.lower_bound(mapping[vi]) != other.data().lower_bound(vi):
                    raise ValueError(f"conflicting lower bounds: {v!r}")

                if self.cppqm.upper_bound(mapping[vi]) != other.data().upper_bound(vi):
                    raise ValueError(f"conflicting upper bounds: {v!r}")
            else:
                # not yet present, let's just track that fact for now
                # in case there is a mismatch so we don't modify our object yet
                mapping.push_back(-1)

        for vi in range(mapping.size()):
            if mapping[vi] != -1:
                continue  # already added and checked

            mapping[vi] = self.num_variables()  # we're about to add a new one

            v = other.variables.at(vi)
            vartype = other.vartype(v)

            self.add_variable(vartype, v,
                              lower_bound=other.data().lower_bound(vi),
                              upper_bound=other.data().upper_bound(vi),
                              )

        # variables are in place!
        
        # the linear biases
        for vi in range(mapping.size()):
            self._add_linear(mapping[vi], other.data().linear(vi))

        # the quadratic biases
        # dev note: for even more speed we could check that mapping is
        # a range, and in that case can just add them without the indirection
        # or the sorting.
        it = other.data().cbegin_quadratic()
        while it != other.data().cend_quadratic():
            self.cppqm.add_quadratic(
                mapping[deref(it).u],
                mapping[deref(it).v],
                deref(it).bias
                )
            inc(it)

        # the offset
        cdef bias_type *b = &(self.cppqm.offset())
        b[0] += other.data().offset()

    def upper_bound(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        return as_numpy_float(self.cppqm.upper_bound(vi))

    def vartype(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        cdef cppVartype cppvartype = self.cppqm.vartype(vi)

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
