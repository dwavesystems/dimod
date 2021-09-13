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

from copy import deepcopy

cimport cython

from cython.operator cimport preincrement as inc, dereference as deref
from libc.math cimport round as cppround
from libcpp.cast cimport static_cast

from dimod.binary.cybqm cimport cyBQM
from dimod.cyutilities cimport as_numpy_float, ConstInteger, ConstNumeric
from dimod.sampleset import as_samples
from dimod.variables import Variables
from dimod.vartypes import as_vartype, Vartype


cdef class cyQM_template(cyQMBase):
    def __init__(self):
        self.dtype = np.dtype(BIAS_DTYPE)
        self.index_dtype = np.dtype(INDEX_DTYPE)
        self.variables = Variables()

    def __deepcopy__(self, memo):
        cdef cyQM_template new = type(self)()
        new.cppqm = self.cppqm
        new.variables = deepcopy(self.variables, memo)
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
    cdef cython.floating[::1] _energies(self,
                                        object samples_like,
                                        cython.floating signal=0):
        samples, labels = as_samples(samples_like, dtype=np.int64)

        cdef np.int64_t[:, :] samples_view = samples

        cdef Py_ssize_t num_samples = samples_view.shape[0]
        cdef Py_ssize_t num_variables = samples_view.shape[1]

        if num_variables != len(labels):
            # as_samples should never return inconsistent sizes, but we do this
            # check because the boundscheck is off and we otherwise might get
            # segfaults
            raise RuntimeError(
                "as_samples returned an inconsistent samples/variables")

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
                raise ValueError(
                    f"missing variable {self.variables[si]!r} in sample(s)")

        cdef cython.floating[::1] energies
        if cython.floating is float:
            energies = np.empty(num_samples, dtype=np.float32)
        else:
            energies = np.empty(num_samples, dtype=np.float64)

        # alright, now let's calculate some energies!
        cdef np.int64_t uspin, vspin
        cdef Py_ssize_t ui, vi
        for si in range(num_samples):
            # offset
            energies[si] = self.cppqm.offset()

            for ui in range(self.num_variables()):
                uspin = samples_view[si, qm_to_sample[ui]]

                # linear
                energies[si] += self.cppqm.linear(ui) * uspin;

                span = self.cppqm.neighborhood(ui)
                while span.first != span.second and deref(span.first).first < ui:
                    vi = deref(span.first).first

                    vspin = samples_view[si, qm_to_sample[vi]]

                    energies[si] += deref(span.first).second * uspin * vspin

                    inc(span.first)

        return energies

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
            else:
                raise RuntimeError
            self.cppqm.add_variable(cpp_vartype, lb_view[vi], ub_view[vi])

        while self.variables.size() < self.cppqm.num_variables():
            self.variables._append()

    cdef void _set_linear(self, Py_ssize_t vi, bias_type bias):
        # unsafe version of .set_linear
        cdef bias_type *b = &(self.cppqm.linear(vi))
        b[0] = bias

    def add_linear(self, v, bias_type bias):
        cdef Py_ssize_t vi = self.variables.index(v)
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

    def add_quadratic(self, u, v, bias_type bias):
        cdef Py_ssize_t ui = self.variables.index(u)
        cdef Py_ssize_t vi = self.variables.index(v)

        if ui == vi and self.cppqm.vartype(ui) != cppVartype.INTEGER:
            raise ValueError(f"{u!r} cannot have an interaction with itself")

        self.cppqm.add_quadratic(ui, vi, bias)

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
                self.cppqm.add_quadratic(irow[vi], icol[vi], qdata[vi])
        else:
            for vi in range(length):
                self.add_quadratic(irow[vi], icol[vi], qdata[vi])

    def add_quadratic_from_iterable(self, quadratic):
        cdef Py_ssize_t ui, vi
        cdef bias_type bias

        for u, v, bias in quadratic:
            ui = self.variables.index(u)
            vi = self.variables.index(v)

            if ui == vi and self.cppqm.vartype(ui) != cppVartype.INTEGER:
                raise ValueError(f"{u!r} cannot have an interaction with itself")
            
            self.cppqm.add_quadratic(ui, vi, bias)

    def add_variable(self, vartype, label=None, *, lower_bound=0, upper_bound=None):
        # as_vartype will raise for unsupported vartypes
        vartype = as_vartype(vartype, extended=True)
        cdef cppVartype cppvartype = self.cppvartype(vartype)

        cdef bias_type lb = lower_bound
        cdef bias_type ub

        cdef Py_ssize_t vi
        if label is not None and self.variables.count(label):
            # it already exists, so check that vartype matches
            vi = self.variables.index(label)
            if self.cppqm.vartype(vi) != cppvartype:
                raise TypeError(f"variable {label!r} already exists with a different vartype")
            if cppvartype == cppVartype.INTEGER:
                if lb != self.cppqm.lower_bound(vi):
                    raise ValueError(
                        f"the specified lower bound, {lower_bound}, for "
                        f"variable {label!r} is different than the existing lower "
                        f"bound, {int(self.cppqm.lower_bound(vi))}")
                if upper_bound is not None:
                    ub = upper_bound
                    if ub != self.cppqm.upper_bound(vi):
                        raise ValueError(
                            f"the specified upper bound, {lower_bound}, for "
                            f"variable {label!r} is different than the existing upper "
                            f"bound, {int(self.cppqm.lower_bound(vi))}")

            return label

        if cppvartype == cppVartype.INTEGER and (lb != 0 or upper_bound is not None):
            if lb < -self.cppqm.max_integer():
                raise ValueError(f"lower_bound cannot be less than {-self.cppqm.max_integer()}")

            if upper_bound is None:
                ub = self.cppqm.max_integer()
            else:
                ub = upper_bound
                if ub > self.cppqm.max_integer():
                    raise ValueError(f"upper_bound cannot be greater than {self.cppqm.max_integer()}")

            if lb > ub:
                raise ValueError("lower_bound must be less than or equal to upper_bound")

            self.cppqm.add_variable(cppvartype, lb, ub)
        else:
            self.cppqm.add_variable(cppvartype)

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
        if vartype is Vartype.SPIN:
            return cppVartype.SPIN
        elif vartype is Vartype.BINARY:
            return cppVartype.BINARY
        elif vartype is Vartype.INTEGER:
            return cppVartype.INTEGER
        else:
            raise TypeError(f"unexpected vartype {vartype!r}")

    cdef const cppQuadraticModel[bias_type, index_type]* data(self):
        """Return a pointer to the C++ QuadraticModel."""
        return &self.cppqm

    def degree(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        return self.cppqm.num_interactions(vi)

    def energies(self, samples_like, dtype=None):
        dtype = self.dtype if dtype is None else np.dtype(dtype)
        if dtype == np.float64:
            return np.asarray(self._energies[np.float64_t](samples_like))
        elif dtype == np.float32:
            return np.asarray(self._energies[np.float32_t](samples_like))
        else:
            raise ValueError("dtype must be None or a floating type.")

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

        if ui == vi and self.cppqm.vartype(ui) != cppVartype.INTEGER:
            raise ValueError(f"{u!r} cannot have an interaction with itself")

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

        if self.cppqm.vartype(vi) != cppVartype.INTEGER:
            raise ValueError(
                "can only set the lower bound for INTEGER variables, "
                f"{v!r} is a {self.vartype(v).name} variable")

        lb = cppround(lb)  # we round to the nearest integer

        if lb < -self.cppqm.max_integer():
            raise ValueError(
                f"the specified lower bound, {int(lb)}, for variable {v!r} is less than "
                f"{int(-self.cppqm.max_integer())}, "
                f"the minimum for QuadraticModel(dtype=np.{self.dtype.name}) ")

        if lb > self.cppqm.upper_bound(vi):
            raise ValueError(
                f"the specified lower bound, {int(lb)}, cannot be set greater than the "
                f"current upper bound, {int(self.cppqm.upper_bound(vi))}"
                )

        cdef bias_type *b = &(self.cppqm.lower_bound(vi))
        b[0] = lb

    def set_upper_bound(self, v, bias_type ub):
        cdef Py_ssize_t vi = self.variables.index(v)

        if self.cppqm.vartype(vi) != cppVartype.INTEGER:
            raise ValueError(
                "can only set the upper bound for INTEGER variables, "
                f"{v!r} is a {self.vartype(v).name} variable")

        ub = cppround(ub)  # we round to the nearest integer

        if ub > self.cppqm.max_integer():
            raise ValueError(
                f"the specified upper bound, {int(ub)}, for variable {v!r} is greater than "
                f"{int(self.cppqm.max_integer())}, "
                f"the maximum for QuadraticModel(dtype=np.{self.dtype.name}) ")

        if ub < self.cppqm.lower_bound(vi):
            raise ValueError(
                f"the specified upper bound, {int(ub)}, cannot be set less than the "
                f"current lower bound, {int(self.cppqm.lower_bound(vi))}"
                )

        cdef bias_type *b = &(self.cppqm.upper_bound(vi))
        b[0] = ub

    def set_quadratic(self, u, v, bias_type bias):
        cdef Py_ssize_t ui = self.variables.index(u)
        cdef Py_ssize_t vi = self.variables.index(v)

        if ui == vi and self.cppqm.vartype(ui) != cppVartype.INTEGER:
            raise ValueError(f"{u!r} cannot have an interaction with itself")
        
        self.cppqm.set_quadratic(ui, vi, bias)

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
        else:
            raise RuntimeError("unexpected vartype")
