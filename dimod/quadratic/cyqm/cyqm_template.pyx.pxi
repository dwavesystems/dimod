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

from copy import deepcopy

from cython.operator cimport preincrement as inc, dereference as deref

from dimod.binary.cybqm cimport cyBQM
from dimod.cyutilities cimport as_numpy_float
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

    cdef void _set_linear(self, Py_ssize_t vi, bias_type bias):
        # unsafe version of .set_linear
        cdef bias_type *b = &(self.cppqm.linear(vi))
        b[0] = bias

    def add_linear(self, v, bias_type bias):
        cdef Py_ssize_t vi = self.variables.index(v)
        self._add_linear(vi, bias)

    def add_quadratic(self, u, v, bias_type bias):
        if u == v:
            raise NotImplementedError

        cdef Py_ssize_t ui = self.variables.index(u)
        cdef Py_ssize_t vi = self.variables.index(v)
        self.cppqm.add_quadratic(ui, vi, bias)

    def add_variable(self, vartype, label=None):
        # as_vartype will raise for unsupported vartypes
        vartype = as_vartype(vartype, extended=True)
        cdef cppVartype cppvartype = self.cppvartype(vartype)

        if label is not None and self.variables.count(label):
            # it already exists, so check that vartype matches
            if self.cppqm.vartype(self.variables.index(label)) != cppvartype:
                raise TypeError(f"variable {label} already exists with a different vartype")
            return label

        # we're adding a new one
        self.variables._append(label)
        self.cppqm.add_variable(cppvartype)

        assert self.cppqm.num_variables() == self.variables.size()

        return self.variables.at(-1)

    cdef cppVartype cppvartype(self, object vartype) except? cppVartype.SPIN:
        if vartype is Vartype.SPIN:
            return cppVartype.SPIN
        elif vartype is Vartype.BINARY:
            return cppVartype.BINARY
        elif vartype is Vartype.INTEGER:
            return cppVartype.INTEGER
        else:
            raise TypeError(f"unexpected vartype {vartype!r}")

    def get_linear(self, v):
        cdef Py_ssize_t vi = self.variables.index(v)
        cdef bias_type bias = self.cppqm.linear(vi)
        return as_numpy_float(bias)

    def get_quadratic(self, u, v, default=None):
        cdef Py_ssize_t ui = self.variables.index(u)
        cdef Py_ssize_t vi = self.variables.index(v)

        if ui == vi:
            raise NotImplementedError  # todo

        # todo: catch error
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

    cpdef Py_ssize_t num_interactions(self):
        return self.cppqm.num_interactions()

    cpdef Py_ssize_t num_variables(self):
        return self.cppqm.num_variables()

    cpdef void scale(self, bias_type scalar):
        self.cppqm.scale(scalar)

    def set_linear(self, v, bias_type bias):
        cdef Py_ssize_t vi = self.variables.index(v)
        self._set_linear(vi, bias)

    def set_quadratic(self, u, v, bias_type bias):
        cdef Py_ssize_t ui = self.variables.index(u)
        cdef Py_ssize_t vi = self.variables.index(v)

        if ui == vi:
            raise NotImplementedError  # todo
        
        self.cppqm.set_quadratic(ui, vi, bias)

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
