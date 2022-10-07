# Copyright 2022 D-Wave Systems Inc.
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

import numbers

cimport cython
cimport numpy as np
import numpy as np

from cython.operator cimport preincrement as inc, dereference as deref

from dimod.cyqmbase.cyqmbase_float64 import _dtype, _index_dtype
from dimod.cyutilities cimport as_numpy_float
from dimod.cyutilities cimport ConstNumeric
from dimod.cyvariables cimport cyVariables
from dimod.libcpp.abc cimport QuadraticModelBase as cppQuadraticModelBase
from dimod.libcpp.vartypes cimport Vartype as cppVartype
from dimod.sampleset import as_samples
from dimod.variables import Variables


cdef class _cyExpression:
    def __init__(self, cyConstrainedQuadraticModel parent):
        self.parent = parent

        self.dtype = _dtype
        self.index_dtype = _index_dtype

    def __repr__(self):
        vartypes = {v: self.vartype(v).name for v in self.variables}
        return f"{type(self).__name__}({self.linear}, {self.quadratic}, {self.offset}, {vartypes})"

    @property
    def num_interactions(self):
        return self.expression().num_interactions()

    @property
    def num_variables(self):
        return self.expression().num_variables()

    @property
    def offset(self):
        return as_numpy_float(self.expression().offset())

    @property
    def variables(self):
        # we could do yet another view, but that way lies madness
        # todo: raise better error for disconnected and test

        variables = self.expression().variables()  # todo: check this a reference?

        cdef Py_ssize_t vi
        out = []
        for vi in range(variables.size()):
            out.append(self.parent.variables.at(variables[vi]))
        return out

    def add_linear(self, v, bias):
        raise NotImplementedError

    def degree(self, v):
        return self.expression().degree(self.parent.variables.index(v))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _energies(self, ConstNumeric[:, ::1] samples, cyVariables labels):
        cdef cppExpression[bias_type, index_type]* expression = self.expression()

        cdef Py_ssize_t num_samples = samples.shape[0]
        cdef Py_ssize_t num_variables = samples.shape[1]

        # let's reindex, using the underlying variable order
        cdef index_type[:] reindex = np.empty(expression.num_variables(), dtype=self.index_dtype)

        for i in range(expression.num_variables()):
            reindex[i] = labels.index(self.parent.variables.at(expression.variables()[i]))

        # we could avoid the copy at this point by checking if it's sorted and
        # the same length of the sample array, but let's not for now

        # we could do this manually, but way better to let NumPy handle it
        cdef ConstNumeric[:, ::1] subsamples = np.ascontiguousarray(np.asarray(samples)[:, reindex])

        cdef np.float64_t[::1] energies = np.empty(num_samples, dtype=np.float64)
        cdef Py_ssize_t si
        if subsamples.shape[1]:
            for si in range(num_samples):
                # we cast to QuadraticModelBase so that we're using the underlying variable order
                # rather than parent's
                energies[si] = (<cppQuadraticModelBase[bias_type, index_type]*>expression).energy(&subsamples[si, 0])
        else:
            for si in range(num_samples):
                energies[si] = 0

        return energies

    def energies(self, samples_like):
        samples, labels = as_samples(samples_like, labels_type=Variables)

        # we need contiguous and unsigned. as_samples actually enforces contiguous
        # but no harm in double checking for some future-proofness
        samples = np.ascontiguousarray(
                samples,
                dtype=f'i{samples.dtype.itemsize}' if np.issubdtype(samples.dtype, np.unsignedinteger) else None,
                )

        try:
            return np.asarray(self._energies(samples, labels))
        except TypeError as err:
            if np.issubdtype(samples.dtype, np.floating) or np.issubdtype(samples.dtype, np.signedinteger):
                raise err
            raise ValueError(f"unsupported sample dtype: {samples.dtype.name}")

    cdef cppExpression[bias_type, index_type]* expression(self) except NULL:
        # Not implemented. To be overwritten by subclasses.
        # We can't raise an error without hurting performance
        raise NotImplementedError

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ilinear(self):
        raise NotImplementedError

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ivarinfo(self):
        raise NotImplementedError

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ineighborhood(self, Py_ssize_t vi, bint lower_triangle=False):
        raise NotImplementedError

    def get_linear(self, v):
        return as_numpy_float(self.expression().linear(self.parent.variables.index(v)))

    def get_quadratic(self, u, v, default=None):
        cdef Py_ssize_t ui = self.parent.variables.index(u)
        cdef Py_ssize_t vi = self.parent.variables.index(v)

        if ui == vi:
            if self.parent.cppcqm.vartype(ui) == cppVartype.SPIN:
                raise ValueError(f"SPIN variables (e.g. {u!r}) "
                                 "cannot have interactions with themselves"
                                 )
            if self.parent.cppcqm.vartype(ui) == cppVartype.BINARY:
                raise ValueError(f"BINARY variables (e.g. {v!r}) "
                                 "cannot have interactions with themselves"
                                 )
        cdef bias_type bias
        try:
            bias = self.expression().quadratic_at(ui, vi)
        except IndexError:
            # out of range error is automatically converted to IndexError
            if default is None:
                raise ValueError(f"{u!r} and {v!r} have no interaction") from None
            bias = default
        return as_numpy_float(bias)

    def is_linear(self):
        return self.expression().is_linear()

    def iter_neighborhood(self, v):
        cdef cppExpression[bias_type, index_type]* expression = self.expression()
        cdef Py_ssize_t vi = self.parent.variables.index(v)

        it = expression.cbegin_neighborhood(vi)
        while it != expression.cend_neighborhood(vi):
            v = self.parent.variables.at(deref(it).v)
            yield v, as_numpy_float(deref(it).bias)
            inc(it)

    def iter_quadratic(self):
        cdef cppExpression[bias_type, index_type]* expression = self.expression()

        it = expression.cbegin_quadratic()
        while it != expression.cend_quadratic():
            u = self.parent.variables.at(deref(it).u)
            v = self.parent.variables.at(deref(it).v)
            yield u, v, as_numpy_float(deref(it).bias)
            inc(it)

    def lower_bound(self, v):
        return self.parent.lower_bound(v)

    def reduce_linear(self):
        raise NotImplementedError

    def reduce_neighborhood(self):
        raise NotImplementedError

    def reduce_quadratic(self):
        raise NotImplementedError

    def remove_interaction(self):
        raise NotImplementedError

    def remove_variable(self):
        raise NotImplementedError

    def set_linear(self, v, bias):
        raise NotImplementedError

    def set_quadratic(self, u, v, bias):
        raise NotImplementedError

    def upper_bound(self, v):
        return self.parent.upper_bound(v)

    def vartype(self, v):
        return self.parent.vartype(v)


cdef class cyObjectiveView(_cyExpression):
    cdef cppExpression[bias_type, index_type]* expression(self) except NULL:
        return &(self.parent.cppcqm.objective)

    # def _ilinear(self):
    #     raise NotImplementedError

cdef class cyConstraintView(_cyExpression):
    def __init__(self, cyConstrainedQuadraticModel parent, object label):
        super().__init__(parent)
        self.label = label

    cdef cppConstraint[bias_type, index_type]* constraint(self) except NULL:
        # dev note: this is the only safe way I can think of to do this. I
        # thought about using shared_ptrs down at the C++ level, but I think
        # that will only introduce ownership issues, not to mention hurting
        # performance.
        # One thing we could do in the future is add a lock() method to this
        # class for when we know we're doing a bunch of operations in a row.
        # When locked we could keep a pointer alive.
        cdef Py_ssize_t ci 
        try:
           ci = self.parent.constraint_labels.index(self.label)
        except ValueError:
            raise RuntimeError(f"unknown constraint label {self.label!r}, "
                               "this constraint is no longer valid") from None
        return &self.parent.cppcqm.constraint_ref(ci)

    cdef cppExpression[bias_type, index_type]* expression(self) except NULL:
        return self.constraint()

    def is_discrete(self):
        constraint = self.constraint()
        return constraint.marked_discrete() and constraint.is_onehot()

    def is_soft(self):
        return self.constraint().is_soft()

    def mark_discrete(self, bint marker = True):
        self.constraint().mark_discrete(marker)
