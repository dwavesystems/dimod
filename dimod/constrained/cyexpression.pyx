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
from libcpp.algorithm cimport lower_bound as cpplower_bound
from libcpp.unordered_map cimport unordered_map

import dimod

from dimod.cyqmbase.cyqmbase_float64 import _dtype, _index_dtype
from dimod.cyutilities cimport as_numpy_float
from dimod.cyutilities cimport ConstNumeric
from dimod.cyvariables cimport cyVariables
from dimod.libcpp.abc cimport QuadraticModelBase as cppQuadraticModelBase
from dimod.libcpp.constrained_quadratic_model cimport Penalty as cppPenalty
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
        cdef cyVariables out = Variables()
        for vi in range(variables.size()):
            out._append(self.parent.variables.at(variables[vi]))
        return out

    def add_linear(self, v, bias_type bias):
        self.expression().add_linear(self.parent.variables.index(v), bias)

    def add_quadratic(self, u, v, bias):
        cdef Py_ssize_t ui = self.parent.variables.index(u)
        cdef Py_ssize_t vi = self.parent.variables.index(v)

        if ui == vi:
            if self.parent.cppcqm.vartype(ui) == cppVartype.SPIN:
                raise ValueError(f"SPIN variables (e.g. {self.variables[ui]!r}) "
                                 "cannot have interactions with themselves"
                                 )
            if self.parent.cppcqm.vartype(ui) == cppVartype.BINARY:
                raise ValueError(f"BINARY variables (e.g. {self.variables[ui]!r}) "
                                 "cannot have interactions with themselves"
                                 )

        if not self.parent.REAL_INTERACTIONS:
            if self.parent.cppcqm.vartype(ui) == cppVartype.REAL:
                raise ValueError(
                    f"REAL variables (e.g. {self.variables[ui]!r}) "
                    "cannot have interactions"
                    )
            if self.parent.cppcqm.vartype(vi) == cppVartype.REAL:
                raise ValueError(
                    f"REAL variables (e.g. {self.variables[vi]!r}) "
                    "cannot have interactions"
                    )

        self.expression().add_quadratic(ui, vi, bias)

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
        raise NotImplementedError

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ilinear(self):
        expression = self.expression()

        cdef bias_type[:] ldata = np.empty(expression.num_variables(), dtype=self.dtype)
        cdef Py_ssize_t vi
        for vi in range(expression.num_variables()):
            ldata[vi] = (<cppQuadraticModelBase[bias_type, index_type]*>expression).linear(vi)
        return np.asarray(ldata)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ivarinfo(self):
        expression = self.expression()

        cdef Py_ssize_t num_variables = expression.num_variables()

        # we could use the bias_type size to determine the vartype dtype to get
        # more alignment, but it complicates the code, so let's keep it simple
        # We choose the field names to mirror the internals of the QuadraticModel
        dtype = np.dtype([('vartype', np.int8), ('lb', self.dtype), ('ub', self.dtype)],
                         align=False)
        varinfo = np.empty(num_variables, dtype)

        cdef np.int8_t[:] vartype_view = varinfo['vartype']
        cdef bias_type[:] lb_view = varinfo['lb']
        cdef bias_type[:] ub_view = varinfo['ub']

        cdef Py_ssize_t i, vi
        for i in range(expression.num_variables()):
            vi = expression.variables()[i]
            vartype_view[i] = expression.vartype(vi)
            lb_view[i] = expression.lower_bound(vi)
            ub_view[i] = expression.upper_bound(vi)

        return varinfo

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ineighborhood(self, Py_ssize_t vi, bint lower_triangle=False):
        # Cython will not let us coerce to the underling QuadraticModelBase
        # because it gets confused about the iterator types.
        # So we do this pretty unnaturally.
        expression = self.expression()

        # Make a NumPy struct array.
        neighborhood = np.empty(
            (<cppQuadraticModelBase[bias_type, index_type]*>expression).degree(vi),
            dtype=np.dtype([('v', self.index_dtype), ('bias', self.dtype)], align=False))

        cdef index_type[:] index_view = neighborhood['v']
        cdef bias_type[:] biases_view = neighborhood['bias']

        if not index_view.shape[0]:
            return neighborhood

        cdef unordered_map[index_type, index_type] indices
        for i in range(expression.num_variables()):
            indices[expression.variables()[i]] = i

        it = expression.cbegin_neighborhood(expression.variables()[vi])
        cdef Py_ssize_t length = 0
        cdef Py_ssize_t ui
        while it != expression.cend_neighborhood(expression.variables()[vi]):
            ui = indices[deref(it).v]

            if ui > vi:
                break

            index_view[length] = ui
            biases_view[length] = deref(it).bias

            inc(it)
            length += 1

        return neighborhood[:length]

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
        # todo: migrate to mixin
        raise NotImplementedError

    def reduce_neighborhood(self):
        # todo: migrate to mixin
        raise NotImplementedError

    def reduce_quadratic(self):
        # todo: migrate to mixin
        raise NotImplementedError

    def remove_interaction(self, u, v):
        cdef Py_ssize_t ui = self.parent.variables.index(u)
        cdef Py_ssize_t vi = self.parent.variables.index(v)

        self.expression().remove_interaction(ui, vi)

    def remove_variable(self):
        raise NotImplementedError

    def set_linear(self, v, bias_type bias):
        cdef Py_ssize_t vi = self.parent.variables.index(v)
        self.expression().set_linear(vi, bias)

    def set_quadratic(self, u, v, bias):
        raise NotImplementedError

    def upper_bound(self, v):
        return self.parent.upper_bound(v)

    def vartype(self, v):
        return self.parent.vartype(v)


cdef class cyObjectiveView(_cyExpression):
    cdef cppExpression[bias_type, index_type]* expression(self) except NULL:
        return &(self.parent.cppcqm.objective)


cdef class cyConstraintView(_cyExpression):
    def __init__(self, cyConstrainedQuadraticModel parent, object label):
        super().__init__(parent)
        self.constraint_ptr = parent.cppcqm.constraint_weak_ptr(parent.constraint_labels.index(label))

    cdef cppConstraint[bias_type, index_type]* constraint(self) except NULL:
        if self.constraint_ptr.expired():
            raise RuntimeError("this constraint is no longer valid")
        return self.constraint_ptr.lock().get()

    cdef cppExpression[bias_type, index_type]* expression(self) except NULL:
        return self.constraint()

    def is_discrete(self):
        constraint = self.constraint()
        return constraint.marked_discrete() and constraint.is_onehot()

    def is_soft(self):
        return self.constraint().is_soft()

    def mark_discrete(self, bint marker = True):
        self.constraint().mark_discrete(marker)

    def penalty(self):
        penalty = self.constraint().penalty()

        if penalty == cppPenalty.LINEAR:
            return "linear"
        elif penalty == cppPenalty.QUADRATIC:
            return "quadratic"
        elif penalty == cppPenalty.CONSTANT:
            return "constrant"
        else:
            raise RuntimeError("unexpected penalty")

    def weight(self):
        return self.constraint().weight()
