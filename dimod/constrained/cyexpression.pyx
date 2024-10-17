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
import numpy as np

from cython.operator cimport preincrement as inc, dereference as deref
from libcpp.algorithm cimport lower_bound as cpplower_bound
from libcpp.unordered_map cimport unordered_map

import dimod

from dimod.cyqmbase.cyqmbase_float64 import _dtype, _index_dtype
from dimod.cyutilities cimport as_numpy_float
from dimod.cyvariables cimport cyVariables
from dimod.libcpp.abc cimport QuadraticModelBase as cppQuadraticModelBase
from dimod.libcpp.constrained_quadratic_model cimport Penalty as cppPenalty
from dimod.libcpp.vartypes cimport Vartype as cppVartype
from dimod.sampleset import as_samples
from dimod.serialization.fileview import (
    SpooledTemporaryFile, _BytesIO,
    read_header, write_header,
    IndicesSection, LinearSection, OffsetSection, NeighborhoodSection, QuadraticSection,
    )
from dimod.typing cimport Numeric, float64_t
from dimod.variables import Variables


EXPRESSION_MAGIC_PREFIX = b"DIMODEXPR"


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

    @offset.setter
    def offset(self, bias_type offset):
        self.expression().set_offset(offset)

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

    def add_variable(self, *args, **kwargs):
        """Add a variable to the model. See :meth:`ConstrainedQuadraticModel.add_variable()`."""
        return self.parent.add_variable(*args, **kwargs)

    def degree(self, v):
        return self.expression().degree(self.parent.variables.index(v))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _energies(self, const Numeric[:, ::1] samples, cyVariables labels):
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
        cdef const Numeric[:, ::1] subsamples = np.ascontiguousarray(np.asarray(samples)[:, reindex])

        cdef float64_t[::1] energies = np.empty(num_samples, dtype=np.float64)
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
    def _iindices(self):
        expression = self.expression()

        cdef index_type[:] indices = np.empty(expression.num_variables(), dtype=self.index_dtype)

        it = expression.variables().const_begin()
        end = expression.variables().const_end()
        cdef Py_ssize_t vi = 0
        while it != end:
            indices[vi] = deref(it)
            vi += 1
            inc(it)

        return np.asarray(indices)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _iindices_load(self, buff, Py_ssize_t num_variables, object dtype):
        expression = self.expression()

        if expression.num_variables():
            raise RuntimeError("indices can only be loaded into an empty expression")

        cdef const index_type[:] indices = np.frombuffer(buff[:dtype.itemsize*num_variables], dtype=dtype)
        for vi in range(num_variables):
            expression.add_linear(indices[vi], 0)

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
    def _ilinear_load(self, buff, Py_ssize_t num_variables, object dtype):
        expression = self.expression()

        if expression.num_variables() != num_variables:
            raise RuntimeError("num_variables must match expression.num_variables()")

        cdef const bias_type[:] ldata = np.frombuffer(buff[:dtype.itemsize*num_variables], dtype=dtype)
        for vi in range(num_variables):
            (<cppQuadraticModelBase[bias_type, index_type]*>expression).set_linear(vi, ldata[vi])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _iquadratic(self):
        expression = self.expression()

        # We choose the field names to mirror the quadratic iterator
        dtype = np.dtype([('u', self.index_dtype), ('v', self.index_dtype), ('bias', self.dtype)],
                         align=False)
        quadratic = np.empty(expression.num_interactions(), dtype)
        cdef index_type[:] irow = quadratic["u"]
        cdef index_type[:] icol = quadratic["v"]
        cdef bias_type[:] qdata = quadratic["bias"]

        cdef cppQuadraticModelBase[bias_type, index_type].const_quadratic_iterator2 it
        it = (<cppQuadraticModelBase[bias_type, index_type]*>expression).cbegin_quadratic()
        for vi in range(qdata.shape[0]):
            irow[vi] = deref(it).u
            icol[vi] = deref(it).v
            qdata[vi] = deref(it).bias

            inc(it)

        return quadratic

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _iquadratic_load(self, buff, Py_ssize_t num_interactions):
        # Note: it's possible to get segfaults by using this method directly.
        # We could add more safety, but for now I think marking it private is good
        # enough.
        expression = self.expression()

        if not expression.is_linear():
            raise RuntimeError("quadratic biases can only be loaded into a linear model")

        dtype = np.dtype([('u', self.index_dtype), ('v', self.index_dtype), ('bias', self.dtype)],
                         align=False)
        quadratic = np.frombuffer(buff[:dtype.itemsize*num_interactions], dtype=dtype)
        cdef const index_type[:] irow = quadratic["u"]
        cdef const index_type[:] icol = quadratic["v"]
        cdef const bias_type[:] qdata = quadratic["bias"]

        for i in range(num_interactions):
            # we're traversing the lower triangle, so it's OK to use add_quadratic_back
            (<cppQuadraticModelBase[bias_type, index_type]*>expression).add_quadratic_back(
                irow[i], icol[i], qdata[i])

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

    def remove_variable(self, v):
        cdef Py_ssize_t vi = self.parent.variables.index(v)
        self.expression().remove_variable(vi)

    def set_linear(self, v, bias_type bias):
        cdef Py_ssize_t vi = self.parent.variables.index(v)
        self.expression().set_linear(vi, bias)

    def set_quadratic(self, u, v, bias):
        raise NotImplementedError

    def _into_file(self, fp):
        """Copy the constraint data into the given file-like.

        See ConstrainedQuadraticModel.to_file() for a description of the file format.
        """
        # This is intended to look like a cut-down quadratic model

        from dimod.constrained.constrained import CQM_SERIALIZATION_VERSION

        data = dict(shape=self.shape,
                    dtype=self.dtype.name,
                    itype=self.index_dtype.name,
                    type=type(self).__name__,
                    )

        write_header(fp, EXPRESSION_MAGIC_PREFIX, data, version=CQM_SERIALIZATION_VERSION)

        # the indices of each variable in the parent model
        fp.write(IndicesSection(self).dumps())

        # offset
        fp.write(OffsetSection(self).dumps())

        # linear
        fp.write(LinearSection(self).dumps())

        fp.write(QuadraticSection(self).dumps())

    def _from_file(self, fp):
        expr = self.expression()

        if expr.num_variables():
            raise RuntimeError("._from_file() can only be called on an empty expression")

        if isinstance(fp, (bytes, bytearray, memoryview)):
            file_like = _BytesIO(fp)
        else:
            file_like = fp

        header_info = read_header(file_like, EXPRESSION_MAGIC_PREFIX)

        # we don't bother checking the header version under the assumption that
        # this method is called from CQM.from_file

        cdef Py_ssize_t num_variables = header_info.data['shape'][0]
        cdef Py_ssize_t num_interactions = header_info.data['shape'][1]
        dtype = np.dtype(header_info.data['dtype'])
        itype = np.dtype(header_info.data['itype'])

        # variable indices
        self._iindices_load(IndicesSection.load(file_like),
                            dtype=self.index_dtype, num_variables=num_variables)

        # offset
        self.offset += OffsetSection.load(file_like, dtype=dtype)

        # linear
        self._ilinear_load(LinearSection.load(file_like, dtype=dtype, num_variables=num_variables),
                           dtype=dtype, num_variables=num_variables)

        # quadratic
        self._iquadratic_load(QuadraticSection.load(file_like), num_interactions=num_interactions)

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

    def is_onehot(self):
        return self.constraint().is_onehot()

    def is_soft(self):
        return self.constraint().is_soft()

    def mark_discrete(self, bint marker = True):
        self.constraint().mark_discrete(marker)

    def penalty(self):
        """The penalty type for a soft constraint.

        Returns:
            If the constraint is soft, returns one of ``'linear'`` or ``'quadratic'``.
            If the constraint is hard, returns ``None``.

        """
        constraint = self.constraint()
        penalty = constraint.penalty()

        if not constraint.is_soft():
            return None
        elif penalty == cppPenalty.LINEAR:
            return "linear"
        elif penalty == cppPenalty.QUADRATIC:
            return "quadratic"
        elif penalty == cppPenalty.CONSTANT:
            # user should never see this, but might as well be future proof
            return "constant"
        else:
            raise RuntimeError("unexpected penalty")

    def set_weight(self, weight, penalty='linear'):
        """Set the weight of the constraint.

        Args:
            weight: Weight for a soft constraint.
                Must be a positive number. If ``None`` or
                ``float('inf')``, the constraint is hard.
                In feasible solutions, all the model's hard constraints
                must be met, while soft constraints might be violated to achieve
                overall good solutions.

            penalty: Penalty type for a soft constraint (a constraint with its
                ``weight`` parameter set). Supported values are ``'linear'`` and
                ``'quadratic'``.
                ``'quadratic'`` is supported for a constraint with binary
                variables only.

        """
        cdef bias_type _weight = float('inf') if weight is None else weight

        if _weight <= 0:
            raise ValueError("weight must be a positive number or None")

        cdef cppConstraint[bias_type, index_type]* constraint = self.constraint()

        cdef cppPenalty _penalty
        if penalty == 'linear':
           _penalty = cppPenalty.LINEAR
        elif penalty == 'quadratic':
            for i in range(constraint.num_variables()):
                vartype = self.parent.cppcqm.vartype(constraint.variables()[i])
                if vartype not in (cppVartype.BINARY, cppVartype.SPIN):
                    raise ValueError("quadratic penalty only allowed if the constraint has binary variables")
            _penalty = cppPenalty.QUADRATIC
        elif penalty == 'constant':
            raise ValueError('penalty should be "linear" or "quadratic"')
        else:
            raise ValueError('penalty should be "linear" or "quadratic"')

        constraint.set_weight(_weight)
        constraint.set_penalty(_penalty)

    def weight(self):
        """The weight of the constraint.

        If the constraint is hard, will be ``float('inf')``.
        """
        return self.constraint().weight()
