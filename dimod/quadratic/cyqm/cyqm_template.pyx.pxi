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
from libc.string cimport memcpy
from libcpp.vector cimport vector

import dimod

from dimod.binary.cybqm cimport cyBQM
from dimod.cyutilities cimport as_numpy_float, cppvartype
from dimod.libcpp.vartypes cimport vartype_info as cppvartype_info
from dimod.quadratic cimport cyQM
from dimod.sampleset import as_samples
from dimod.typing cimport Integer, Numeric, int8_t
from dimod.variables import Variables
from dimod.vartypes import as_vartype, Vartype


ctypedef fused cyBQM_and_QM:
    cyBQM
    cyQM


cdef class cyQM_template(cyQMBase):
    def __cinit__(self):
        self.cppqm = self.base = new cppQuadraticModel[bias_type, index_type]()

    def __dealloc__(self):
        if self.cppqm is not NULL:
            del self.cppqm

    def __init__(self):
        super().__init__()
        self.REAL_INTERACTIONS = dimod.REAL_INTERACTIONS

    def __deepcopy__(self, memo):
        cdef cyQM_template new = type(self)()
        new.cppqm[0] = self.cppqm[0]  # *cppqm = *cppqm
        new.variables = deepcopy(self.variables, memo)

        new.REAL_INTERACTIONS = self.REAL_INTERACTIONS

        memo[id(self)] = new
        return new

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _ilower_triangle_load(self, Py_ssize_t vi, Py_ssize_t num_neighbors, const unsigned char[:] buff):
        cdef Py_ssize_t index_itemsize = self.index_dtype.itemsize
        cdef Py_ssize_t bias_itemsize = self.dtype.itemsize
        cdef Py_ssize_t itemsize = index_itemsize + bias_itemsize

        if num_neighbors*itemsize > buff.size:
            raise RuntimeError

        cdef Py_ssize_t i
        cdef index_type ui
        cdef bias_type bias
        for i in range(num_neighbors):
            memcpy(&ui, &buff[i*itemsize], index_itemsize)
            memcpy(&bias, &buff[i*itemsize+index_itemsize], bias_itemsize)

            self.cppqm.add_quadratic_back(ui, vi, bias)

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
        cdef const int8_t[:] vartype_view = arr['vartype']
        cdef const bias_type[:] lb_view = arr['lb']
        cdef const bias_type[:] ub_view = arr['ub']

        cdef Py_ssize_t vi
        cdef cppVartype cpp_vartype
        for vi in range(num_variables):
            self.cppqm.add_variable(<cppVartype>(vartype_view[vi]), lb_view[vi], ub_view[vi])

        while self.variables.size() < self.cppqm.num_variables():
            self.variables._append()

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

        self.cppqm.add_linear(vi, bias)

    def add_linear_from_array(self, const Numeric[:] linear):
        cdef Py_ssize_t length = linear.shape[0]
        cdef Py_ssize_t vi

        if self.variables._is_range():
            if length > self.num_variables():
                raise ValueError("variables must already exist")
            for vi in range(length):
                self.cppqm.add_linear(vi, linear[vi])
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
                                  const Integer[::1] irow, const Integer[::1] icol,
                                  const Numeric[::1] qdata):
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
        return self.cppqm

    @classmethod
    def from_cybqm(cls, cyBQM bqm):

        cdef cyQM_template qm = cls()

        qm.offset = bqm.offset

        # linear
        cdef Py_ssize_t vi
        cdef cppVartype vartype = bqm.cppbqm.vartype()
        for vi in range(bqm.num_variables()):
            qm.cppqm.add_variable(vartype)
            qm.cppqm.set_linear(vi, bqm.cppbqm.linear(vi))
        qm.variables._extend(bqm.variables)

        # quadratic
        it = bqm.cppbqm.cbegin_quadratic()
        while it != bqm.cppbqm.cend_quadratic():
            qm.cppqm.set_quadratic(deref(it).u, deref(it).v, deref(it).bias)
            inc(it)

        return qm

    def set_linear(self, v, bias_type bias):
        cdef Py_ssize_t vi = self.variables.index(v)
        self.cppqm.set_linear(vi, bias)

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

        self.cppqm.set_lower_bound(vi, lb)

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

        self.cppqm.set_upper_bound(vi, ub)

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
            self.cppqm.add_linear(mapping[vi], other.data().linear(vi))

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
        self.cppqm.add_offset(other.data().offset())
