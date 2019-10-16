# distutils: language = c++
# cython: language_level=3
#
# Copyright 2019 D-Wave Systems Inc.
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
#
# =============================================================================

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from numbers import Integral

import numpy as np

from dimod.bqm.cppbqm cimport (num_variables, num_interactions,
                               get_linear, get_quadratic,
                               set_linear, set_quadratic,
                               add_variable, add_interaction,
                               pop_variable, remove_interaction)
from dimod.bqm.utils cimport as_numpy_scalar
from dimod.core.bqm import ShapeableBQM
from dimod.vartypes import as_vartype


cdef class cyAdjVectorBQM:
    """
    """

    def __cinit__(self, *args, **kwargs):
        # Developer note: if VarIndex or Bias were fused types, we would want
        # to do a type check here but since they are fixed...
        self.dtype = np.dtype(np.double)
        self.index_dtype = np.dtype(np.uintc)

        # otherwise these would be None
        self._label_to_idx = dict()
        self._idx_to_label = dict()

    def __init__(self, object obj=0, object vartype=None):

        # handle the case where only vartype is given
        if vartype is None:
            try:
                vartype = obj.vartype
            except AttributeError:
                vartype = obj
                obj = 0
        self.vartype = as_vartype(vartype)

        cdef Bias [:, :] D  # in case it's dense
        cdef size_t num_variables, i
        cdef VarIndex u, v
        cdef Bias b
        cdef VarIndex ui, vi

        if isinstance(obj, Integral):
            if obj < 0:
                raise ValueError
            num_variables = obj
            # we could do this in bulk with a resize but let's try using the
            # functions instead
            for i in range(num_variables):
                add_variable(self.adj_)
        elif isinstance(obj, tuple):
            if len(obj) == 2:
                linear, quadratic = obj
            elif len(obj) == 3:
                linear, quadratic, self.offset_ = obj
            else:
                raise ValueError()

            if isinstance(linear, abc.Mapping):
                for var, b in linear.items():
                    self.set_linear(var, b)
            else:
                raise NotImplementedError
            
            if isinstance(quadratic, abc.Mapping):
                for (uvar, var), b in quadratic.items():
                    self.set_quadratic(uvar, var, b)
            else:
                raise NotImplementedError
        else:
            # assume it's dense
            D = np.atleast_2d(np.asarray(obj, dtype=self.dtype))

            num_variables = D.shape[0]

            if D.ndim != 2 or num_variables != D.shape[1]:
                raise ValueError("expected dense to be a 2 dim square array")

            # so we only need to copy once if realloc
            self.adj_.resize(num_variables) 

            for ui in range(num_variables):
                set_linear(self.adj_, ui, D[ui, ui])

                for vi in range(ui + 1, num_variables):
                    b = D[ui, vi] + D[vi, ui]  # add upper and lower

                    if b == 0:  # ignore 0 off-diagonal
                        continue

                    # we know that they are ordered, so we can go ahead and
                    # append directly onto the vector rather than doing an
                    # insert from the binary search
                    self.adj_[ui].first.push_back(pair[VarIndex, Bias](vi, b))
                    self.adj_[vi].first.push_back(pair[VarIndex, Bias](ui, b))


    # dev note: almost all of these are identical to AdjMapBQM. Unfortunately
    # we can't really make an abstract base class in cython that will handle the
    # template functions the way that we want.

    @property
    def num_variables(self):
        return num_variables(self.adj_)

    @property
    def num_interactions(self):
        return num_interactions(self.adj_)

    @property
    def offset(self):
        return as_numpy_scalar(self.offset_, self.dtype)

    @offset.setter
    def offset(self, Bias offset):
        self.offset_ = offset

    cdef VarIndex label_to_idx(self, object v) except *:
        """Get the index in the underlying array from the python label."""
        cdef VarIndex vi

        try:
            if not self._label_to_idx:
                # there are no arbitrary labels so v must be integer labelled
                vi = v
            elif v in self._idx_to_label:
                # v is an integer label that has been overwritten
                vi = self._label_to_idx[v]
            else:
                vi = self._label_to_idx.get(v, v)
        except (OverflowError, TypeError, KeyError) as ex:
            raise ValueError("{} is not a variable".format(v)) from ex

        if vi < 0 or vi >= num_variables(self.adj_):
            raise ValueError("{} is not a variable".format(v))

        return vi

    def add_variable(self, object label=None):
        if label is None:
            # if nothing has been overwritten we can go ahead and exit here
            # this is not necessary but good for performance
            if not self._label_to_idx:
                return add_variable(self.adj_)

            label = num_variables(self.adj_)

            if self.has_variable(label):
                # if the integer label already is taken, there must be a missing
                # smaller integer we can use
                for v in range(label):
                    if not self.has_variable(v):
                        break
                label = v

        else:
            try:
                self.label_to_idx(label)
            except ValueError:
                pass
            else:
                # it exists already
                return label

        cdef object vi = add_variable(self.adj_)
        if vi != label:
            self._label_to_idx[label] = vi
            self._idx_to_label[vi] = label

        return label

    def iter_linear(self):
        # this should really be done with an iterator, but we don't have them
        # yet so we'll do it manually
        cdef VarIndex vi
        cdef object v
        cdef Bias b

        for vi in range(num_variables(self.adj_)):
            b = self.adj_[vi].second
            v = self._idx_to_label.get(vi, vi)
            yield v, as_numpy_scalar(b, self.dtype)

    def iter_quadratic(self, object variables=None):
        # this would be much easier if interaction_iterator was implemented, but
        # we'll do this by-hand for now
        cdef VarIndex ui, vi  # c indices
        cdef object u, v  # python labels
        cdef Bias b

        cdef pair[VarIndex, Bias] vi_b

        if variables is None:
            # in the case that variables is unlabelled we can speed things up
            # by just walking through the range
            for ui in range(num_variables(self.adj_)):
                u = self._idx_to_label.get(ui, ui)

                # we could find the index of the first vi > ui, but this is
                # simpler for now and make for more generic code between
                # vector/map implementations
                for vi_b in self.adj_[ui].first:
                    vi = vi_b.first
                    b = vi_b.second

                    if vi < ui:
                        continue

                    v = self._idx_to_label.get(vi, vi)

                    yield u, v, as_numpy_scalar(b, self.dtype)
        elif self.has_variable(variables):
            yield from self.iter_quadratic([variables])
        else:
            seen = set()
            for u in variables:
                ui = self.label_to_idx(u)
                seen.add(u)

                for vi_b in self.adj_[ui].first:
                    vi = vi_b.first
                    b = vi_b.second

                    v = self._idx_to_label.get(vi, vi)

                    if v in seen:
                        continue

                    yield u, v, as_numpy_scalar(b, self.dtype)

    def pop_variable(self):
        if num_variables(self.adj_) == 0:
            raise ValueError("pop from empty binary quadratic model")

        cdef object v = pop_variable(self.adj_)  # cast to python object

        # delete any relevant labels if present
        if self._label_to_idx:
            v = self._idx_to_label.pop(v, v)
            self._label_to_idx.pop(v, None)

        return v

    def get_linear(self, object v):
        return as_numpy_scalar(get_linear(self.adj_, self.label_to_idx(v)),
                               self.dtype)

    def get_quadratic(self, object u, object v):
        if u == v:
            raise ValueError("No interaction between {} and itself".format(u))

        cdef VarIndex ui = self.label_to_idx(u)
        cdef VarIndex vi = self.label_to_idx(v)
        cdef pair[Bias, bool] out = get_quadratic(self.adj_, ui, vi)

        if not out.second:
            raise ValueError('No interaction between {} and {}'.format(u, v))

        return as_numpy_scalar(out.first, self.dtype)

    def remove_interaction(self, object u, object v):
        if u == v:
            # developer note: maybe we should raise a ValueError instead?
            return False

        cdef VarIndex ui = self.label_to_idx(u)
        cdef VarIndex vi = self.label_to_idx(v)
        cdef bool removed = remove_interaction(self.adj_, ui, vi)

        return removed

    def set_linear(self, object v, Bias b):
        cdef VarIndex vi

        # this try-catch it not necessary but it speeds things up in the case
        # that the variable already exists which is the typical case
        try:
            vi = self.label_to_idx(v)
        except ValueError:
            vi = self.label_to_idx(self.add_variable(v))

        set_linear(self.adj_, vi, b)

    def set_quadratic(self, object u, object v, Bias b):
        cdef VarIndex ui, vi

        # these try-catchs are not necessary but it speeds things up in the case
        # that the variables already exists which is the typical case
        try:
            ui = self.label_to_idx(u)
        except ValueError:
            ui = self.label_to_idx(self.add_variable(u))
        try:
            vi = self.label_to_idx(v)
        except ValueError:
            vi = self.label_to_idx(self.add_variable(v))

        set_quadratic(self.adj_, ui, vi, b)


class AdjVectorBQM(cyAdjVectorBQM, ShapeableBQM):
    __doc__ = cyAdjVectorBQM.__doc__
