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

cimport cython

import numpy as np

from dimod.bqm.adjarraybqm cimport cyAdjArrayBQM
from dimod.bqm.cppbqm cimport (num_variables, num_interactions,
                               add_variable, add_interaction,
                               pop_variable, remove_interaction,
                               get_linear, set_linear,
                               get_quadratic, set_quadratic)
from dimod.bqm.utils cimport as_numpy_scalar
from dimod.core.bqm import ShapeableBQM
from dimod.vartypes import as_vartype


cdef class cyAdjMapBQM:
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


    @cython.boundscheck(False)
    @cython.wraparound(False)
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

        cdef map[VarIndex, Bias].iterator Nv_it, Nu_it  # neighbourhood iterators

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

        elif hasattr(obj, "to_adjvector"):
            # we might want a more generic is_bqm function or similar
            raise NotImplementedError  # update docstring
        else:
            # assume it's dense
            D = np.atleast_2d(np.asarray(obj, dtype=self.dtype))

            num_variables = D.shape[0]

            if D.ndim != 2 or num_variables != D.shape[1]:
                raise ValueError("expected dense to be a 2 dim square array")

            # so we only need to copy once if realloc
            self.adj_.resize(num_variables) 

            for vi in range(num_variables):
                set_linear(self.adj_, vi, D[vi, vi])

            for ui in range(num_variables):

                for vi in range(ui + 1, num_variables):
                    b = D[ui, vi] + D[vi, ui]  # add upper and lower

                    if b == 0:  # ignore 0 off-diagonal
                        continue

                    # we'd like to use set_quadratic(self.adj_, ui, vi, b) but
                    # because we know that we're always adding to the end
                    # of the map, we can provide a location hint to insert. We
                    # should really do this with an iterator  from c++ space,
                    # but those are not implemented yet
                    self.adj_[ui].first.insert(self.adj_[ui].first.end(),
                                               pair[VarIndex, Bias](vi, b))
                    self.adj_[vi].first.insert(self.adj_[vi].first.end(),
                                               pair[VarIndex, Bias](ui, b))


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
        """Add a variable to the binary quadratic model.

        Args:
            label (hashable, optional):
                A label for the variable. Defaults to the length of the binary
                quadratic model, if that label is available. Otherwise defaults
                to the lowest available positive integer label.

        Returns:
            hashable: The label of the added variable.

        Raises:
            TypeError: If the label is not hashable.

        Examples:

            >>> bqm = dimod.AdjMapBQM('SPIN')
            >>> bqm.add_variable()
            0
            >>> bqm.add_variable('a')
            'a'
            >>> bqm.add_variable()
            2

            >>> bqm = dimod.AdjMapBQM('SPIN')
            >>> bqm.add_variable(1)
            1
            >>> bqm.add_variable()  # 1 is taken
            0
            >>> bqm.add_variable()
            2

        """

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
        """Remove a variable from the binary quadratic model.

        Returns:
            hashable: The last variable added to the binary quadratic model.

        Raises:
            ValueError: If the binary quadratic model is empty.

        """
        if num_variables(self.adj_) == 0:
            raise ValueError("pop from empty binary quadratic model")

        cdef object v = pop_variable(self.adj_)  # cast to python object

        # delete any relevant labels if present
        if self._label_to_idx:
            v = self._idx_to_label.pop(v, v)
            self._label_to_idx.pop(v, None)

        return v

    def get_linear(self, object v):
        """Get the linear bias of v.

        Args:
            v (hashable):
                A variable in the binary quadratic model.

        Returns:
            float: The linear bias of v.

        Raises:
            ValueError: If v is not a variable in the binary quadratic model.

        """
        return as_numpy_scalar(get_linear(self.adj_, self.label_to_idx(v)),
                               self.dtype)

    def get_quadratic(self, object u, object v):
        """Get the quadratic bias of (u, v).

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

        Returns:
            float: The quadratic bias of (u, v).

        Raises:
            ValueError: If either u or v is not a variable in the binary
            quadratic model, if u == v or if (u, v) is not an interaction in
            the binary quadratic model.

        """
        if u == v:
            raise ValueError("No interaction between {} and itself".format(u))

        cdef VarIndex ui = self.label_to_idx(u)
        cdef VarIndex vi = self.label_to_idx(v)
        cdef pair[Bias, bool] out = get_quadratic(self.adj_, ui, vi)

        if not out.second:
            raise ValueError('No interaction between {} and {}'.format(u, v))

        return as_numpy_scalar(out.first, self.dtype)

    def remove_interaction(self, object u, object v):
        """Remove the interaction between variables u and v.

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

        Returns:
            bool: If there was an interaction to remove.

        Raises:
            ValueError: If either u or v is not a variable in the binary
            quadratic model.

        """
        if u == v:
            # developer note: maybe we should raise a ValueError instead?
            return False

        cdef VarIndex ui = self.label_to_idx(u)
        cdef VarIndex vi = self.label_to_idx(v)
        cdef bool removed = remove_interaction(self.adj_, ui, vi)

        return removed

    def set_linear(self, object v, Bias b):
        """Set the linear biase of a variable v.

        Args:
            v (hashable):
                A variable in the binary quadratic model. It is added if not
                already in the model.

            b (numeric):
                The linear bias of v.

        Raises:
            TypeError: If v is not hashable
        """
        cdef VarIndex vi

        # this try-catch it not necessary but it speeds things up in the case
        # that the variable already exists which is the typical case
        try:
            vi = self.label_to_idx(v)
        except ValueError:
            vi = self.label_to_idx(self.add_variable(v))

        set_linear(self.adj_, vi, b)

    def set_quadratic(self, object u, object v, Bias b):
        """Set the quadratic bias of (u, v).

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

            b (numeric):
                The linear bias of v.

        Raises:
            TypeError: If u or v is not hashable.

        """
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

    def to_adjarray(self):
        # this is always a copy

        # make a 0-length BQM but then manually resize it, note that this
        # treats them as vectors
        cdef cyAdjArrayBQM bqm = cyAdjArrayBQM(vartype=self.vartype)  # empty
        bqm.invars_.resize(self.adj_.size())
        bqm.outvars_.resize(2*self.num_interactions)

        cdef pair[VarIndex, Bias] outvar
        cdef VarIndex u
        cdef size_t outvar_idx = 0
        for u in range(self.adj_.size()):
            
            # set the linear bias
            bqm.invars_[u].second = self.adj_[u].second
            bqm.invars_[u].first = outvar_idx

            # set the quadratic biases
            for outvar in self.adj_[u].first:
                bqm.outvars_[outvar_idx] = outvar
                outvar_idx += 1

        bqm.offset_ = self.offset_

        # set up the variable labels
        bqm._label_to_idx.update(self._label_to_idx)
        bqm._idx_to_label.update(self._idx_to_label)

        return bqm

class AdjMapBQM(cyAdjMapBQM, ShapeableBQM):
    __doc__ = cyAdjMapBQM.__doc__
