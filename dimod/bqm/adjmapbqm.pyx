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

from dimod.bqm.cppbqm cimport (num_variables, num_interactions,
                               add_variable, add_interaction,
                               pop_variable, remove_interaction,
                               get_linear, set_linear,
                               get_quadratic, set_quadratic)
# from dimod.bqm.adjarraybqm cimport AdjArrayBQM



cdef class AdjMapBQM:
    """
    """

    def __cinit__(self, *args, **kwargs):
        # Developer note: if VarIndex or Bias were fused types, we would want
        # to do a type check here but since they are fixed...
        self.dtype = np.dtype(np.double)
        self.index_dtype = np.dtype(np.uintc)

        # otherwise these would be None
        self.label_to_idx = dict()
        self.idx_to_label = dict()


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, object arg1=0):

        cdef Bias [:, :] D  # in case it's dense
        cdef size_t num_variables, i
        cdef VarIndex u, v
        cdef Bias b

        if isinstance(arg1, Integral):
            if arg1 < 0:
                raise ValueError
            num_variables = arg1
            # we could do this in bulk with a resize but let's try using the
            # functions instead
            for i in range(num_variables):
                add_variable(self.adj_)
        elif isinstance(arg1, tuple):
            if len(arg1) == 2:
                linear, quadratic = arg1
            else:
                raise ValueError()

            if isinstance(linear, abc.Mapping):
                for var, b in linear.items():
                    self.set_linear(var, b)
            else:
                raise NotImplementedError
            
            if isinstance(quadratic, abc.Mapping):
                for uvar, var, b in quadratic.items():
                    self.set_quadratic(uvar, var, b)
            else:
                raise NotImplementedError

        elif hasattr(arg1, "to_adjvector"):
            # we might want a more generic is_bqm function or similar
            raise NotImplementedError  # update docstring
        else:
            # assume it's dense
            D = np.atleast_2d(np.asarray(arg1, dtype=self.dtype))

            num_variables = D.shape[0]

            if D.ndim != 2 or num_variables != D.shape[1]:
                raise ValueError("expected dense to be a 2 dim square array")

            # we could do this in bulk with a resize but let's try using the
            # functions instead
            for i in range(num_variables):
                add_variable(self.adj_)

            for u in range(num_variables):
                for v in range(num_variables):
                    b = D[u, v]

                    if u == v and b != 0:
                        set_linear(self.adj_, u, b)
                    elif b != 0:  # ignore the 0 off-diagonal
                        set_quadratic(self.adj_, u, v, b)

    @property
    def num_variables(self):
        return num_variables(self.adj_)

    @property
    def num_interactions(self):
        return num_interactions(self.adj_)

    @property
    def shape(self):
        return self.num_variables, self.num_interactions

    def add_variable(self, object label=None):

        if label is None:
            # if nothing has been overwritten we can go ahead and exit here
            # this is not necessary but good for performance
            if not self.label_to_idx:
                return add_variable(self.adj_)

            label = num_variables(self.adj_)


        # don't create variables that already exist
        if self.has_variable(label):
            return label

        cdef object v = add_variable(self.adj_)

        if v != label:
            self.label_to_idx[label] = v
            self.idx_to_label[v] = label

        return label

    def has_variable(self, object v):
        if v in self.label_to_idx:
            return True
        return (v in range(num_variables(self.adj_)) and  # handles non-ints
                v not in self.idx_to_label)  # not overwritten

    def iter_variables(self):
        cdef object v
        for v in range(num_variables(self.adj_)):
            yield self.idx_to_label.get(v, v)

    def pop_variable(self):
        if num_variables(self.adj_) == 0:
            raise ValueError("pop from empty binary quadratic model")

        cdef object v = pop_variable(self.adj_)  # cast to python object

        # delete any relevant labels if present
        if self.label_to_idx:
            v = self.idx_to_label.pop(v, v)
            self.label_to_idx.pop(v, None)

        return v

    def get_linear(self, object v):
        if not self.has_variable(v):
            raise ValueError('{} is not a variable'.format(v))
        cdef VarIndex vi = self.label_to_idx.get(v, v)
        return get_linear(self.adj_, vi)

    def get_quadratic(self, object u, object v):
        # todo: return default?

        if not self.has_variable(u):
            raise ValueError('{} is not a variable'.format(u))
        if not self.has_variable(v):
            raise ValueError('{} is not a variable'.format(v))

        cdef VarIndex ui = self.label_to_idx.get(u, u)
        cdef VarIndex vi = self.label_to_idx.get(v, v)
        cdef pair[Bias, bool] out = get_quadratic(self.adj_, ui, vi)

        if not out.second:
            raise ValueError('No interaction between {} and {}'.format(u, v))

        return out.first

    def remove_interaction(self, object u, object v):
        if not self.has_variable(u):
            raise ValueError('{} is not a variable'.format(u))
        if not self.has_variable(v):
            raise ValueError('{} is not a variable'.format(v))

        cdef VarIndex ui = self.label_to_idx.get(u, u)
        cdef VarIndex vi = self.label_to_idx.get(v, v)

        return remove_interaction(self.adj_, ui, vi)

    def set_linear(self, object v, Bias b):

        self.add_variable(v)  # add if it doesn't exist

        cdef VarIndex vi = self.label_to_idx.get(v, v)

        set_linear(self.adj_, vi, b)

    def set_quadratic(self, object u, object v, Bias b):
        # add if they don't already exist
        self.add_variable(u)
        self.add_variable(v)

        cdef VarIndex ui = self.label_to_idx.get(u, u)
        cdef VarIndex vi = self.label_to_idx.get(v, v)

        set_quadratic(self.adj_, ui, vi, b)

    # def to_lists(self, object sort_and_reduce=True):
    #     """Dump to a list of lists, mostly for testing"""
    #     # todo: use functions
    #     return list((list(neighbourhood.items()), bias)
    #                 for neighbourhood, bias in self.adj_)

    # def to_adjarray(self):
    #     # this is always a copy

    #     # make a 0-length BQM but then manually resize it, note that this
    #     # treats them as vectors
    #     cdef AdjArrayBQM bqm = AdjArrayBQM()  # empty
    #     bqm.invars_.resize(self.adj_.size())
    #     bqm.outvars_.resize(2*self.num_interactions)

    #     cdef pair[VarIndex, Bias] outvar
    #     cdef VarIndex u
    #     cdef size_t outvar_idx = 0
    #     for u in range(self.adj_.size()):
            
    #         # set the linear bias
    #         bqm.invars_[u].second = self.adj_[u].second
    #         bqm.invars_[u].first = outvar_idx

    #         # set the quadratic biases
    #         for outvar in self.adj_[u].first:
    #             bqm.outvars_[outvar_idx] = outvar
    #             outvar_idx += 1

    #     return bqm
