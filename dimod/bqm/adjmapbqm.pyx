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

from numbers import Integral

cimport cython

import numpy as np

from dimod.bqm.cppbqm cimport num_variables, num_interactions
from dimod.bqm.cppbqm cimport add_variable, add_interaction, pop_variable
from dimod.bqm.cppbqm cimport  get_linear, set_linear
from dimod.bqm.cppbqm cimport get_quadratic, set_quadratic
# from dimod.bqm.adjarraybqm cimport AdjArrayBQM



cdef class AdjMapBQM:
    """
    """

    def __cinit__(self, *args, **kwargs):
        # Developer note: if VarIndex or Bias were fused types, we would want
        # to do a type check here but since they are fixed...
        self.dtype = np.dtype(np.double)
        self.index_dtype = np.dtype(np.uintc)


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
            raise NotImplementedError  # update docstring
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

    def __len__(self):
        return self.num_variables

    @property
    def num_variables(self):
        return num_variables(self.adj_)

    @property
    def num_interactions(self):
        return num_interactions(self.adj_)

    @property
    def shape(self):
        return self.num_variables, self.num_interactions

    def add_variable(self):
        return add_variable(self.adj_)

    def pop_variable(self):
        return pop_variable(self.adj_)

    def get_linear(self, object v):
        if v < 0 or v >= self.num_variables:
            raise ValueError
        cdef VarIndex var = v
        return get_linear(self.adj_, var)

    def get_quadratic(self, VarIndex u, VarIndex v):
        return get_quadratic(self.adj_, u, v)

    def set_linear(self, VarIndex v, Bias b):
        set_linear(self.adj_, v, b)

    def set_quadratic(self, VarIndex u, VarIndex v, Bias b):
        set_quadratic(self.adj_, u, v, b)

    # derived methods

    def append_linear(self, Bias b):
        set_linear(self.adj_, add_variable(self.adj_), b)

    def pop_linear(self):
        self.pop_variable()

    def to_lists(self, object sort_and_reduce=True):
        """Dump to a list of lists, mostly for testing"""
        # todo: use functions
        return list((list(neighbourhood.items()), bias)
                    for neighbourhood, bias in self.adj_)

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
