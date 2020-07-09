# distutils: language = c++
# cython: language_level=3
#
# Copyright 2020 D-Wave Systems Inc.
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

cimport cython
import numpy as np

from dimod.bqm import AdjVectorBQM
from dimod.bqm cimport cyAdjVectorBQM
from dimod.bqm.common import dtype
from dimod.generators.constraints import combinations

__all__ = ['DiscreteQuadraticModel', 'DQM']


cdef class cyDiscreteQuadraticModel:

    @property
    def num_variables(self):
        return self.variables_.size()


    def num_cases(self, v=None):
        """The total number of cases."""
        if v is None:
            return self.bqm_.num_variables()

        cdef VarIndex vi = v  # todo: handle non-index

        if vi < 0 or vi >= self.variables_.size():
            raise ValueError("variable out of range")

        return self.variables_[vi].stop - self.variables_[vi].start


    def add_variable(self, Py_ssize_t num_cases):

        cdef Py_ssize_t start = self.bqm_.num_variables()

        for _ in range(num_cases):
            self.bqm_.add_variable()

        cdef Py_ssize_t stop = self.bqm_.num_variables()

        self.variables_.push_back(Range(start, stop))

        return self.variables_.size() - 1  # variable label


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_interaction(self, VarIndex u, VarIndex v, object biases):

        cdef Range u_range = self.variables_[u]
        cdef Range v_range = self.variables_[v]

        # do some type coercion/checking
        # todo: raise some better error messages
        # todo: handle dicts for sparse
        biases = np.asarray(biases, dtype=dtype, order='C')
        biases = biases.reshape((u_range.stop - u_range.start,
                                 v_range.stop - v_range.start))

        cdef Bias[:, :] biases_view = biases

        cdef VarIndex uc, vc
        cdef Py_ssize_t ui, vi
        cdef Bias bias
        ui = 0
        for uc in range(u_range.start, u_range.stop):
            vi = 0
            for vc in range(v_range.start, v_range.stop):
                bias = biases_view[ui, vi]

                if bias > 0:
                    # this could be sped up since we're doing a new binary
                    # search each time
                    self.bqm_.set_quadratic(uc, vc, bias)

                vi += 1
            ui += 1

    def to_bqm(self, strength):
        # todo: automatic penalty strength
        
        cdef cyAdjVectorBQM bqm = AdjVectorBQM('BINARY')

        # copy the cases
        bqm.bqm_ = self.bqm_

        # add the 1-in-N constraints. This is handled in python-space at the
        # moment, but we could speed it up if necessary
        cdef Py_ssize_t vi
        for vi in range(self.variables_.size()):
            labels = range(self.variables_[vi].start, self.variables_[vi].stop)
            bqm.update(combinations(labels, 1, strength))

        # relabel
        cdef Py_ssize_t vc
        mapping = {}
        for vi in range(self.variables_.size()):
            c = 0
            for vc in range(self.variables_[vi].start, self.variables_[vi].stop):
                mapping[vc] = (vi, c)
                c += 1

        bqm.relabel_variables(mapping)

        return bqm


class DiscreteQuadraticModel(cyDiscreteQuadraticModel):
    __doc__ = cyDiscreteQuadraticModel.__doc__


DQM = DiscreteQuadraticModel  # alias
