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

from libcpp.vector cimport vector

cimport numpy as np

from dimod.discrete.cppdqm cimport AdjVectorDQM as cppAdjVectorDQM
from dimod.bqm.common cimport Integral32plus, Numeric, Numeric32plus

ctypedef np.float64_t Bias_t
ctypedef np.int64_t VarIndex_t

ctypedef fused Unsigned:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t


cdef class cyDiscreteQuadraticModel:
    cdef cppAdjVectorDQM[VarIndex_t, Bias_t] dqm_

    cdef readonly object dtype
    cdef readonly object variable_dtype

    cpdef Py_ssize_t add_variable(self, Py_ssize_t) except -1
    cpdef Bias_t[:] energies(self, VarIndex_t[:, :])
    cpdef Bias_t get_linear_case(self, VarIndex_t, VarIndex_t) except? -45.3
    cpdef Py_ssize_t num_cases(self, Py_ssize_t v=*) except -1
    cpdef Py_ssize_t num_case_interactions(self)
    cpdef Py_ssize_t num_variable_interactions(self) except -1
    cpdef Py_ssize_t num_variables(self)
    cpdef Py_ssize_t set_linear(self, VarIndex_t v, Numeric[:] biases) except -1
    cpdef Py_ssize_t set_linear_case(self, VarIndex_t, VarIndex_t, Bias_t) except -1
    cpdef Py_ssize_t set_quadratic_case(
        self, VarIndex_t, VarIndex_t, VarIndex_t, VarIndex_t, Bias_t) except -1
    cpdef Bias_t get_quadratic_case(
        self, VarIndex_t, VarIndex_t, VarIndex_t, VarIndex_t)  except? -45.3

    cdef void _into_numpy_vectors(self, Unsigned[:] starts, Bias_t[:] ldata,
        Unsigned[:] irow, Unsigned[:] icol, Bias_t[:] qdata)
