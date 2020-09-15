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

from dimod.bqm.cppbqm cimport AdjVectorBQM as cppAdjVectorBQM


ctypedef np.float64_t Bias
ctypedef np.int64_t CaseIndex
ctypedef np.int64_t VarIndex


cdef class cyDiscreteQuadraticModel:
    cdef cppAdjVectorBQM[CaseIndex, Bias] bqm_
    cdef vector[CaseIndex] case_starts_  # len(adj_) + 1
    cdef vector[vector[VarIndex]] adj_

    cdef readonly object dtype

    cpdef Py_ssize_t add_variable(self, Py_ssize_t) except -1
    cpdef Bias get_linear_case(self, VarIndex, CaseIndex) except? -45.3
    cpdef Py_ssize_t num_cases(self, Py_ssize_t v=*) except -1
    cpdef Py_ssize_t num_variables(self)
    cpdef Py_ssize_t set_linear(self, VarIndex, Bias[:]) except -1
    cpdef Py_ssize_t set_linear_case(self, VarIndex, CaseIndex, Bias) except -1
    cpdef Py_ssize_t set_quadratic_case(
        self, VarIndex, CaseIndex, VarIndex, CaseIndex, Bias) except -1
    cpdef Bias get_quadratic_case(
        self, VarIndex, CaseIndex, VarIndex, CaseIndex)  except? -45.3
