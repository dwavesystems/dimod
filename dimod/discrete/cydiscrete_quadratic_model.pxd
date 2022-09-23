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

from dimod.libcpp.binary_quadratic_model cimport BinaryQuadraticModel as cppBinaryQuadraticModel

ctypedef np.float64_t bias_type
ctypedef np.int32_t index_type

ctypedef fused Unsigned:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t

ctypedef fused Integral32plus:
    np.uint32_t
    np.uint64_t
    np.int32_t
    np.int64_t

ctypedef fused Numeric:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t
    np.float32_t
    np.float64_t

ctypedef fused Numeric32plus:
    Integral32plus
    np.float32_t
    np.float64_t

cdef class cyDiscreteQuadraticModel:
    cdef cppBinaryQuadraticModel[bias_type, index_type] cppbqm
    cdef vector[index_type] case_starts_  # len(adj_) + 1
    cdef vector[vector[index_type]] adj_

    cdef readonly object dtype
    cdef readonly object case_dtype

    cpdef Py_ssize_t add_variable(self, Py_ssize_t) except -1
    cpdef bias_type[:] energies(self, index_type[:, :])
    cpdef bias_type get_linear_case(self, index_type, index_type) except? -45.3
    cpdef Py_ssize_t num_cases(self, Py_ssize_t v=*) except -1
    cpdef Py_ssize_t num_case_interactions(self)
    cpdef Py_ssize_t num_variable_interactions(self) except -1
    cpdef Py_ssize_t num_variables(self)
    cpdef Py_ssize_t set_linear(self, index_type v, Numeric[:] biases) except -1
    cpdef Py_ssize_t set_linear_case(self, index_type, index_type, bias_type) except -1
    cpdef Py_ssize_t set_quadratic_case(
        self, index_type, index_type, index_type, index_type, bias_type) except -1
    cpdef bias_type get_quadratic_case(
        self, index_type, index_type, index_type, index_type)  except? -45.3

    cdef void _into_numpy_vectors(self, Unsigned[:] starts, bias_type[:] ldata,
        Unsigned[:] irow, Unsigned[:] icol, bias_type[:] qdata)
