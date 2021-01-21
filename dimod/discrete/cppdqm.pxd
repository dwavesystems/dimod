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

from dimod.bqm.cppbqm cimport AdjVectorBQM as cppAdjVectorBQM
from dimod.bqm.common cimport VarIndex, Bias
from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector

cimport numpy as np


cdef extern from "dimod/adjvectordqm.h" namespace "dimod" nogil:

    cdef cppclass AdjVectorDQM[V, B]:
        ctypedef V variable_type
        ctypedef B bias_type
        ctypedef size_t size_type

        cppAdjVectorBQM[variable_type, bias_type] bqm_
        vector[variable_type] case_starts_
        vector[vector[variable_type]] adj_

        # constructors
        # cython cannot handle templated constructors, so we call out the types
        # explicitly

        AdjVectorDQM() except +

        # the actual signature is more general, but we already have a large
        # number of these so we'll add them as needed
        # AdjVectorDQM(variable_type* case_starts, size_type num_variables, bias_type* linear_biases,
        #           size_type num_cases, variable_type* irow, variable_type* icol,
        #           bias_type* quadratic_biases, size_type num_interactions)
        AdjVectorDQM(np.uint32_t*, size_type, np.uint32_t*,  size_type, np.uint32_t*, np.uint32_t*, np.uint32_t*,  size_type)
        AdjVectorDQM(np.uint32_t*, size_type, np.uint64_t*,  size_type, np.uint32_t*, np.uint32_t*, np.uint64_t*,  size_type)
        AdjVectorDQM(np.uint32_t*, size_type, np.int32_t*,   size_type, np.uint32_t*, np.uint32_t*, np.int32_t*,   size_type)
        AdjVectorDQM(np.uint32_t*, size_type, np.int64_t*,   size_type, np.uint32_t*, np.uint32_t*, np.int64_t*,   size_type)
        AdjVectorDQM(np.uint32_t*, size_type, np.float32_t*, size_type, np.uint32_t*, np.uint32_t*, np.float32_t*, size_type)
        AdjVectorDQM(np.uint32_t*, size_type, np.float64_t*, size_type, np.uint32_t*, np.uint32_t*, np.float64_t*, size_type)
        AdjVectorDQM(np.uint64_t*, size_type, np.uint32_t*,  size_type, np.uint64_t*, np.uint64_t*, np.uint32_t*,  size_type)
        AdjVectorDQM(np.uint64_t*, size_type, np.uint64_t*,  size_type, np.uint64_t*, np.uint64_t*, np.uint64_t*,  size_type)
        AdjVectorDQM(np.uint64_t*, size_type, np.int32_t*,   size_type, np.uint64_t*, np.uint64_t*, np.int32_t*,   size_type)
        AdjVectorDQM(np.uint64_t*, size_type, np.int64_t*,   size_type, np.uint64_t*, np.uint64_t*, np.int64_t*,   size_type)
        AdjVectorDQM(np.uint64_t*, size_type, np.float32_t*, size_type, np.uint64_t*, np.uint64_t*, np.float32_t*, size_type)
        AdjVectorDQM(np.uint64_t*, size_type, np.float64_t*, size_type, np.uint64_t*, np.uint64_t*, np.float64_t*, size_type)
        AdjVectorDQM(np.int32_t*,  size_type, np.uint32_t*,  size_type, np.int32_t*,  np.int32_t*,  np.uint32_t*,  size_type)
        AdjVectorDQM(np.int32_t*,  size_type, np.uint64_t*,  size_type, np.int32_t*,  np.int32_t*,  np.uint64_t*,  size_type)
        AdjVectorDQM(np.int32_t*,  size_type, np.int32_t*,   size_type, np.int32_t*,  np.int32_t*,  np.int32_t*,   size_type)
        AdjVectorDQM(np.int32_t*,  size_type, np.int64_t*,   size_type, np.int32_t*,  np.int32_t*,  np.int64_t*,   size_type)
        AdjVectorDQM(np.int32_t*,  size_type, np.float32_t*, size_type, np.int32_t*,  np.int32_t*,  np.float32_t*, size_type)
        AdjVectorDQM(np.int32_t*,  size_type, np.float64_t*, size_type, np.int32_t*,  np.int32_t*,  np.float64_t*, size_type)
        AdjVectorDQM(np.int64_t*,  size_type, np.uint32_t*,  size_type, np.int64_t*,  np.int64_t*,  np.uint32_t*,  size_type)
        AdjVectorDQM(np.int64_t*,  size_type, np.uint64_t*,  size_type, np.int64_t*,  np.int64_t*,  np.uint64_t*,  size_type)
        AdjVectorDQM(np.int64_t*,  size_type, np.int32_t*,   size_type, np.int64_t*,  np.int64_t*,  np.int32_t*,   size_type)
        AdjVectorDQM(np.int64_t*,  size_type, np.int64_t*,   size_type, np.int64_t*,  np.int64_t*,  np.int64_t*,   size_type)
        AdjVectorDQM(np.int64_t*,  size_type, np.float32_t*, size_type, np.int64_t*,  np.int64_t*,  np.float32_t*, size_type)
        AdjVectorDQM(np.int64_t*,  size_type, np.float64_t*, size_type, np.int64_t*,  np.int64_t*,  np.float64_t*, size_type)

        # methods

        bool self_loop_present() except +
        bool connection_present(variable_type, variable_type) except +
        size_type num_variables() except +
        size_type num_variable_interactions() except +
        size_type num_cases(variable_type) except +
        size_type num_case_interactions() except +
        bias_type get_linear_case(variable_type, variable_type) except +
        void set_linear_case(variable_type, variable_type, bias_type) except +
        void get_linear[io_bias_type](variable_type, io_bias_type*) except +
        void set_linear[io_bias_type](variable_type, io_bias_type*) except +
        pair[bias_type, bool] get_quadratic_case(variable_type, variable_type, variable_type, variable_type) except +
        bool set_quadratic_case(variable_type, variable_type, variable_type, variable_type, bias_type) except +
        bool get_quadratic[io_bias_type](variable_type, variable_type, io_bias_type*) except +
        bool set_quadratic[io_bias_type](variable_type, variable_type, io_bias_type*) except +
        void get_energies[io_variable_type, io_bias_type](io_variable_type*, int, variable_type, io_bias_type*) except +
        void extract_data[io_variable_type, io_bias_type](io_variable_type*, io_bias_type*, io_variable_type*, io_variable_type*, io_bias_type*) except +

        # shapeable methods

        variable_type add_variable(variable_type) except +
