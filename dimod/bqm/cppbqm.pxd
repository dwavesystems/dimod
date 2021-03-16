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

from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector

cimport numpy as np

from dimod.bqm.common cimport VarIndex, Bias

cdef extern from "dimod/adjvectorbqm.h" namespace "dimod" nogil:

    cdef cppclass AdjVectorBQM[V, B]:
        ctypedef V variable_type
        ctypedef B bias_type
        ctypedef size_t size_type

        vector[pair[vector[pair[variable_type, bias_type]], bias_type]] adj

        cppclass outvars_iterator:
            pair[variable_type, bias_type]& operator*()
            outvars_iterator operator++()
            outvars_iterator operator--()
            outvars_iterator operator+(size_type)
            outvars_iterator operator-(size_type)
            size_t operator-(outvars_iterator)
            bint operator==(outvars_iterator)
            bint operator!=(outvars_iterator)
            bint operator<(outvars_iterator)
            bint operator>(outvars_iterator)
            bint operator<=(outvars_iterator)
            bint operator>=(outvars_iterator)

        cppclass const_outvars_iterator:
            pair[variable_type, bias_type]& operator*()
            const_outvars_iterator operator++()
            const_outvars_iterator operator--()
            const_outvars_iterator operator+(size_type)
            const_outvars_iterator operator-(size_type)
            size_t operator-(const_outvars_iterator)
            bint operator==(const_outvars_iterator)
            bint operator!=(const_outvars_iterator)
            bint operator<(const_outvars_iterator)
            bint operator>(const_outvars_iterator)
            bint operator<=(const_outvars_iterator)
            bint operator>=(const_outvars_iterator)

        # constructors
        # cython cannot handle templated constructors, so we call out the types
        # explicitly

        AdjVectorBQM() except +
        AdjVectorBQM(const AdjVectorBQM&) except +
        AdjVectorBQM(const float[], size_type)
        AdjVectorBQM(const float[], size_type, bool)
        AdjVectorBQM(const double[], size_type)
        AdjVectorBQM(const double[], size_type, bool)

        # the actual signature is more general, but we already have a large
        # number of these so we'll add them as needed
        AdjVectorBQM(np.uint32_t*, np.uint32_t*, np.uint32_t*, size_type, bool)
        AdjVectorBQM(np.uint32_t*, np.uint32_t*, np.uint64_t*, size_type, bool)
        AdjVectorBQM(np.uint32_t*, np.uint32_t*, np.int32_t*, size_type, bool)
        AdjVectorBQM(np.uint32_t*, np.uint32_t*, np.int64_t*, size_type, bool)
        AdjVectorBQM(np.uint32_t*, np.uint32_t*, np.float32_t*, size_type, bool)
        AdjVectorBQM(np.uint32_t*, np.uint32_t*, np.float64_t*, size_type, bool)
        AdjVectorBQM(np.uint64_t*, np.uint64_t*, np.uint32_t*, size_type, bool)
        AdjVectorBQM(np.uint64_t*, np.uint64_t*, np.uint64_t*, size_type, bool)
        AdjVectorBQM(np.uint64_t*, np.uint64_t*, np.int32_t*, size_type, bool)
        AdjVectorBQM(np.uint64_t*, np.uint64_t*, np.int64_t*, size_type, bool)
        AdjVectorBQM(np.uint64_t*, np.uint64_t*, np.float32_t*, size_type, bool)
        AdjVectorBQM(np.uint64_t*, np.uint64_t*, np.float64_t*, size_type, bool)
        AdjVectorBQM(np.int32_t*, np.int32_t*, np.uint32_t*, size_type, bool)
        AdjVectorBQM(np.int32_t*, np.int32_t*, np.uint64_t*, size_type, bool)
        AdjVectorBQM(np.int32_t*, np.int32_t*, np.int32_t*, size_type, bool)
        AdjVectorBQM(np.int32_t*, np.int32_t*, np.int64_t*, size_type, bool)
        AdjVectorBQM(np.int32_t*, np.int32_t*, np.float32_t*, size_type, bool)
        AdjVectorBQM(np.int32_t*, np.int32_t*, np.float64_t*, size_type, bool)
        AdjVectorBQM(np.int64_t*, np.int64_t*, np.uint32_t*, size_type, bool)
        AdjVectorBQM(np.int64_t*, np.int64_t*, np.uint64_t*, size_type, bool)
        AdjVectorBQM(np.int64_t*, np.int64_t*, np.int32_t*, size_type, bool)
        AdjVectorBQM(np.int64_t*, np.int64_t*, np.int64_t*, size_type, bool)
        AdjVectorBQM(np.int64_t*, np.int64_t*, np.float32_t*, size_type, bool)
        AdjVectorBQM(np.int64_t*, np.int64_t*, np.float64_t*, size_type, bool)


        # methods

        size_type degree(variable_type) except +
        bias_type get_linear(variable_type) except +
        pair[bias_type, bool] get_quadratic(variable_type, variable_type) except +
        bias_type& linear(variable_type) except +
        pair[outvars_iterator, outvars_iterator] neighborhood(variable_type) except +
        pair[const_outvars_iterator, const_outvars_iterator] neighborhood(variable_type, variable_type) except +
        void normalize_neighborhood()
        void normalize_neighborhood(variable_type)
        void normalize_neighborhood[Iter](Iter, Iter)
        size_type num_interactions() except +
        size_type num_variables() except +
        void set_linear(variable_type, bias_type) except +
        bool set_quadratic(variable_type, variable_type, bias_type) except +

        # shapeable methods

        variable_type add_variable() except +
        variable_type pop_variable() except +
        bool remove_interaction(variable_type, variable_type) except +
