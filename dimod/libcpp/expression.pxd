# distutils: include_dirs = dimod/include/

# Copyright 2022 D-Wave Systems Inc.
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

from libcpp.vector cimport vector

from dimod.libcpp.abc cimport QuadraticModelBase

__all__ = ['Expression']


cdef extern from "dimod/expression.h" namespace "dimod" nogil:
    cdef cppclass Expression[Bias, Index](QuadraticModelBase[Bias, Index]):
        ctypedef Bias bias_type
        ctypedef Index index_type
        ctypedef size_t size_type

        cppclass const_quadratic_iterator:
            cppclass value_type:
                index_type u
                index_type v
                bias_type bias

            const_quadratic_iterator()
            const_quadratic_iterator(const_quadratic_iterator&) except +
            operator=(const_quadratic_iterator&) except +
            const value_type& operator*()
            const_quadratic_iterator operator++()
            const_quadratic_iterator operator++(int)
            bint operator==(const_quadratic_iterator)
            bint operator!=(const_quadratic_iterator)

        cppclass const_neighborhood_iterator:
            cppclass value_type:
                index_type v
                bias_type bias

            const_neighborhood_iterator()
            const_neighborhood_iterator(const_neighborhood_iterator&) except +
            operator=(const_neighborhood_iterator&) except +
            const value_type& operator*()
            const_neighborhood_iterator operator++()
            const_neighborhood_iterator operator++(int)
            bint operator==(const_neighborhood_iterator)
            bint operator!=(const_neighborhood_iterator)

        bint has_variable(index_type)
        bint is_disjoint(const Expression&)
        const vector[index_type]& variables()
