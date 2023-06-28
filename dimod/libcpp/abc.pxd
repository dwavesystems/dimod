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

from libcpp.utility cimport pair
from dimod.libcpp.vartypes cimport Vartype

__all__ = ['BinaryQuadraticModelBase']

cdef extern from "dimod/abc.h" namespace "dimod::abc" nogil:
    cdef cppclass QuadraticModelBase[Bias, Index]:

        ctypedef Bias bias_type
        ctypedef Index index_type
        ctypedef size_t size_type

        cppclass const_neighborhood_iterator:
            cppclass value_type:
                index_type v
                bias_type bias

                # todo: change to v, bias appropriately
                index_type first "v"
                bias_type second "bias"

            const_neighborhood_iterator() except +
            const_neighborhood_iterator(const_neighborhood_iterator&) except +
            operator=(const_neighborhood_iterator&) except +
            const value_type& operator*()
            const_neighborhood_iterator operator++()
            const_neighborhood_iterator operator--()
            const_neighborhood_iterator operator++(int)
            const_neighborhood_iterator operator--(int)
            const_neighborhood_iterator operator+(size_type)
            const_neighborhood_iterator operator-(size_type)
            ptrdiff_t operator-(const_neighborhood_iterator)
            bint operator==(const_neighborhood_iterator)
            bint operator!=(const_neighborhood_iterator)
            bint operator<(const_neighborhood_iterator)
            bint operator>(const_neighborhood_iterator)
            bint operator<=(const_neighborhood_iterator)
            bint operator>=(const_neighborhood_iterator)


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

        # Without this, Cython can get confused about the return type of
        # .cbegin_quadratic() when casting.
        cppclass const_quadratic_iterator2 "const_quadratic_iterator":
            cppclass value_type:
                index_type u
                index_type v
                bias_type bias

            const_quadratic_iterator2()
            const_quadratic_iterator2(const_quadratic_iterator2&) except +
            operator=(const_quadratic_iterator2&) except +
            const value_type& operator*()
            const_quadratic_iterator2 operator++()
            const_quadratic_iterator2 operator++(int)
            bint operator==(const_quadratic_iterator2)
            bint operator!=(const_quadratic_iterator2)

        # developer note: we avoid any overloads, due to a bug in Cython
        # https://github.com/cython/cython/issues/1357
        # https://github.com/cython/cython/issues/1868

        void add_linear(index_type, bias_type)
        void add_offset(bias_type)
        void add_quadratic(index_type, index_type, bias_type)
        void add_quadratic_from_coo "add_quadratic" [ItRow, ItCol, ItBias](ItRow, ItCol, ItBias, index_type)
        void add_quadratic_back(index_type, index_type, bias_type)
        void add_quadratic_from_dense[T](const T dense[], index_type)
        const_neighborhood_iterator cbegin_neighborhood(index_type)
        const_neighborhood_iterator cend_neighborhood(index_type)
        const_quadratic_iterator cbegin_quadratic()
        const_quadratic_iterator cend_quadratic()
        void clear()
        bias_type energy[Iter](Iter)
        void fix_variable[T](index_type, T)
        bint is_linear()
        bias_type linear(index_type)
        bias_type lower_bound(index_type)
        size_type nbytes()
        size_type nbytes(bint)
        size_type num_interactions()
        size_type degree "num_interactions" (index_type)
        size_type num_variables()
        bias_type offset()
        bias_type quadratic(index_type, index_type)
        bias_type quadratic_at(index_type, index_type) except+
        bint remove_interaction(index_type, index_type)
        void remove_variable(index_type)
        void scale(bias_type)
        void set_linear(index_type, bias_type)
        void set_offset(bias_type)
        void set_quadratic(index_type, index_type, bias_type) except+
        bias_type upper_bound(index_type)
        Vartype vartype(index_type)
