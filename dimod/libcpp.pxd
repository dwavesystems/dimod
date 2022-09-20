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

cdef extern from "dimod/vartypes.h" namespace "dimod" nogil:
    enum Vartype:
        BINARY
        SPIN
        INTEGER
        REAL

    cdef cppclass vartype_info [Bias]:
        @staticmethod
        Bias default_max(Vartype)
        @staticmethod
        Bias default_min(Vartype)
        @staticmethod
        Bias max(Vartype)
        @staticmethod
        Bias min(Vartype)


cdef extern from "dimod/quadratic_model.h" namespace "dimod" nogil:
    cdef cppclass BinaryQuadraticModel[Bias, Index]:
        # Methods/attributes inherited from or overwriting the abstract base class
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

        void add_linear(index_type, bias_type)
        void add_offset(bias_type)
        void add_quadratic(index_type, index_type, bias_type)
        void add_quadratic[ItRow, ItCol, ItBias](ItRow, ItCol, ItBias, index_type)
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
        pair[const_neighborhood_iterator, const_neighborhood_iterator] neighborhood(index_type)  # deprecated
        pair[const_neighborhood_iterator, const_neighborhood_iterator] neighborhood(index_type, index_type)  # deprecated
        size_type nbytes()
        size_type nbytes(bint)
        size_type num_interactions()
        size_type num_interactions(index_type)
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

        # Methods/attributes specific to the BQM
        BinaryQuadraticModel()
        BinaryQuadraticModel(Vartype)
        BinaryQuadraticModel(index_type, Vartype)

        index_type add_variable()
        void change_vartype(Vartype) except+
        void resize(index_type)
        Vartype vartype()


    cdef cppclass QuadraticModel[Bias, Index]:
        # Methods/attributes inherited from or overwriting the abstract base class
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

        void add_linear(index_type, bias_type)
        void add_offset(bias_type)
        void add_quadratic(index_type, index_type, bias_type)
        void add_quadratic[ItRow, ItCol, ItBias](ItRow, ItCol, ItBias, index_type)
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
        pair[const_neighborhood_iterator, const_neighborhood_iterator] neighborhood(index_type)  # deprecated
        pair[const_neighborhood_iterator, const_neighborhood_iterator] neighborhood(index_type, index_type)  # deprecated
        size_type nbytes()
        size_type nbytes(bint)
        size_type num_interactions()
        size_type num_interactions(index_type)
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

        # Methods/attributes specific to the QM
        index_type add_variable(Vartype)
        index_type add_variable(Vartype, bias_type, bias_type)
        index_type add_variables(Vartype, index_type)
        index_type add_variables(Vartype, index_type, bias_type bias_type)
        void change_vartype(Vartype, index_type) except+
        void resize(index_type) except+
        void resize(index_type, Vartype) except+
        void resize(index_type, Vartype, bias_type, bias_type)
        void set_lower_bound(index_type, bias_type)
        void set_upper_bound(index_type, bias_type)
        void set_vartype(index_type, Vartype)

from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

cdef extern from "dimod/lp.h" namespace "dimod::lp" nogil:
    cdef cppclass Expression[Bias, Index]:
        QuadraticModel[Bias, Index] model
        unordered_map[string, Index] labels
        string name

    cdef cppclass Constraint[Bias, Index]:
        Expression[Bias, Index] lhs
        string sense
        Bias rhs

    cdef cppclass LPModel[Bias, Index]:
        Expression[Bias, Index] objective
        vector[Constraint[Bias, Index]] constraints
        bint minimize

    LPModel[Bias, Index] read[Bias, Index] (const string) except +
