# Copyright 2021 D-Wave Systems Inc.
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

"""
Make dimod's C++ code available to Cython.

If, for example, you wished to access dimod's C++ BinaryQuadraticModel
implementation in an external library, you use

>>> from dimod.libcpp cimport cppBinaryQuadraticModel

A convention of prepending 'cpp' to the classes and types is followed in
order to differentiate the C++ classes from Python and Cython classes.

Note that in some cases the Cython declarations do not perfectly match the
C++ ones because these declarations are only used by Cython as a guide to
generate the relevant C++ code.

"""

from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


cdef extern from "dimod/quadratic_model.h" namespace "dimod" nogil:
    enum cppVartype "dimod::Vartype":
        BINARY
        SPIN
        INTEGER
        REAL

    cpdef cppclass cppvartype_info "dimod::vartype_info" [Bias]:
        @staticmethod
        Bias default_max(cppVartype)
        @staticmethod
        Bias default_min(cppVartype)
        @staticmethod
        Bias max(cppVartype)
        @staticmethod
        Bias min(cppVartype)

    cdef cppclass cppBinaryQuadraticModel "dimod::BinaryQuadraticModel" [Bias, Index]:
        ctypedef Bias bias_type
        ctypedef size_t size_type
        ctypedef Index index_type

        cppclass const_neighborhood_iterator:
            pair[index_type, bias_type] operator*()
            const_neighborhood_iterator operator++()
            const_neighborhood_iterator operator--()
            bint operator==(const_neighborhood_iterator)
            bint operator!=(const_neighborhood_iterator)

        cppclass const_quadratic_iterator:
            cppclass value_type:
                index_type u
                index_type v
                bias_type bias

            value_type operator*()
            const_quadratic_iterator operator++()
            bint operator==(const_quadratic_iterator&)
            bint operator!=(const_quadratic_iterator&)

        cppBinaryQuadraticModel()
        cppBinaryQuadraticModel(cppVartype)

        void add_bqm[B, I](const cppBinaryQuadraticModel[B, I]&)
        void add_bqm[B, I, T](const cppBinaryQuadraticModel[B, I]&, const vector[T]&) except +
        void add_quadratic(index_type, index_type, bias_type) except +
        void add_quadratic[T](const T dense[], index_type)
        void add_quadratic[ItRow, ItCol, ItBias](ItRow, ItCol, ItBias, index_type) except +
        index_type add_variable()
        const_quadratic_iterator cbegin_quadratic()
        const_quadratic_iterator cend_quadratic()
        void change_vartype(cppVartype)
        bint is_linear()
        bias_type& linear(index_type)
        const bias_type lower_bound(index_type)
        bias_type& offset()
        bias_type quadratic(index_type, index_type)
        bias_type quadratic_at(index_type, index_type) except +
        size_type nbytes()
        size_type nbytes(bint)
        pair[const_neighborhood_iterator, const_neighborhood_iterator] neighborhood(index_type)
        pair[const_neighborhood_iterator, const_neighborhood_iterator] neighborhood(index_type, index_type)
        size_type num_variables()
        size_type num_interactions()
        size_type num_interactions(index_type)
        bint remove_interaction(index_type, index_type) except +
        void resize(index_type)
        void scale(bias_type)
        void set_quadratic(index_type, index_type, bias_type) except +
        void swap_variables(index_type, index_type)
        const bias_type upper_bound(index_type)
        cppVartype& vartype()
        cppVartype& vartype(index_type)

    cdef cppclass cppQuadraticModel "dimod::QuadraticModel" [Bias, Index]:
        ctypedef Bias bias_type
        ctypedef size_t size_type
        ctypedef Index index_type

        cppclass const_neighborhood_iterator:
            pair[index_type, bias_type] operator*()
            const_neighborhood_iterator operator++()
            const_neighborhood_iterator operator--()
            bint operator==(const_neighborhood_iterator)
            bint operator!=(const_neighborhood_iterator)

        cppclass const_quadratic_iterator:
            cppclass value_type:
                index_type u
                index_type v
                bias_type bias

            value_type operator*()
            const_quadratic_iterator operator++()
            bint operator==(const_quadratic_iterator&)
            bint operator!=(const_quadratic_iterator&)

        void add_quadratic(index_type, index_type, bias_type) except +
        index_type add_variable(cppVartype) except+
        index_type add_variable(cppVartype, bias_type, bias_type) except +
        const_quadratic_iterator cbegin_quadratic()
        void change_vartype(cppVartype, index_type) except +
        const_quadratic_iterator cend_quadratic()
        bint is_linear()
        bias_type& linear(index_type)
        const bias_type& lower_bound(index_type)
        size_type nbytes()
        size_type nbytes(bint)
        pair[const_neighborhood_iterator, const_neighborhood_iterator] neighborhood(size_type)
        size_type num_variables()
        size_type num_interactions()
        size_type num_interactions(index_type)
        bias_type& offset()
        bias_type quadratic_at(index_type, index_type) except +
        bint remove_interaction(index_type, index_type) except +
        void resize(index_type) except +
        void scale(bias_type)
        void set_quadratic(index_type, index_type, bias_type)
        void swap(cppQuadraticModel&)
        void swap_variables(index_type, index_type)
        const bias_type& upper_bound(index_type)
        const cppVartype& vartype(index_type)

cdef extern from "dimod/lp.h" namespace "dimod::lp" nogil:
    cdef cppclass cppExpression "dimod::lp::Expression" [Bias, Index]:
        cppQuadraticModel[Bias, Index] model
        unordered_map[string, Index] labels
        string name

    cdef cppclass cppConstraint "dimod::lp::Constraint" [Bias, Index]:
        cppExpression[Bias, Index] lhs
        string sense
        Bias rhs

    cdef cppclass cppLPModel "dimod::lp::LPModel" [Bias, Index]:
        cppExpression[Bias, Index] objective
        vector[cppConstraint[Bias, Index]] constraints
        bint minimize

    cppLPModel[Bias, Index] cppread_lp "dimod::lp::read" [Bias, Index] (const string) except +
