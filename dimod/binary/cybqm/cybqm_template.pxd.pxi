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

cimport cython

from libcpp.pair cimport pair
from libcpp.vector cimport vector

from dimod.binary.cybqm.base cimport cyBQMBase
from dimod.cyvariables cimport cyVariables
from dimod.cyutilities cimport ConstNumeric


cdef extern from "dimod/quadratic_model.h" namespace "dimod" nogil:
    enum cppVartype "dimod::Vartype":
        BINARY
        SPIN
        INTEGER

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
        const_quadratic_iterator cbegin_quadratic()
        const_quadratic_iterator cend_quadratic()
        void change_vartype(cppVartype)
        bint is_linear()
        bias_type& linear(index_type)
        bias_type& offset()
        bias_type quadratic_at(index_type, index_type) except +
        pair[const_neighborhood_iterator, const_neighborhood_iterator] neighborhood(size_type)
        size_type num_variables()
        size_type num_interactions()
        size_type num_interactions(index_type)
        bint remove_interaction(index_type, index_type) except +
        void resize(index_type)
        void set_quadratic(index_type, index_type, bias_type) except +
        cppVartype& vartype()


cdef class cyBQM_template(cyBQMBase):
    cdef cppBinaryQuadraticModel[bias_type, index_type] cppbqm

    cdef readonly object dtype
    cdef readonly object index_dtype
    cdef readonly cyVariables variables

    # developer note: we mostly use Py_ssize_t rather than size_t
    # since python does not really have an unsigned integer type that it
    # likes to use

    cdef void _add_linear(self, Py_ssize_t, bias_type)
    cdef void _add_offset(self, bias_type)
    cdef cython.floating[::1] _energies(self, object samples_like, cython.floating signal=*)
    cdef Py_ssize_t _index(self, object, bint permissive=*) except -1
    cdef void _set_linear(self, Py_ssize_t, bias_type)
    cdef void _set_offset(self, bias_type)
    cpdef Py_ssize_t add_linear_from_array(self, ConstNumeric[:] linear) except -1
    cpdef Py_ssize_t add_quadratic_from_dense(self, ConstNumeric[:, ::1] quadratic) except -1
    cpdef Py_ssize_t change_vartype(self, object) except -1
    cpdef bint is_linear(self)
    cpdef Py_ssize_t num_interactions(self)
    cpdef Py_ssize_t num_variables(self)
    cpdef Py_ssize_t resize(self, Py_ssize_t) except -1
