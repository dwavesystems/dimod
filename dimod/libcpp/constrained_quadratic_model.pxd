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

from libcpp.memory cimport weak_ptr
from libcpp.vector cimport vector

from dimod.libcpp.abc cimport QuadraticModelBase
from dimod.libcpp.vartypes cimport Vartype

__all__ = ['ConstrainedQuadraticModel', 'Sense', 'Penalty', 'Constraint', 'Expression']


cdef extern from "dimod/constrained_quadratic_model.h" namespace "dimod" nogil:
    enum Sense:
        EQ
        LE
        GE

    enum Penalty:
        LINEAR
        QUADRATIC
        CONSTANT

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


        # void add_offset(bias_type)
        # void add_linear(index_type, bias_type)
        # void add_quadratic(index_type, index_type, bias_type)
        # void clear()
        # size_type num_variables()

        bint has_variable(index_type)
        bint is_disjoint(const Expression&)

        # exprconst_quadratic_iterator cbegin_quadratic()
        # exprconst_quadratic_iterator cend_quadratic()

        vector[index_type]& variables()

    cdef cppclass Constraint[Bias, Index](Expression[Bias, Index]):
        ctypedef Bias bias_type
        ctypedef Index index_type
        ctypedef size_t size_type

        bias_type rhs()
        Sense sense()
        Penalty penalty()
        bias_type weight()

        bint is_onehot()
        bint is_soft()
        void mark_discrete();
        void mark_discrete(bint mark);
        bint marked_discrete() const;
        void set_rhs(bias_type)
        void set_sense(Sense)
        void set_penalty(Penalty)
        void set_weight(bias_type)


    cdef cppclass ConstrainedQuadraticModel[bias_type, index_type]:
        Expression[bias_type, index_type] objective
        # vector[Constraint[bias_type, index_type]] constraints

        index_type add_constraint()
        index_type add_constraint(Constraint[bias_type, index_type]) except+
        index_type add_constraint[B, I, T](QuadraticModelBase[B, I]&, Sense, bias_type, vector[T])
        index_type add_constraints(index_type)
        index_type add_variable(Vartype)
        index_type add_variable(Vartype, bias_type, bias_type)
        void change_vartype(Vartype, index_type) except+
        void clear()
        Constraint[bias_type, index_type]& constraint_ref(index_type)
        weak_ptr[Constraint[bias_type, index_type]] constraint_weak_ptr(index_type)
        void fix_variable[T](index_type, T)
        bias_type lower_bound(index_type)
        Constraint[bias_type, index_type] new_constraint()
        size_t num_constraints()
        size_t num_interactions()
        size_t num_variables()
        void remove_constraint(index_type)
        void remove_variable(index_type)
        void set_lower_bound(index_type, bias_type)
        void set_objective[B, I](QuadraticModelBase&)
        void set_objective[B, I, T](QuadraticModelBase[B, I]&, vector[T])
        void set_upper_bound(index_type, bias_type)
        void substitute_variable(index_type, bias_type, bias_type)
        bias_type upper_bound(index_type)
        Vartype vartype(index_type)
