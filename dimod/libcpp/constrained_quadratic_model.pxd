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
from dimod.libcpp.constraint cimport Constraint, Penalty, Sense
from dimod.libcpp.expression cimport Expression
from dimod.libcpp.vartypes cimport Vartype

__all__ = ['ConstrainedQuadraticModel']


cdef extern from "dimod/constrained_quadratic_model.h" namespace "dimod" nogil:
    cdef cppclass ConstrainedQuadraticModel[bias_type, index_type]:
        Expression[bias_type, index_type] objective

        index_type add_constraint()
        index_type add_constraint(Constraint[bias_type, index_type]) except+
        index_type add_constraint[B, I, T](QuadraticModelBase[B, I]&, Sense, bias_type, vector[T])
        index_type add_constraint(QuadraticModelBase[bias_type, index_type]&, Sense, bias_type, vector[index_type])
        index_type add_constraints(index_type)
        index_type add_variable(Vartype)
        index_type add_variable(Vartype, bias_type, bias_type)
        void change_vartype(Vartype, index_type) except+
        void clear()
        Constraint[bias_type, index_type]& constraint_ref(index_type)
        weak_ptr[Constraint[bias_type, index_type]] constraint_weak_ptr(index_type)
        void fix_variable[T](index_type, T)
        ConstrainedQuadraticModel fix_variables[VarIter, AssignmentIter](VarIter, VarIter, AssignmentIter)
        bias_type lower_bound(index_type)
        Constraint[bias_type, index_type] new_constraint()
        size_t num_constraints()
        size_t num_interactions()
        size_t num_variables()
        void remove_constraint(index_type)
        void remove_variable(index_type)
        void set_lower_bound(index_type, bias_type)
        void set_objective[B, I](QuadraticModelBase[B, I]&)
        void set_objective[B, I, T](QuadraticModelBase[B, I]&, vector[T])
        void set_upper_bound(index_type, bias_type)
        void substitute_variable(index_type, bias_type, bias_type)
        bias_type upper_bound(index_type)
        Vartype vartype(index_type)
