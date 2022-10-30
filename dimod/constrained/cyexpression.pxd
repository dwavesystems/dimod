# distutils: language = c++
# cython: language_level=3

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

from dimod.constrained.cyconstrained cimport cyConstrainedQuadraticModel
from dimod.cyqmbase.cyqmbase_float64 cimport bias_type, index_type
from dimod.libcpp.abc cimport QuadraticModelBase as cppQuadraticModelBase
from dimod.libcpp.constrained_quadratic_model cimport Expression as cppExpression, Constraint as cppConstraint


cdef class _cyExpression:
    cdef readonly cyConstrainedQuadraticModel parent

    cdef readonly object dtype
    cdef readonly object index_dtype

    cdef cppExpression[bias_type, index_type]* expression(self) except NULL


cdef class cyObjectiveView(_cyExpression):
    pass


cdef class cyConstraintView(_cyExpression):
    cdef weak_ptr[cppConstraint[bias_type, index_type]] constraint_ptr

    cdef cppConstraint[bias_type, index_type]* constraint(self) except NULL
