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

# cimport numpy as np

from dimod.libcpp.constrained_quadratic_model cimport ConstrainedQuadraticModel as cppConstrainedQuadraticModel
from dimod.constrained.cyexpression cimport cyObjectiveView, cyConstraintView
from dimod.cyqmbase.cyqmbase_float64 cimport cyQMBase_float64, bias_type, index_type
from dimod.cyvariables cimport cyVariables


cdef class cyConstrainedQuadraticModel:    
    cdef cppConstrainedQuadraticModel[bias_type, index_type] cppcqm
    
    cdef readonly cyObjectiveView objective
    """Objective to be minimized."""

    cdef readonly cyVariables variables
    """Variables in use over the objective and all constraints."""

    cdef readonly cyVariables constraint_labels
    """The labels for each of the constraints."""

    cdef readonly object dtype
    """The type of the biases."""

    cdef readonly object index_dtype
    """The type of the indices."""

    cdef public bint REAL_INTERACTIONS

    cdef readonly object constraints
    """Constraints as a mapping.

    This dictionary and its contents should not be modified.
    """

cdef object make_cqm(cppConstrainedQuadraticModel[bias_type, index_type] cppcqm)
