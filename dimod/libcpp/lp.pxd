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

from libcpp.string cimport string
from libcpp.vector cimport vector

from dimod.libcpp.constrained_quadratic_model cimport ConstrainedQuadraticModel

__all__ = ['LPModel', 'read']


cdef extern from "dimod/lp.h" namespace "dimod::lp" nogil:
    cdef cppclass LPModel[Bias, Index]:
        ConstrainedQuadraticModel[Bias, Index] model
        vector[string] variable_labels
        vector[string] constraint_labels

    LPModel[Bias, Index] read[Bias, Index] (const string) except +
