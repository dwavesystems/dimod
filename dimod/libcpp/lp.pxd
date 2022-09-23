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
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

from dimod.libcpp.quadratic_model cimport QuadraticModel

__all__ = ['LPModel', 'read']


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
