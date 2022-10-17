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

from dimod.cyqmbase.cyqmbase_float64 cimport bias_type, index_type
from dimod.cyvariables cimport cyVariables
from dimod.libcpp.presolve cimport PostSolver as cppPostSolver, PreSolver as cppPreSolver


cdef class cyPreSolver:
    cdef cppPreSolver[bias_type, index_type, double] cpppresolver  # dev note: terrible name...
    cdef readonly cyVariables variables
