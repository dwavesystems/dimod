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
from dimod.typing cimport Numeric
from dimod.cyvariables cimport cyVariables

from dwave.preprocessing.libcpp cimport Presolver as cppPresolver

__all__ = ['cyPresolver']


cdef extern from "<mutex>" namespace "std" nogil:
    cppclass mutex:
        void lock()
        void unlock()


cdef class cyPresolver:
    cdef cppPresolver[bias_type, index_type, double]* cpppresolver  # dev note: terrible name...

    cdef mutex mutex
    
    cdef cyVariables _original_variables
    cdef Py_ssize_t _model_num_variables

    cpdef bint apply(self) except*
    cpdef bint normalize(self) except*
    cpdef bint presolve(self, double time_limit_s=*) except*

    cdef Py_ssize_t restore_sample(self, const Numeric[::1] reduced_sample, double[::1] original_sample) except -1 nogil
