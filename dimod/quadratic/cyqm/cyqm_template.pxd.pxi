# distutils: language = c++
# cython: language_level=3

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

from dimod.cyutilities cimport ConstNumeric
from dimod.cyvariables cimport cyVariables
from dimod.quadratic.cyqm.cyqm_base cimport cyQMBase
from dimod.libcpp cimport cppQuadraticModel, cppVartype


cdef class cyQM_template(cyQMBase):
    cdef cppQuadraticModel[bias_type, index_type] cppqm

    cdef readonly object dtype
    cdef readonly object index_dtype
    cdef readonly cyVariables variables

    cdef public int REAL_INTERACTIONS

    cdef void _add_linear(self, Py_ssize_t, bias_type)
    cdef Py_ssize_t _add_quadratic(self, index_type, index_type, bias_type) except -1
    cdef np.float64_t[::1] _energies(self, ConstNumeric[:, ::1] samples, object labels)
    cdef void _set_linear(self, Py_ssize_t, bias_type)
    cdef cppVartype cppvartype(self, object) except? cppVartype.SPIN
    cdef const cppQuadraticModel[bias_type, index_type]* data(self)

    cpdef bint is_linear(self)
    cpdef void scale(self, bias_type)
    cpdef Py_ssize_t nbytes(self, bint capacity=*)
    cpdef Py_ssize_t num_interactions(self)
    cpdef Py_ssize_t num_variables(self)
