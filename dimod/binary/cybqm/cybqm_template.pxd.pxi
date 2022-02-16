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

from dimod.binary.cybqm.base cimport cyBQMBase
from dimod.cyvariables cimport cyVariables
from dimod.cyutilities cimport ConstNumeric
from dimod.libcpp cimport cppBinaryQuadraticModel


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
    cdef np.float64_t[::1] _energies(self, ConstNumeric[:, ::1] samples, object labels)
    cdef Py_ssize_t _index(self, object, bint permissive=*) except -1
    cdef void _set_linear(self, Py_ssize_t, bias_type)
    cdef void _set_offset(self, bias_type)
    cpdef Py_ssize_t add_linear_from_array(self, ConstNumeric[:] linear) except -1
    cpdef Py_ssize_t add_quadratic_from_dense(self, ConstNumeric[:, ::1] quadratic) except -1
    cpdef Py_ssize_t change_vartype(self, object) except -1
    cdef const cppBinaryQuadraticModel[bias_type, index_type]* data(self)
    cpdef bint is_linear(self)
    cpdef Py_ssize_t nbytes(self, bint capacity=*)
    cpdef Py_ssize_t num_interactions(self)
    cpdef Py_ssize_t num_variables(self)
    cpdef Py_ssize_t resize(self, Py_ssize_t) except? 0
    cpdef void scale(self, bias_type)
