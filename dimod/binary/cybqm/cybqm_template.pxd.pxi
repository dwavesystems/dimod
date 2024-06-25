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

from dimod.typing cimport Numeric
from dimod.libcpp.binary_quadratic_model cimport BinaryQuadraticModel as cppBinaryQuadraticModel


cdef class cyBQM_template(cyQMBase):
    cdef cppBinaryQuadraticModel[bias_type, index_type]* cppbqm

    # developer note: we mostly use Py_ssize_t rather than size_t
    # since python does not really have an unsigned integer type that it
    # likes to use

    cdef Py_ssize_t _index(self, object, bint permissive=*) except -1
    cpdef Py_ssize_t add_linear_from_array(self, const Numeric[:] linear) except -1
    cpdef Py_ssize_t add_quadratic_from_dense(self, const Numeric[:, ::1] quadratic) except -1
    cpdef Py_ssize_t change_vartype(self, object) except -1
    cdef const cppBinaryQuadraticModel[bias_type, index_type]* data(self)
    cpdef Py_ssize_t resize(self, Py_ssize_t) except? 0
