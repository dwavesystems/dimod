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

from libcpp.utility cimport pair

from dimod.libcpp.abc cimport QuadraticModelBase
from dimod.libcpp.vartypes cimport Vartype

__all__ = ['BinaryQuadraticModel']


cdef extern from "dimod/binary_quadratic_model.h" namespace "dimod" nogil:
    cdef cppclass BinaryQuadraticModel[Bias, Index](QuadraticModelBase[Bias, Index]):
        ctypedef Bias bias_type
        ctypedef Index index_type
        ctypedef size_t size_type

        pair[const_neighborhood_iterator, const_neighborhood_iterator] neighborhood(index_type)  # deprecated
        pair[const_neighborhood_iterator, const_neighborhood_iterator] neighborhood(index_type, index_type)  # deprecated

        # Methods/attributes specific to the BQM
        BinaryQuadraticModel()
        BinaryQuadraticModel(Vartype)
        BinaryQuadraticModel(index_type, Vartype)

        index_type add_variable()
        void change_vartype(Vartype) except+
        void resize(index_type)
        Vartype vartype()
