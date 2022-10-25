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

__all__ = ['QuadraticModel']


cdef extern from "dimod/quadratic_model.h" namespace "dimod" nogil:
    cdef cppclass QuadraticModel[Bias, Index](QuadraticModelBase[Bias, Index]):
        ctypedef Bias bias_type
        ctypedef Index index_type
        ctypedef size_t size_type

        pair[const_neighborhood_iterator, const_neighborhood_iterator] neighborhood(index_type)  # deprecated
        pair[const_neighborhood_iterator, const_neighborhood_iterator] neighborhood(index_type, index_type)  # deprecated

        # Methods/attributes specific to the QM
        index_type add_variable(Vartype)
        index_type add_variable(Vartype, bias_type, bias_type)
        index_type add_variables(Vartype, index_type)
        index_type add_variables(Vartype, index_type, bias_type bias_type)
        void change_vartype(Vartype, index_type) except+
        void resize(index_type) except+
        void resize(index_type, Vartype) except+
        void resize(index_type, Vartype, bias_type, bias_type)
        void set_lower_bound(index_type, bias_type)
        void set_upper_bound(index_type, bias_type)
        void set_vartype(index_type, Vartype)
