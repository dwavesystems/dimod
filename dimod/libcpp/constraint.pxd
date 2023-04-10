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

from dimod.libcpp.expression cimport Expression

__all__ = ['Sense', 'Penalty', 'Constraint']


cdef extern from "dimod/constraint.h" namespace "dimod" nogil:
    enum Sense:
        EQ
        LE
        GE

    enum Penalty:
        LINEAR
        QUADRATIC
        CONSTANT

    cdef cppclass Constraint[Bias, Index](Expression[Bias, Index]):
        ctypedef Bias bias_type
        ctypedef Index index_type
        ctypedef size_t size_type

        bias_type rhs()
        Sense sense()
        Penalty penalty()
        bias_type weight()

        bint is_onehot()
        bint is_soft()
        void mark_discrete();
        void mark_discrete(bint mark);
        bint marked_discrete() const;
        void set_rhs(bias_type)
        void set_sense(Sense)
        void set_penalty(Penalty)
        void set_weight(bias_type)
