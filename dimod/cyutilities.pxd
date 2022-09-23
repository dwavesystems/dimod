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
cimport numpy as np

from dimod.libcpp.vartypes cimport Vartype as cppVartype

__all__ = [
    'as_numpy_float',
    'coo_sort',
    'Integer',
    'SignedInteger',
    'UnsignedInteger',
    ]

cdef fused SignedInteger:
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t

cdef fused UnsignedInteger:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t

cdef fused Integer:
    SignedInteger
    UnsignedInteger

cdef fused ConstInteger:
    const short
    const int
    const long long
    const unsigned short
    const unsigned int
    const unsigned long long

ctypedef fused Numeric:
    signed char
    cython.integral  # short, int, long
    cython.floating  # float, double

# cython doesn't like const Numeric
ctypedef fused ConstNumeric:
    const signed char
    const signed short
    const signed int
    const signed long long
    const float
    const double


cdef object as_numpy_float(cython.floating)


cpdef Py_ssize_t coo_sort(Integer[:], Integer[:], cython.floating[:]) except -1

cdef cppVartype cppvartype(object) except? cppVartype.SPIN
