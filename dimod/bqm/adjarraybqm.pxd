# distutils: language = c++
# cython: language_level=3
#
# Copyright 2019 D-Wave Systems Inc.
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
#
# =============================================================================

from libcpp cimport bool
from libcpp.pair cimport pair
from libcpp.vector cimport vector

# Developer note: we'd like to use fused types here, which would allow us to
# construct AdjMapBQMs of various type combinations. Unfortunately, cython
# does not allow fused types on cdef classes (yet) so for now we just fix them.
ctypedef unsigned int VarIndex
ctypedef double Bias

ctypedef fused SampleVar:
    signed char  # np.int8
    signed short  # np.int16
    signed int  # np.int32
    signed long long  # np.int64
    unsigned char  # np.uint8
    unsigned short  # np.uint16
    unsigned int  # np.uint32
    unsigned long long  #np.uint64
    float  # np.float32
    double  # np.float64


cdef class AdjArrayBQM:
    cdef vector[pair[size_t, Bias]] invars_
    cdef vector[pair[VarIndex, Bias]] outvars_

    cdef public object dtype
    cdef public object index_dtype

    # these are not public because the user has no way to access the underlying
    # variable indices
    cdef object label_to_idx
    cdef object idx_to_label

