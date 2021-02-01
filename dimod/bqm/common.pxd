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
cimport numpy as np

# Developer note: we'd like to use fused types here, which would allow us to
# construct cyBQMs of various type combinations. Unfortunately, cython
# does not allow fused types on cdef classes (yet) so for now we just fix them.
ctypedef np.uint32_t VarIndex
ctypedef np.float64_t Bias

ctypedef size_t NeighborhoodIndex  

# convenience fused types
ctypedef fused Integral:
    np.uint8_t
    np.uint16_t
    np.uint32_t
    np.uint64_t
    np.int8_t
    np.int16_t
    np.int32_t
    np.int64_t

ctypedef fused Integral32plus:
    np.uint32_t
    np.uint64_t
    np.int32_t
    np.int64_t

ctypedef fused Numeric:
    Integral
    np.float32_t
    np.float64_t

ctypedef fused Numeric32plus:
    Integral32plus
    np.float32_t
    np.float64_t
