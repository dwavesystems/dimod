# Copyright 2024 D-Wave Inc.
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

from libc.stdint cimport (
    int8_t, int16_t, int32_t, int64_t,
    uint8_t, uint16_t, uint32_t, uint64_t,
    )

# Follow NumPy
# https://github.com/numpy/numpy/blob/882611cf11e1925bd8f3a8d7aa00d74675a7601c/numpy/__init__.pxd#L73-L74
# By following their behavior but not cimporting NumPy directly we can avoid  requiring
# NumPy at build-time.
ctypedef float float32_t
ctypedef double float64_t

# cython.integral only has short, int, long, so we need our own to also support int8
cdef fused SignedInteger:
    int8_t
    int16_t
    int32_t
    int64_t

# cython.integral doesn't support unsigned at all.
cdef fused UnsignedInteger:
    uint8_t
    uint16_t
    uint32_t
    uint64_t

cdef fused Integer:
    SignedInteger
    UnsignedInteger

ctypedef fused Numeric:
    int8_t
    int16_t
    int32_t
    int64_t
    float32_t
    float64_t
