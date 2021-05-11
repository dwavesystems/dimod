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

__all__ = ['cyBQMBase']

# todo: be consistent about sign/unsign and support int8

ctypedef fused Integral32plus:
    np.uint32_t
    np.uint64_t
    np.int32_t
    np.int64_t

ctypedef fused Numeric:
    cython.integral  # short, int, long
    cython.floating  # float, double

ctypedef fused Numeric32plus:
    Integral32plus
    np.float32_t
    np.float64_t


cdef class cyBQMBase:
    pass
