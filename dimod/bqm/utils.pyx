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

cdef extern from "numpy/arrayobject.h":
    # The comment in
    # https://github.com/numpy/numpy/blob/c0992ed4856df9fe02c2b31744a8a7e9088aedbc/numpy/__init__.pxd#L522
    # implies that this function steals the reference to dtype. However,
    # https://github.com/numpy/numpy/blob/5ce770ae3de63861c768229573397cadd052f712/numpy/core/src/multiarray/scalarapi.c#L617
    # seems to contradict that. A cursory read of the code seems to support the
    # latter conclusion, so for now we'll assume this is safe to use.
    object PyArray_Scalar(void* data, np.dtype dtype, object itemsize)

np.import_array()

cdef object as_numpy_scalar(double a, np.dtype dtype):
    """Note that the memory is interpreted to match dtype, not a cast"""
    return PyArray_Scalar(&a, dtype, None)
