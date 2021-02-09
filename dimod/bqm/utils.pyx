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
cimport cython

import numpy as np

from dimod.bqm.common cimport Bias, VarIndex

cdef extern from "numpy/arrayobject.h":
    # The comment in
    # https://github.com/numpy/numpy/blob/c0992ed4856df9fe02c2b31744a8a7e9088aedbc/numpy/__init__.pxd#L522
    # implies that this function steals the reference to dtype. However,
    # https://github.com/numpy/numpy/blob/5ce770ae3de63861c768229573397cadd052f712/numpy/core/src/multiarray/scalarapi.c#L617
    # seems to contradict that. A cursory read of the code seems to support the
    # latter conclusion, so for now we'll assume this is safe to use.
    object PyArray_Scalar(void* data, np.dtype descr, object base)

np.import_array()

cdef object as_numpy_scalar(double a, np.dtype dtype):
    """Note that the memory is interpreted to match dtype, not a cast"""
    return PyArray_Scalar(&a, dtype, None)

# def cylinear_min(cyBQM bqm, default=None):
#     if bqm.num_variables == 0:
#         if default is None:
#             raise ValueError("Argument is an empty sequence")
#         else:
#             return default

#     cdef double min_ = sys.float_info.max
#     cdef Py_ssize_t vi
#     for vi in range(bqm.bqm_.num_variables()):
#         val = bqm.bqm_.get_linear(vi)
#         if val < min_:
#             min_ = val
    
#     return min_

# def cylinear_max(cyBQM bqm, default=None):
#     if bqm.num_variables == 0:
#         if default is None:
#             raise ValueError("Argument is an empty sequence")
#         else:
#             return default

#     cdef double max_ = -sys.float_info.max
#     cdef Py_ssize_t vi
#     for vi in range(bqm.bqm_.num_variables()):
#         val = bqm.bqm_.get_linear(vi)
#         if val > max_:
#             max_ = val
    
#     return max_

# def cylinear_sum(cyBQM bqm, Bias start=0):
#     """Return the sum of the linear biases."""
#     cdef VarIndex vi
#     for vi in range(bqm.bqm_.num_variables()):
#         start += bqm.bqm_.get_linear(vi)

#     return start

# def cyquadratic_min(cyBQM bqm, default=None):
#     if bqm.num_interactions == 0:
#         if default is None:
#             raise ValueError("Argument is an empty sequence")
#         else:
#             return default

#     cdef double min_ = sys.float_info.max
#     cdef Py_ssize_t vi
#     for vi in range(bqm.bqm_.num_variables()):
#         span = bqm.bqm_.neighborhood(vi)

#         while span.first != span.second and deref(span.first).first < vi:
#             if deref(span.first).second < min_:
#                 min_ = deref(span.first).second

#             inc(span.first)

#     return min_

# def cyquadratic_max(cyBQM bqm, default=None):
#     if bqm.num_interactions == 0:
#         if default is None:
#             raise ValueError("Argument is an empty sequence")
#         else:
#             return default

#     cdef double max_ = -sys.float_info.max

#     cdef Py_ssize_t vi
#     for vi in range(bqm.bqm_.num_variables()):
#         span = bqm.bqm_.neighborhood(vi)
        
#         while span.first != span.second and deref(span.first).first < vi:
#             if deref(span.first).second > max_:
#                 max_ = deref(span.first).second

#             inc(span.first)

#     return max_

# def cyquadratic_sum(cyBQM bqm, Bias start=0):
#     """Return the sum of the quadratic biases."""
#     cdef VarIndex vi
#     for vi in range(bqm.bqm_.num_variables()):
#         span = bqm.bqm_.neighborhood(vi)
#         while span.first != span.second and deref(span.first).first < vi:
#             start += deref(span.first).second
#             inc(span.first)
    
#     return start

# def cyneighborhood_max(cyBQM bqm, object v, object default=None):
#     if not bqm.degree(v):
#         if default is None:
#             raise ValueError("Argument is an empty sequence")
#         else:
#             return default

#     cdef VarIndex vi = bqm.label_to_idx(v)

#     cdef double max_ = -sys.float_info.max

#     span = bqm.bqm_.neighborhood(vi)
#     while span.first != span.second:
#         if deref(span.first).second > max_:
#             max_ = deref(span.first).second

#         inc(span.first)

#     return max_

# def cyneighborhood_min(cyBQM bqm, object v, object default=None):
#     if not bqm.degree(v):
#         if default is None:
#             raise ValueError("Argument is an empty sequence")
#         else:
#             return default

#     cdef VarIndex vi = bqm.label_to_idx(v)

#     cdef double min_ = sys.float_info.max

#     span = bqm.bqm_.neighborhood(vi)
#     while span.first != span.second:
#         if deref(span.first).second < min_:
#             min_ = deref(span.first).second

#         inc(span.first)

#     return min_

# def cyneighborhood_sum(cyBQM bqm, object v, Bias start=0):
#     cdef VarIndex vi = bqm.label_to_idx(v)

#     span = bqm.bqm_.neighborhood(vi)
#     while span.first != span.second:
#         start += deref(span.first).second
#         inc(span.first)

#     return start

@cython.boundscheck(False)
@cython.wraparound(False)
def coo_sort(VarIndex[:] irow, VarIndex[:] icol, Bias[:] qdata):
    """Sort the COO-vectors inplace by row then by column. This function will
    not add duplicates.
    """
    cdef Py_ssize_t num_interactions = irow.shape[0]

    if icol.shape[0] != num_interactions or qdata.shape[0] != num_interactions:
        raise ValueError("vectors should all be the same length")

    # row index should be less than col index, this handles upper-triangular vs
    # lower-triangular
    cdef Py_ssize_t i
    for i in range(num_interactions):
        if irow[i] > icol[i]:
            irow[i], icol[i] = icol[i], irow[i]

    # next we get to sorting!
    quicksort_coo(irow, icol, qdata, 0, num_interactions - 1)


cdef void quicksort_coo(VarIndex[:] irow, VarIndex[:] icol, Bias[:] qdata,
                        Py_ssize_t low, Py_ssize_t high):
    # nb: high is inclusive
    cdef Py_ssize_t p
    if low < high:
        p = partition_coo(irow, icol, qdata, low, high)
        quicksort_coo(irow, icol, qdata, low, p - 1)
        quicksort_coo(irow, icol, qdata, p + 1, high)


cdef Py_ssize_t partition_coo(VarIndex[:] irow, VarIndex[:] icol, Bias[:] qdata,
                              Py_ssize_t low, Py_ssize_t high):

    # median of three pivot
    cdef Py_ssize_t mid = low + (high - low) // 2
    if less(irow, icol, mid, low):
        swap(irow, icol, qdata, low, mid)
    if less(irow, icol, high, low):
        swap(irow, icol, qdata, low, high)
    if less(irow, icol, mid, high):
        swap(irow, icol, qdata, mid, high)

    cdef Py_ssize_t pi = high    

    cdef Py_ssize_t i = low

    cdef Py_ssize_t j
    for j in range(low, high):
        if less(irow, icol, j, pi):
            # ok, smaller than pivot
            swap(irow, icol, qdata, i, j)

            i = i + 1

    swap(irow, icol, qdata, i, pi)

    return i

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint less(VarIndex[:] irow, VarIndex[:] icol,
                      Py_ssize_t a, Py_ssize_t b):
    """Return True if a < b"""
    return irow[a] < irow[b] or (irow[a] == irow[b] and icol[a] < icol[b])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void swap(VarIndex[:] irow, VarIndex[:] icol, Bias[:] qdata,
                      Py_ssize_t a, Py_ssize_t b):
    # swap the data
    irow[a], irow[b] = irow[b], irow[a]
    icol[a], icol[b] = icol[b], icol[a]
    qdata[a], qdata[b] = qdata[b], qdata[a]
