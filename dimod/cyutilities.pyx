# Copyright 2021 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Un_lt required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import typing

cimport cython

import numpy as np
cimport numpy as np

from dimod.libcpp.vartypes cimport vartype_info as cppvartype_info
from dimod.typing import VartypeLike
from dimod.vartypes import as_vartype, Vartype

__all__ = ['vartype_info']


cdef extern from "numpy/arrayobject.h":
    ctypedef struct PyArray_Descr:
        pass

    object PyArray_Scalar(void*, PyArray_Descr*, object)

np.import_array()  # needed for PyArray_Scalar

# preconstruct these dtypes for speed, we could possibly improve it more
# by casting to PyArray_Descr but I had trouble getting that to work
cdef object _float32_dtype = np.dtype(np.float32)
cdef object _float64_dtype = np.dtype(np.float64)

cdef object as_numpy_float(cython.floating a):
    if cython.floating == double:
        return PyArray_Scalar(&a, <PyArray_Descr*>_float64_dtype, None)
    elif cython.floating == float:
        return PyArray_Scalar(&a, <PyArray_Descr*>_float32_dtype, None)
    else:
        raise NotImplementedError

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline bint _lt(Integer[:] row, Integer[:] col, Py_ssize_t i, Py_ssize_t j):
    """Return (row[i], col[i]) < (row[j], col[j])."""
    return row[i] < row[j] or (row[i] == row[j] and col[i] < col[j])

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _swap(Integer[:] row, Integer[:] col, cython.floating[:] data,
                       Py_ssize_t i, Py_ssize_t j):
    """Swap the values in the arrays at i, j"""
    row[i], row[j] = row[j], row[i]
    col[i], col[j] = col[j], col[i]
    data[i], data[j] = data[j], data[i]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef Py_ssize_t coo_sort(Integer[:] row, Integer[:] col, cython.floating[:] data) except -1:
    """Sort COO arrays in-place by row then by column."""
    cdef Py_ssize_t num_interactions = row.shape[0]
    if col.shape[0] != num_interactions or data.shape[0] != num_interactions:
        raise ValueError("vectors should all be the same length")

    # row index should be _lt than col index, this handles upper-triangular vs
    # lower-triangular
    cdef Py_ssize_t i
    for i in range(num_interactions):
        if row[i] > col[i]:
            row[i], col[i] = col[i], row[i]

    quicksort_coo(row, col, data, 0, num_interactions - 1)


cdef void quicksort_coo(Integer[:] row, Integer[:] col, cython.floating[:] data,
                        Py_ssize_t low, Py_ssize_t high):
    # nb: high is inclusive
    cdef Py_ssize_t p
    if low < high:
        p = partition_coo(row, col, data, low, high)
        quicksort_coo(row, col, data, low, p - 1)
        quicksort_coo(row, col, data, p + 1, high)


cdef Py_ssize_t partition_coo(Integer[:] row, Integer[:] col, cython.floating[:] data,
                              Py_ssize_t low, Py_ssize_t high):

    # median of three pivot
    cdef Py_ssize_t mid = low + (high - low) // 2
    if _lt(row, col, mid, low):
        _swap(row, col, data, low, mid)
    if _lt(row, col, high, low):
        _swap(row, col, data, low, high)
    if _lt(row, col, mid, high):
        _swap(row, col, data, mid, high)

    cdef Py_ssize_t pi = high    

    cdef Py_ssize_t i = low

    cdef Py_ssize_t j
    for j in range(low, high):
        if _lt(row, col, j, pi):
            # ok, smaller than pivot
            _swap(row, col, data, i, j)

            i += 1

    _swap(row, col, data, i, pi)

    return i


cdef cppVartype cppvartype(vartype) except? cppVartype.SPIN:
    if vartype is Vartype.SPIN:
        return cppVartype.SPIN
    elif vartype is Vartype.BINARY:
        return cppVartype.BINARY
    elif vartype is Vartype.INTEGER:
        return cppVartype.INTEGER
    elif vartype is Vartype.REAL:
        return cppVartype.REAL
    else:
        raise TypeError(f"unexpected vartype {vartype!r}")


# todo: type annotations, fix docs. This needs a followup PR
def vartype_info(vartype, dtype=np.float64):
    """Information about the variable bounds by variable type.

    Args:
        vartype:
            Variable type. One of:

                * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
                * :class:`~dimod.Vartype.INTEGER`, ``'INTEGER'``

        dtype: One of :class:`~numpy.float64` and :class:`~numpy.float32`.

    Returns:
        A named tuple with ``default_min``, ``default_max``, ``min``, and
        ``max`` fields. These specify the default and largest bounds for
        the given variable type.

    """
    cdef cppVartype vt = cppvartype(as_vartype(vartype, extended=True))

    Info = typing.NamedTuple("VartypeLimits",
                      (("default_min", float),
                       ("default_max", float),
                       ("min", float),
                       ("max", float),
                      ))

    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.float32):
        return Info(
            as_numpy_float(cppvartype_info[np.float32_t].default_min(vt)),
            as_numpy_float(cppvartype_info[np.float32_t].default_max(vt)),
            as_numpy_float(cppvartype_info[np.float32_t].min(vt)),
            as_numpy_float(cppvartype_info[np.float32_t].max(vt)),
            )
    elif dtype == np.dtype(np.float64):
        return Info(
            as_numpy_float(cppvartype_info[np.float64_t].default_min(vt)),
            as_numpy_float(cppvartype_info[np.float64_t].default_max(vt)),
            as_numpy_float(cppvartype_info[np.float64_t].min(vt)),
            as_numpy_float(cppvartype_info[np.float64_t].max(vt)),
            )
    else:
        raise ValueError("only supports np.float64 and np.float32 dtypes")
