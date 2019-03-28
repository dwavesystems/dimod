# cython: language_level = 3, boundscheck=False, wraparound=False
# distutils: language = c++

import numpy as np
cimport numpy as np

ctypedef fused index_type:
    np.npy_int8
    np.npy_int16
    np.npy_int32
    np.npy_int64
    np.npy_uint8
    np.npy_uint16
    np.npy_uint32
    np.npy_uint64

ctypedef fused bias_type:
    np.npy_float32
    np.npy_float64


def coo_sort(index_type[:] irow, index_type[:] icol, bias_type[:] qdata):
    
    cdef Py_ssize_t num_interactions = irow.shape[0]

    assert icol.shape[0] == num_interactions
    assert qdata.shape[0] == num_interactions


    cdef Py_ssize_t i

    # row index should be less than col index, this handles upper-triangular vs
    # lower-triangular
    for i in range(num_interactions):
        if irow[i] > icol[i]:
            irow[i], icol[i] = icol[i], irow[i]

    # next we get to sorting!
    cdef Py_ssize_t low = 0
    quicksort_coo(irow, icol, qdata, low, num_interactions - 1)

    return np.asarray(irow), np.asarray(icol), np.asarray(qdata)


cdef void quicksort_coo(index_type[:] irow, index_type[:] icol, bias_type[:] qdata, Py_ssize_t low, Py_ssize_t high):
    # nb: high is inclusive
    cdef Py_ssize_t p
    if low < high:
        p = partition_coo(irow, icol, qdata, low, high)
        quicksort_coo(irow, icol, qdata, low, p - 1)
        quicksort_coo(irow, icol, qdata, p + 1, high)


cdef Py_ssize_t partition_coo(index_type[:] irow, index_type[:] icol, bias_type[:] qdata,
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

cdef inline bint less(index_type[:] irow, index_type[:] icol, Py_ssize_t a, Py_ssize_t b):
    """Return True if a < b"""
    return irow[a] < irow[b] or (irow[a] == irow[b] and icol[a] < icol[b])

cdef inline void swap(index_type[:] irow, index_type[:] icol, bias_type[:] qdata,
               Py_ssize_t a, Py_ssize_t b):
    # swap the data
    irow[a], irow[b] = irow[b], irow[a]
    icol[a], icol[b] = icol[b], icol[a]
    qdata[a], qdata[b] = qdata[b], qdata[a]
