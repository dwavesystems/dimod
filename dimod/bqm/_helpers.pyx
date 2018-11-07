# distutils: language = c++
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: language_level = 3

cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as np

ctypedef fused samples_type:
    np.npy_int8
    np.npy_int16
    np.npy_int32
    np.npy_int64
    np.npy_float32
    np.npy_float64

ctypedef fused bias_type:
    np.npy_float32
    np.npy_float64

ctypedef fused index_type:
    np.npy_int8
    np.npy_int16
    np.npy_int32
    np.npy_int64


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bias_type _energy(bias_type[:] ldata,  # linear
                       index_type[:] irow, index_type[:] icol, bias_type[:] qdata,  # quadratic
                       samples_type[:] sample,
                       Py_ssize_t num_variables,
                       Py_ssize_t num_interactions) nogil:
    cdef bias_type energy = 0

    cdef Py_ssize_t li, qi

    for li in range(num_variables):
        energy = energy + sample[li] * ldata[li]

    for qi in range(num_interactions):
        energy = energy + sample[irow[qi]] * sample[icol[qi]] * qdata[qi]

    return energy


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_energy(offset,
                bias_type[:] ldata,  # linear
                index_type[:] irow, index_type[:] icol, bias_type[:] qdata,  # quadratic
                samples_type[:, :] samples):

    cdef Py_ssize_t num_samples = samples.shape[0]
    cdef Py_ssize_t num_variables = samples.shape[1]
    cdef Py_ssize_t num_interactions = irow.shape[0]

    if bias_type is np.npy_float32:
        dtype = np.float32
    elif bias_type is np.npy_float64:
        dtype = np.float64
    else:
        raise RuntimeError

    energies = np.full(num_samples, offset, dtype=dtype)
    cdef bias_type[::1] energies_view = energies

    cdef Py_ssize_t row

    for row in prange(num_samples, nogil=True):
        energies_view[row] = energies_view[row] + _energy(ldata,
                                                          irow, icol, qdata,
                                                          samples[row, :],
                                                          num_variables,
                                                          num_interactions)

    return energies
