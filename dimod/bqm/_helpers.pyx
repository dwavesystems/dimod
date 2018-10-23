# distutils: language = c++
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

from dimod.bqm.fast_bqm import FastBQM
from dimod.sampleset import as_samples

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


def fast_energy(bqm, samples_like):
    if not isinstance(bqm, FastBQM):
        raise TypeError("bqm should be a FastBQM")

    samples, sample_labels = as_samples(samples_like)

    ldata = bqm.ldata
    irow = bqm.irow
    icol = bqm.icol
    variables = bqm.variables

    if sample_labels != variables:
        # We assume in general that it's better to reorder the biases rather
        # than the samples as the samples should be a larger copy normally

        # we could also cython this at some point

        # variables.index is fast because variables is a VariableView object
        bqm_to_samples = np.fromiter((variables.index(s) for s in sample_labels),
                            dtype=irow.dtype, count=len(sample_labels))
        ldata = ldata[bqm_to_samples]

        samples_to_bqm = np.empty(len(bqm_to_samples), dtype=irow.dtype)
        for si, v in enumerate(sample_labels):
            samples_to_bqm[variables.index(v)] = si

        irow = samples_to_bqm[irow]
        icol = samples_to_bqm[icol]


    return _fast_energy(bqm.offset, ldata, irow, icol, bqm.qdata, samples)

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
def _fast_energy(offset,
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

    energies = np.ones(num_samples, dtype=dtype) * offset
    cdef bias_type[::1] energies_view = energies

    cdef Py_ssize_t row

    for row in prange(num_samples, nogil=True):
        energies_view[row] = energies_view[row] + _energy(ldata,
                                                          irow, icol, qdata,
                                                          samples[row, :],
                                                          num_variables,
                                                          num_interactions)

    return energies
