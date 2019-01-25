# distutils: language = c++
# cython: language_level = 3
# =============================================================================
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
def fast_energy(bias_type[:] ldata,  # linear
                index_type[:] irow, index_type[:] icol, bias_type[:] qdata,  # quadratic
                bias_type offset,
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

    energies = np.empty(num_samples, dtype=dtype)
    cdef bias_type[::1] energies_view = energies

    cdef Py_ssize_t row

    # we can use static schedule because each energy calculation takes
    # approximately the same amount of time
    for row in prange(num_samples, nogil=True, schedule='static'):
        energies_view[row] = offset + _energy(ldata,
                                              irow, icol, qdata,
                                              samples[row, :],
                                              num_variables,
                                              num_interactions)

    return energies
