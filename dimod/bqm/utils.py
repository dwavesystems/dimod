import numpy as np


def energies(bqm, samples, _use_cpp_ext=True):
    # use the faster c++ extension if it's available

    if hasattr(samples, "dtype"):
        samples = np.asarray(samples, dtype=samples.dtype)  # handle subclasses
    else:
        samples = np.asarray(samples, dtype=np.int8)

    if _use_cpp_ext:
        try:
            from dimod.bqm._utils import fast_energy
            return fast_energy(bqm.offset, bqm.ldata, bqm.irow, bqm.icol, bqm.qdata, samples)
        except ImportError:
            # no c++ extension
            pass
        # except TypeError:
        #     # dtype is the wrong type
        #     pass

    try:
        num_samples, num_variables = samples.shape
    except ValueError:
        raise ValueError("samples should be a square array where each row is a sample")

    energy = np.full(num_samples, bqm.offset)  # offset

    energy += samples.dot(bqm.ldata)  # linear

    energy += (samples[:, bqm.irow]*samples[:, bqm.icol]).dot(bqm.qdata)  # quadratic

    return energy


def reduce_coo(row, col, data, dtype=None, index_dtype=None, copy=True):
    """
    """
    # method adapted from scipy's coo_matrix
    #
    # Copyright (c) 2001, 2002 Enthought, Inc.
    # All rights reserved.
    #
    # Copyright (c) 2003-2017 SciPy Developers.
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are met:
    #
    #   a. Redistributions of source code must retain the above copyright notice,
    #      this list of conditions and the following disclaimer.
    #   b. Redistributions in binary form must reproduce the above copyright
    #      notice, this list of conditions and the following disclaimer in the
    #      documentation and/or other materials provided with the distribution.
    #   c. Neither the name of Enthought nor the names of the SciPy Developers
    #      may be used to endorse or promote products derived from this software
    #      without specific prior written permission.
    #
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    # ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
    # BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
    # OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
    # SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
    # INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
    # CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
    # ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
    # THE POSSIBILITY OF SUCH DAMAGE.
    #

    row = np.array(row, dtype=index_dtype, copy=copy)
    col = np.array(col, dtype=index_dtype, copy=copy)
    data = np.array(data, dtype=dtype, copy=copy)

    if row.ndim != 1 or col.ndim != 1 or data.ndim != 1 or len(row) != len(col) or len(col) != len(data):
        raise ValueError("row, col and data should all be vectors of equal length")

    if len(row) == 0:
        # empty arrays are already sorted
        return row, col, data

    # row index should be less than col index, this handles upper-triangular vs lower-triangular
    swaps = row > col
    if swaps.any():
        # in-place
        row[swaps], col[swaps] = col[swaps], row[swaps]

    # sort lexigraphically
    order = np.lexsort((row, col))
    if not (order == range(len(order))).all():
        # copy
        row = row[order]
        col = col[order]
        data = data[order]

    # reduce unique
    unique = ((row[1:] != row[:-1]) | (col[1:] != col[:-1]))
    if not unique.all():
        # copy
        unique = np.append(True, unique)

        row = row[unique]
        col = col[unique]

        unique_idxs, = np.nonzero(unique)
        data = np.add.reduceat(data, unique_idxs, dtype=data.dtype)

    return row, col, data
