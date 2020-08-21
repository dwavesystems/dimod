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

from cython.operator cimport postincrement as inc, dereference as deref

import numpy as np

import dimod

import sys

from dimod.bqm cimport cyBQM
from dimod.bqm.common import itype, dtype
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

def cylinear_min(cyBQM bqm, default=None):
    if bqm.num_variables == 0:
        if default is None:
            raise ValueError("Argument is an empty sequence")
        else:
            return default

    cdef double min_ = sys.float_info.max
    cdef Py_ssize_t vi
    for vi in range(bqm.bqm_.num_variables()):
        val = bqm.bqm_.get_linear(vi)
        if val < min_:
            min_ = val
    
    return min_

def cylinear_max(cyBQM bqm, default=None):
    if bqm.num_variables == 0:
        if default is None:
            raise ValueError("Argument is an empty sequence")
        else:
            return default

    cdef double max_ = -sys.float_info.max
    cdef Py_ssize_t vi
    for vi in range(bqm.bqm_.num_variables()):
        val = bqm.bqm_.get_linear(vi)
        if val > max_:
            max_ = val
    
    return max_

def cylinear_sum(cyBQM bqm, Bias start=0):
    """Return the sum of the linear biases."""
    cdef VarIndex vi
    for vi in range(bqm.bqm_.num_variables()):
        start += bqm.bqm_.get_linear(vi)

    return start

def cyquadratic_min(cyBQM bqm, default=None):
    if bqm.num_interactions == 0:
        if default is None:
            raise ValueError("Argument is an empty sequence")
        else:
            return default

    cdef double min_ = sys.float_info.max
    cdef Py_ssize_t vi
    for vi in range(bqm.bqm_.num_variables()):
        span = bqm.bqm_.neighborhood(vi)

        while span.first != span.second and deref(span.first).first < vi:
            if deref(span.first).second < min_:
                min_ = deref(span.first).second

            inc(span.first)

    return min_

def cyquadratic_max(cyBQM bqm, default=None):
    if bqm.num_interactions == 0:
        if default is None:
            raise ValueError("Argument is an empty sequence")
        else:
            return default

    cdef double max_ = -sys.float_info.max

    cdef Py_ssize_t vi
    for vi in range(bqm.bqm_.num_variables()):
        span = bqm.bqm_.neighborhood(vi)
        
        while span.first != span.second and deref(span.first).first < vi:
            if deref(span.first).second > max_:
                max_ = deref(span.first).second

            inc(span.first)

    return max_

def cyquadratic_sum(cyBQM bqm, Bias start=0):
    """Return the sum of the quadratic biases."""
    cdef VarIndex vi
    for vi in range(bqm.bqm_.num_variables()):
        span = bqm.bqm_.neighborhood(vi)
        while span.first != span.second and deref(span.first).first < vi:
            start += deref(span.first).second
            inc(span.first)
    
    return start

def cyneighborhood_max(cyBQM bqm, object v, object default=None):
    if not bqm.degree(v):
        if default is None:
            raise ValueError("Argument is an empty sequence")
        else:
            return default

    cdef VarIndex vi = bqm.label_to_idx(v)

    cdef double max_ = -sys.float_info.max

    span = bqm.bqm_.neighborhood(vi)
    while span.first != span.second:
        if deref(span.first).second > max_:
            max_ = deref(span.first).second

        inc(span.first)

    return max_

def cyneighborhood_min(cyBQM bqm, object v, object default=None):
    if not bqm.degree(v):
        if default is None:
            raise ValueError("Argument is an empty sequence")
        else:
            return default

    cdef VarIndex vi = bqm.label_to_idx(v)

    cdef double min_ = sys.float_info.max

    span = bqm.bqm_.neighborhood(vi)
    while span.first != span.second:
        if deref(span.first).second < min_:
            min_ = deref(span.first).second

        inc(span.first)

    return min_

def cyneighborhood_sum(cyBQM bqm, object v, Bias start=0):
    cdef VarIndex vi = bqm.label_to_idx(v)

    span = bqm.bqm_.neighborhood(vi)
    while span.first != span.second:
        start += deref(span.first).second
        inc(span.first)

    return start

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


@cython.boundscheck(False)
@cython.wraparound(False)
def cyenergies(cyBQM bqm, samples_like):
    """Determine the energies of the given samples.

    Args:
        bqm (cybqm):
            A cybqm.

        samples_like (samples_like):
            A collection of raw samples. `samples_like` is an extension of
            NumPy's array_like structure. See :func:`.as_samples`.

    Returns:
        :obj:`numpy.ndarray`: The energies.

    """
    samples, labels = dimod.as_samples(samples_like, dtype=np.int8)

    cdef np.int8_t[:, :] samples_view = samples

    cdef Py_ssize_t num_samples = samples_view.shape[0]
    cdef Py_ssize_t num_variables = samples_view.shape[1]

    if num_variables != bqm.bqm_.num_variables():
        raise ValueError("inconsistent number of variables")
    if num_variables != len(labels):
        # an internal error to as_samples. We do this check because
        # the boundscheck is off
        msg = "as_samples returned an inconsistent samples/variables"
        raise RuntimeError(msg)

    # we want a map such that bqm_to_sample[vi] = si
    cdef VarIndex[:] bqm_to_sample = np.empty(num_variables, dtype=itype)
    cdef VarIndex ui, vi, si
    for si in range(num_variables):
        v = labels[si]  # python label
        bqm_to_sample[bqm.label_to_idx(v)] = si

    # now calculate the energies
    energies = np.zeros(num_samples, dtype=dtype)
    cdef Bias[:] energies_view = energies

    cdef np.int8_t uspin, vspin
    for si in range(num_samples):

        energies_view[si] += bqm.offset_

        for ui in range(num_variables):
            uspin = samples_view[si, bqm_to_sample[ui]]

            energies_view[si] += bqm.bqm_.get_linear(ui) * uspin

            span = bqm.bqm_.neighborhood(ui)
            while span.first != span.second and deref(span.first).first < ui:
                vi = deref(span.first).first

                vspin = samples_view[si, bqm_to_sample[vi]]

                energies_view[si] += uspin * vspin * deref(span.first).second

                inc(span.first)

    return energies


def cyrelabel(cyBQM bqm, mapping, inplace=True):
    if not inplace:
        return cyrelabel(bqm.copy(), mapping, inplace=True)

    # in the future we could maybe do something that doesn't require a copy
    existing = set(bqm.iter_variables())

    for submap in dimod.utilities.iter_safe_relabels(mapping, existing):
        
        for old, new in submap.items():
            if old == new:
                continue

            vi = bqm._label_to_idx.pop(old, old)

            if new != vi:
                bqm._label_to_idx[new] = vi
                bqm._idx_to_label[vi] = new  # overwrites old vi if it's there
            else:
                bqm._idx_to_label.pop(vi, None)  # remove old reference

    return bqm


def cyrelabel_variables_as_integers(cyBQM bqm, inplace=True):
    """Relabel the variables in the BQM to integers.

    Note that this fuction uses the natural labelling of the underlying c++
    objects.
    """
    if not inplace:
        return cyrelabel_variables_as_integers(bqm.copy(), inplace=True)

    inverse = bqm._idx_to_label.copy()  # everything is hashable so no deep copy
    
    bqm._label_to_idx.clear()
    bqm._idx_to_label.clear()

    return bqm, inverse


@cython.boundscheck
@cython.wraparound
def ilinear_biases(cyBQM bqm):
    """Get the linear biases as well as the neighborhood indices."""

    cdef Py_ssize_t numvar = bqm.bqm_.num_variables()

    dtype = np.dtype([('ni', bqm.ntype), ('b', bqm.dtype)], align=False)
    ldata = np.empty(numvar, dtype=dtype)

    if numvar == 0:
        return ldata

    # if in the future the BQM does not have fixed dtypes, these will error
    cdef size_t[:] neighbors_view = ldata['ni']
    cdef Bias[:] bias_view = ldata['b']

    neighbors_view[0] = 0

    cdef VarIndex vi
    for vi in range(numvar):
        if vi + 1 < numvar:
            neighbors_view[vi + 1] = neighbors_view[vi] + bqm.bqm_.degree(vi)

        bias_view[vi] = bqm.bqm_.get_linear(vi)

    return ldata


@cython.boundscheck(False)
@cython.wraparound(False)
def ineighborhood(cyBQM bqm, VarIndex ui):
    """Get the neighborhood (as a struct array) of variable ui.

    Note that this function is in terms of the underlying index, NOT the
    labels.

    Returns:
        A numpy struct array with two fields, `'vi'` corresponding to the
        neighbours of `ui` and `'b'` corresponding to their associated
        quadratic biases.

    """

    if ui >= bqm.bqm_.num_variables():
        raise ValueError("out of range variable, {!r}".format(ui))

    cdef Py_ssize_t d = bqm.bqm_.degree(ui)

    dtype = np.dtype([('ui', bqm.itype), ('b', bqm.dtype)], align=False)
    neighbors = np.empty(d, dtype=dtype)

    # if in the future the BQM does not have fixed dtypes, these will error
    cdef VarIndex[:] index_view = neighbors['ui']
    cdef Bias[:] bias_view = neighbors['b']

    span = bqm.bqm_.neighborhood(ui)

    cdef Py_ssize_t i = 0
    while span.first != span.second:

        index_view[i] = deref(span.first).first
        bias_view[i] = deref(span.first).second

        i += 1
        inc(span.first)

    return neighbors
