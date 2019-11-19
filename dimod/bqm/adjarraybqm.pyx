# distutils: language = c++
# cython: language_level=3
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

import io
import struct

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from numbers import Integral

cimport cython

from libc.string cimport memcpy
from cython.operator cimport postincrement as inc, dereference as deref
from cython.view cimport array as cvarray

import numpy as np

from dimod.bqm cimport cyAdjVectorBQM
from dimod.bqm.cppbqm cimport (num_variables,
                               num_interactions,
                               get_linear,
                               get_quadratic,
                               degree,
                               neighborhood,
                               set_linear,
                               set_quadratic,
                               )
from dimod.bqm.utils cimport as_numpy_scalar
from dimod.core.bqm import BQM
from dimod.vartypes import as_vartype


class FileView(io.RawIOBase):
    """

    Format specification:

    The first 5 bytes are a magic string: exactly "DIMOD".

    The next 1 byte is an unsigned byte: the major version of the file format.

    The next 1 byte is an unsigned byte: the minor version of the file format.

    The next 4 bytes form a little-endian unsigned int, the length of the header
    data HEADER_LEN.

    The next HEADER_LEN bytes form the header data. TODO, though let's do the
    "divisible by 16" thing.

    See also
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.format.html

    """

    def __init__(self, bqm):
        self.bqm = bqm
        # todo: increment viewcount

        self.pos = 0

        # # we'll need to know the degree of each of the variables
        # self.degrees = np.empty(len(bqm), dtype=np.intc)
        # self.degree_pos = 0  # location that has been filled in degrees

        # set up the header
        prefix = b'DIMOD'
        version = bytes([1, 0])  # version 1.0

        header_data = b'dummy data'  # todo

        header_len = struct.pack('<I', len(header_data))

        header = prefix + version + header_len + header_data

        # pad the header so that it's length is divisible by 16
        self.header = header + b' '*(16 - len(header) % 16)

    def readinto(self, char[:] buff):
        """Accepts a writeable buffer, fills it and returns the num filled.
        Return 0 once it's done.
        """
        print('trying to readinto {} bytes'.format(len(buff)))

        cdef int pos = self.pos  # convert to C space
        cdef int start = pos
        cdef int end = pos + len(buff)  # the maximum
        cdef int num_written = 0

        # determine if there is any header to read

        cdef const unsigned char[:] header = self.header  # view
        while pos < len(header) and pos < end:
            buff[pos] = header[pos]
            pos += 1
        print('{} header bytes written'.format(pos - start))


        pos += self._linear_readinto(buff[pos:], pos - len(header))
        pos += self._quadratic_readinto(buff[pos:], pos - len(header) - self.bqm.num_variables*(sizeof(size_t) + sizeof(Bias)))

        self.pos = pos
        return pos - start

    def _linear_readinto(self, char[:] buff, int pos):
        # read the linear biases into the given buffer, note that this does not
        # increment the position on the object itself
        # pos is relative to the beginning of the linear biases
        if pos < 0:
            raise ValueError("negative position")

        if len(buff) == 0:
            return 0

        cdef cyAdjArrayBQM bqm = self.bqm

        cdef Py_ssize_t bi = 0  # location in buffer
        cdef Py_ssize_t end = len(buff)

        cdef Py_ssize_t step_size = sizeof(size_t) + sizeof(Bias)

        if pos >= step_size*num_variables(bqm.adj_):
            # we're already past the linear stuff
            return 0

        if pos % step_size or len(buff) < step_size:
            # need to handle partial, either we're part-way through or we don't
            # have room for at least one (neighbor, bias) pair
            raise NotImplementedError

        cdef VarIndex vi
        for vi in range(pos // step_size, num_variables(bqm.adj_)):
            if bi + step_size >= end:
                break

            # this is specific to adjarray, will want to generalize later
            memcpy(&buff[bi], &bqm.adj_.first[vi].first, sizeof(size_t))
            bi += sizeof(size_t)
            memcpy(&buff[bi], &bqm.adj_.first[vi].second, sizeof(Bias))            
            bi += sizeof(Bias)

        print('{} linear bytes written'.format(bi))

        return bi  # number of bytes written

    def _quadratic_readinto(self, char[:] buff, int pos):
        # pos is relative to start of quadratic
        if pos < 0:
            raise ValueError

        cdef cyAdjArrayBQM bqm = self.bqm
        cdef Py_ssize_t step_size = sizeof(VarIndex) + sizeof(Bias)

        if len(buff) == 0 or pos >= 2*step_size*num_interactions(bqm.adj_):
            # either there is no room in the buffer or the position is past all
            # the data we want to write, in either case we don't need to do
            # anything
            return 0

        cdef Py_ssize_t bi = 0  # location in buffer
        cdef Py_ssize_t end = len(buff)

        if pos % step_size or len(buff) < step_size:
            # need to handle partial, either we're part-way through or we don't
            # have room for at least one (neighbor, bias) pair
            raise NotImplementedError

        # determine which in-variable we're on based on position, this is very
        # specific to adjarray, we'll want to generalize later
        cdef Py_ssize_t qi
        for qi in range(pos // step_size, 2*num_interactions(bqm.adj_)):
            if bi + step_size >= end:
                break

            memcpy(&buff[bi], &bqm.adj_.second[qi].first, sizeof(VarIndex))
            bi += sizeof(VarIndex)
            memcpy(&buff[bi], &bqm.adj_.second[qi].second, sizeof(Bias))            
            bi += sizeof(Bias)

        print('{} quadratic bytes written'.format(bi))

        return bi  # number of bytes written

    def close(self):
        # todo: decrement viewcount
        super(FileView, self).close()


# developer note: we use a function rather than a method because we want to
# use nogil
# developer note: we probably want to make this a template function in c++
# so we can determine the return type. For now we'll match Bias
@cython.boundscheck(False)
@cython.wraparound(False)
cdef Bias energy(vector[pair[size_t, Bias]] invars,
                 vector[pair[VarIndex, Bias]] outvars,
                 SampleVar[:] sample):
    """Calculate the energy of a single sample"""
    cdef Bias energy = 0

    if invars.size() == 0:
        return energy

    cdef Bias b
    cdef size_t u, v, qi
    cdef size_t qimax = outvars.size()

    # iterate backwards because it makes finding the neighbourhoods slightly
    # nicer and the order does not matter.
    # we could possibly parallelize this step (cython knows how to use +=)
    for u in reversed(range(0, invars.size())):  # throws a comp warning
        # linear bias
        energy = energy + invars[u].second * sample[u]

        # quadratic bias
        for qi in range(invars[u].first, qimax):
            v = outvars[qi].first

            if v > u:
                # we're only interested in upper-triangular
                break

            b = outvars[qi].second
            energy = energy + b * sample[u] * sample[v]

        qimax = invars[u].first

    return energy


@cython.embedsignature(True)
cdef class cyAdjArrayBQM:
    """

    This can be instantiated in several ways:

        AdjArrayBQM(vartype)
            Creates an empty binary quadratic model

        AdjArrayBQM((linear, [quadratic, [offset]]))
            Where linear, quadratic are:
                dict[int, bias]
                sequence[bias]  *NOT IMPLEMENTED YET*

        AdjArrayBQM(bqm)
            Where bqm is another binary quadratic model (equivalent to
            bqm.to_adjarray())
            *NOT IMPLEMENTED YET*

        AdjArrayBQM(D, vartype)
            Where D is a dense matrix D

        AdjArrayBQM(n, vartype)
            Where n is the number of nodes.

    """

    def __cinit__(self, *args, **kwargs):
        # Developer note: if VarIndex or Bias were fused types, we would want
        # to do a type check here but since they are fixed...
        self.dtype = np.dtype(np.double)
        self.index_dtype = np.dtype(np.uintc)

        # otherwise these would be None
        self._label_to_idx = dict()
        self._idx_to_label = dict()


    def __init__(self, object obj=0, object vartype=None):

        # handle the case where only vartype is given
        if vartype is None:
            try:
                vartype = obj.vartype
            except AttributeError:
                vartype = obj
                obj = 0
        self.vartype = as_vartype(vartype)
        
        cdef Bias [:, :] D  # in case it's dense
        cdef size_t num_variables, num_interactions, degree
        cdef VarIndex ui, vi
        cdef Bias b
        cdef cyAdjArrayBQM other

        if isinstance(obj, Integral):
            self.adj_.first.resize(obj)
        elif isinstance(obj, tuple):
            self.__init__(cyAdjVectorBQM(obj, vartype))  # via the map version
        elif hasattr(obj, "to_adjarray"):
            # this is not very elegent...
            other = obj.to_adjarray()
            self.adj_ = other.adj_  # is this a copy? We probably want to move
            self.offset_ = other.offset_
            self._label_to_idx = other._label_to_idx
            self._idx_to_label = other._idx_to_label
        else:
            # assume it's dense

            D = np.atleast_2d(np.asarray(obj, dtype=self.dtype))

            num_variables = D.shape[0]

            if D.ndim != 2 or num_variables != D.shape[1]:
                raise ValueError("expected dense to be a 2 dim square array")

            self.adj_.first.resize(num_variables)

            # we could grow the vectors going through it one at a time, but
            # in the interest of future-proofing we will go through once,
            # resize the adj_.second then go through it again to fill

            # figure out the degree of each variable and consequently the
            # number of interactions
            num_interactions = 0  # 2x num_interactions because count degree
            for ui in range(num_variables):
                degree = 0
                for vi in range(num_variables):
                    if ui != vi and (D[vi, ui] or D[ui, vi]):
                        degree += 1

                if ui < num_variables - 1:
                    self.adj_.first[ui + 1].first = degree + self.adj_.first[ui].first

                num_interactions += degree

            self.adj_.second.resize(num_interactions)

            # todo: fix this, we're assigning twice
            for ui in range(num_variables):
                degree = 0
                for vi in range(num_variables):
                    if ui == vi:
                        self.adj_.first[ui].second = D[ui, vi]
                    elif D[vi, ui] or D[ui, vi]:
                        self.adj_.second[self.adj_.first[ui].first + degree].first = vi
                        self.adj_.second[self.adj_.first[ui].first + degree].second = D[vi, ui] + D[ui, vi]
                        degree += 1

    @property
    def num_variables(self):
        return num_variables(self.adj_)

    @property
    def num_interactions(self):
        """int: The number of interactions in the model."""
        return num_interactions(self.adj_)

    @property
    def offset(self):
        return as_numpy_scalar(self.offset_, self.dtype)

    @offset.setter
    def offset(self, Bias offset):
        self.offset_ = offset

    cdef VarIndex label_to_idx(self, object v) except *:
        """Get the index in the underlying array from the python label."""
        cdef VarIndex vi

        try:
            if not self._label_to_idx:
                # there are no arbitrary labels so v must be integer labelled
                vi = v
            elif v in self._idx_to_label:
                # v is an integer label that has been overwritten
                vi = self._label_to_idx[v]
            else:
                vi = self._label_to_idx.get(v, v)
        except (OverflowError, TypeError, KeyError) as ex:
            raise ValueError("{} is not a variable".format(v)) from ex

        if vi < 0 or vi >= num_variables(self.adj_):
            raise ValueError("{} is not a variable".format(v))

        return vi

    def degree(self, object v):
        cdef VarIndex vi = self.label_to_idx(v)
        return degree(self.adj_, vi)

    # todo: overwrite degrees

    def iter_linear(self):
        cdef VarIndex vi
        cdef object v
        cdef Bias b

        for vi in range(num_variables(self.adj_)):
            v = self._idx_to_label.get(vi, vi)
            b = self.adj_.first[vi].second

            yield v, as_numpy_scalar(b, self.dtype)

    def iter_quadratic(self, object variables=None):

        cdef VarIndex ui, vi  # c indices
        cdef object u, v  # python labels
        cdef Bias b

        cdef pair[vector[pair[VarIndex, Bias]].const_iterator,
                  vector[pair[VarIndex, Bias]].const_iterator] it_eit
        cdef vector[pair[VarIndex, Bias]].const_iterator it, eit

        if variables is None:
            # in the case that variables is unlabelled we can speed things up
            # by just walking through the range
            for ui in range(num_variables(self.adj_)):
                it_eit = neighborhood(self.adj_, ui)
                it = it_eit.first
                eit = it_eit.second

                u = self._idx_to_label.get(ui, ui)

                while it != eit:
                    vi = deref(it).first
                    b = deref(it).second
                    if vi > ui:  # have not already visited
                        v = self._idx_to_label.get(vi, vi)
                        yield u, v, as_numpy_scalar(b, self.dtype)

                    inc(it)
        elif self.has_variable(variables):
            yield from self.iter_quadratic([variables])
        else:
            seen = set()
            for u in variables:
                ui = self.label_to_idx(u)
                seen.add(u)

                it_eit = neighborhood(self.adj_, ui)
                it = it_eit.first
                eit = it_eit.second

                while it != eit:
                    vi = deref(it).first
                    b = deref(it).second

                    v = self._idx_to_label.get(vi, vi)

                    if v not in seen:
                        yield u, v, as_numpy_scalar(b, self.dtype)

                    inc(it)

    def get_linear(self, object v):
        return as_numpy_scalar(get_linear(self.adj_, self.label_to_idx(v)),
                               self.dtype)

    def get_quadratic(self, object u, object v):

        if u == v:
            raise ValueError('No interaction between {} and {}'.format(u, v))
        cdef VarIndex ui = self.label_to_idx(u)
        cdef VarIndex vi = self.label_to_idx(v)

        cdef pair[Bias, bool] out = get_quadratic(self.adj_, ui, vi)

        if not out.second:
            raise ValueError('No interaction between {} and {}'.format(u, v))

        return as_numpy_scalar(out.first, self.dtype)

    def set_linear(self, object v, Bias b):
        set_linear(self.adj_, self.label_to_idx(v), b)

    def set_quadratic(self, object u, object v, Bias b):

        if u == v:
            raise ValueError('No interaction between {} and {}'.format(u, v))
        cdef VarIndex ui = self.label_to_idx(u)
        cdef VarIndex vi = self.label_to_idx(v)

        cdef bool isset = set_quadratic(self.adj_, ui, vi, b)

        if not isset:
            raise ValueError('No interaction between {} and {}'.format(u, v))

    # note that this is identical to the implemenation in shapeablebqm.pyx.src
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def to_coo(self):
        cdef Py_ssize_t nv = num_variables(self.adj_)
        cdef Py_ssize_t ni = num_interactions(self.adj_)

        # numpy arrays, we will return these
        ldata = np.empty(nv, dtype=self.dtype)
        irow = np.empty(ni, dtype=self.index_dtype)
        icol = np.empty(ni, dtype=self.index_dtype)
        qdata = np.empty(ni, dtype=self.dtype)

        # views into the numpy arrays for faster cython access
        cdef Bias[:] ldata_view = ldata
        cdef VarIndex[:] irow_view = irow
        cdef VarIndex[:] icol_view = icol
        cdef Bias[:] qdata_view = qdata

        # types needed for the loop
        cdef pair[NeighborIterator, NeighborIterator] span
        cdef VarIndex vi
        cdef Py_ssize_t qi = 0  # index in the quadratic arrays

        for vi in range(nv):
            ldata_view[vi] = get_linear(self.adj_, vi)

            # The last argument indicates we should only iterate over the
            # neighbours that have higher index than vi
            span = neighborhood(self.adj_, vi, True)

            while span.first != span.second:
                irow_view[qi] = vi
                icol_view[qi] = deref(span.first).first
                qdata_view[qi] = deref(span.first).second

                qi += 1
                inc(span.first)

        # all python objects
        labels = [self._idx_to_label.get(v, v) for v in range(nv)]

        return ldata, (irow, icol, qdata), self.offset, labels

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def energies(self, SampleVar[:, :] samples):
        cdef size_t num_samples = samples.shape[0]

        if samples.shape[1] != len(self):
            raise ValueError("Mismatched variables")

        # type is hardcoded for now
        energies = np.empty(num_samples, dtype=self.dtype)  # gil
        cdef Bias[::1] energies_view = energies

        # todo: prange and nogil, we can use static schedule because the
        # calculation should be the same for each sample.
        # See https://github.com/dwavesystems/dimod/pull/379 for a discussion
        # of some of the issues around OMP_NUM_THREADS
        cdef size_t row
        for row in range(num_samples):
            energies_view[row] = energy(self.adj_.first,
                                        self.adj_.second,
                                        samples[row, :])

        return energies

    def to_lists(self):
        """Dump to two lists, mostly for testing"""
        return list(self.adj_.first), list(self.adj_.second)


class AdjArrayBQM(cyAdjArrayBQM, BQM):
    __doc__ = cyAdjArrayBQM.__doc__
