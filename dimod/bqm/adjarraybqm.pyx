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

from numbers import Integral

cimport cython

from cython.operator cimport postincrement as inc, dereference as deref

import numpy as np

from dimod.bqm cimport cyAdjVectorBQM
from dimod.bqm.common import dtype, itype, ntype
from dimod.bqm.common cimport NeighborhoodIndex
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
        self.dtype = dtype
        self.itype = itype
        self.ntype = ntype

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

    def copy(self):
        """Return a copy."""
        cdef cyAdjArrayBQM bqm = type(self)(self.vartype)

        bqm.adj_ = self.adj_
        bqm.offset_ = self.offset_

        bqm._label_to_idx = self._label_to_idx.copy()
        bqm._idx_to_label = self._idx_to_label.copy()

        return bqm

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

    @classmethod
    def _load(cls, fp, data, offset=0):
        """This method is used by :func:`.load` and should not be invoked
        directly.

        `offset` should point to the start of the offset data.
        """
        fp.seek(offset, 0)

        # these are the vartypes used to encode
        dtype = np.dtype(data['dtype'])
        itype = np.dtype(data['itype'])  # variable index type
        ntype = np.dtype(data['ntype'])  # index of neighborhood type

        cdef Py_ssize_t num_var = data['shape'][0]
        cdef Py_ssize_t num_int = data['shape'][1]

        # make the bqm with the right number of variables and outvars
        cdef cyAdjArrayBQM bqm = cls(data['vartype'])

        # resize the vectors
        bqm.adj_.first.resize(num_var)
        bqm.adj_.second.resize(2*num_int)

        # set the labels, skipping over the redundant ones
        for vi, v in enumerate(data['variables']):
            if vi != v:
                bqm._label_to_idx[v] = vi
                bqm._idx_to_label[vi] = v

        # offset, using the vartype it was encoded with
        bqm.offset = np.frombuffer(fp.read(dtype.itemsize), dtype)[0]

        # linear
        # annoyingly this does two copies, one into the bytes object returned
        # by read, the other into the array. We could potentially get around
        # this by using readinto and providing the bytesarray, then passing that
        # into np.asarray()
        ldata = np.frombuffer(fp.read(num_var*(ntype.itemsize + dtype.itemsize)),
                              dtype=np.dtype([('nidx', ntype), ('bias', dtype)]))

        # and using the dtypes of the bqm. If they match, this is luckily not a
        # copy! Numpy is cool
        cdef const Bias[:] lbiases = np.asarray(ldata['bias'], dtype=bqm.dtype)
        cdef const NeighborhoodIndex[:] nidxs = np.asarray(ldata['nidx'],
                                                           dtype=bqm.ntype)

        cdef VarIndex ui
        cdef NeighborhoodIndex nstart, nend
        cdef Py_ssize_t num_bytes, i
        cdef const Bias[:] qbiases
        cdef const VarIndex[:] outvars
        for ui in range(num_var):
            bqm.adj_.first[ui].first = nidxs[ui]
            bqm.adj_.first[ui].second = lbiases[ui]

            # pull the neighborhoods one variable at a time to save on memory

            nstart = nidxs[ui]
            nend = nidxs[ui+1] if ui < num_var-1 else 2*num_int

            num_bytes = (nend-nstart)*(itype.itemsize+dtype.itemsize)

            qdata = np.frombuffer(fp.read(num_bytes),
                                  dtype=np.dtype([('outvar', itype),
                                                  ('bias', dtype)]))

            # convert to the correct vartypes, not a copy if they match
            outvars = np.asarray(qdata['outvar'], dtype=bqm.itype)
            qbiases = np.asarray(qdata['bias'], dtype=bqm.dtype)

            for i in range(outvars.shape[0]):
                bqm.adj_.second[nstart+i].first = outvars[i]
                bqm.adj_.second[nstart+i].second = qbiases[i]

        return bqm

    def readinto_linear(self, buff, Py_ssize_t pos=0, accumulated_degrees=None):
        """Read bytes representing the linear biases and their neighborhoods
        into a pre-allocated, writeable bytes-like object.

        Args:
            buff (bytes-likes):
                A pre-allocated, writeable bytes-like object.

            pos (int):
                The stream position, relative to the start of the linear biases.

            accumulated_degrees (sequence, optional):
                This is not used, but accepted in order to maintain
                compatibility with other `readinto_linear` methods.

        Returns:
            int: The number of bytes read.

        """
        cdef unsigned char[:] buff_view = buff
        cdef Py_ssize_t bi = 0  # current position in the buffer

        cdef Py_ssize_t num_var = num_variables(self.adj_)

        cdef Py_ssize_t ntype_size = self.ntype.itemsize
        cdef Py_ssize_t dtype_size = self.dtype.itemsize
        cdef Py_ssize_t step_size = ntype_size + dtype_size

        if pos >= num_var*step_size:
            return 0

        # some type definitions
        stype = np.dtype([('neigborhood', self.ntype), ('bias', self.dtype)],
                         align=False)
        cdef size_t[:] neig_view
        cdef Bias[:] bias_view
        cdef const unsigned char[:] out
        cdef Py_ssize_t num_bytes

        # we want to break the buffer into two sections, the head and the body.
        # The body is made of (neighborhood, bias) pairs. The head is any
        # partial pairs that preceed or the length of buffer in the case that
        # that len(buff) is less than the length of a single pair

        # determine which variable we're on
        cdef VarIndex vi = pos // step_size

        # determine the head length
        cdef Py_ssize_t head_length = step_size - (pos % step_size)
        if head_length == step_size and buff_view.shape[0] >= step_size:
            # we're in the correct position already and we have room for at
            # least one step of the body
            head_length -= step_size
        else:
            # the simplest thing to do here is just make a new 1-length struct
            # array and then copy the appropriate slice into the buffer.
            # cython and numpy were having a lot of fights about type so this
            # got a bit messy... but it seems to work
            ldata = np.empty(1, dtype=stype)
            neig_view = ldata['neigborhood']
            bias_view = ldata['bias']
            neig_view[0] = self.adj_.first[vi].first
            bias_view[0] = self.adj_.first[vi].second
            out = ldata.tobytes()

            # copy into buffer, making sure that everything fits correctly
            num_bytes = min(head_length, buff_view.shape[0])
            buff_view[:num_bytes] = out[-head_length:out.shape[0]-head_length+num_bytes]
            bi += num_bytes
            vi += 1

        # determine the number of elements in the body
        cdef Py_ssize_t body_shape = (buff_view.shape[0] - bi) // step_size
        cdef Py_ssize_t body_length = body_shape * step_size

        # there might be an easier way to view the bytes as a struct array but
        # this works for now
        ldata = np.frombuffer(buff_view[bi:bi+body_length],
                              dtype=stype)
        neig_view = ldata['neigborhood']
        bias_view = ldata['bias']

        cdef Py_ssize_t i
        for i in range(body_shape):
            if vi >= num_var:
                break

            neig_view[i] = self.adj_.first[vi].first
            bias_view[i] = self.adj_.first[vi].second

            vi += 1
            bi += step_size

        return bi

    def readinto_offset(self, buff, pos=0):
        cdef unsigned char[:] buff_view = buff
        cdef const unsigned char[:] offset = self.offset.tobytes()

        cdef Py_ssize_t num_bytes
        num_bytes = min(buff.shape[0], offset.shape[0] - pos)

        if num_bytes < 0 or pos < 0:
            return 0

        buff_view[:num_bytes] = offset[pos:pos+num_bytes]

        return num_bytes

    def readinto_quadratic(self, buff, Py_ssize_t pos=0,
                           accumulated_degrees=None):
        """Read bytes representing the quadratic biases and the outvars
        into a pre-allocated, writeable bytes-like object.

        Args:
            buff (bytes-likes):
                A pre-allocated, writeable bytes-like object.

            pos (int):
                The stream position, relative to the start of the quadratic
                data.

            accumulated_degrees (sequence, optional):
                This is not used, but accepted in order to maintain
                compatibility with other `readinto_linear` methods.

        Returns:
            int: The number of bytes read.

        """
        cdef unsigned char[:] buff_view = buff
        cdef Py_ssize_t bi = 0  # current position in the buffer

        cdef Py_ssize_t num_int = num_interactions(self.adj_)

        cdef Py_ssize_t itype_size = self.itype.itemsize
        cdef Py_ssize_t dtype_size = self.dtype.itemsize
        cdef Py_ssize_t step_size = itype_size + dtype_size

        if pos >= 2*num_int*step_size:
            return 0

        # some type definitions
        stype = np.dtype([('outvar', self.itype), ('bias', self.dtype)],
                         align=False)  # we're packing for serialization
        cdef VarIndex[:] outv_view
        cdef Bias[:] bias_view
        cdef const unsigned char[:] out
        cdef Py_ssize_t num_bytes

        # we want to break the buffer into two sections, the head and the body.
        # The body is made of (neighborhood, bias) pairs. The head is any
        # partial pairs that preceed or the length of buffer in the case that
        # that len(buff) is less than the length of a single pair

        # determine where we are in the ourvars
        cdef Py_ssize_t ni = pos // step_size

        # determine the head length
        cdef Py_ssize_t head_length = step_size - (pos % step_size)
        if head_length == step_size and buff_view.shape[0] >= step_size:
            # we're in the correct position already and we have room for at
            # least one step of the body
            head_length -= step_size
        else:
            # the simplest thing to do here is just make a new 1-length struct
            # array and then copy the appropriate slice into the buffer.
            # cython and numpy were having a lot of fights about type so this
            # got a bit messy... but it seems to work
            qdata = np.empty(1, dtype=stype)
            outv_view = qdata['outvar']
            bias_view = qdata['bias']
            outv_view[0] = self.adj_.second[ni].first
            bias_view[0] = self.adj_.second[ni].second
            out = qdata.tobytes()

            # copy into buffer, making sure that everything fits correctly
            num_bytes = min(head_length, buff_view.shape[0])
            buff_view[:num_bytes] = out[-head_length:out.shape[0]-head_length+num_bytes]
            bi += num_bytes
            ni += 1

        # determine the number of elements in the body
        cdef Py_ssize_t body_shape = (buff_view.shape[0] - bi) // step_size
        cdef Py_ssize_t body_length = body_shape * step_size

        # there might be an easier way to view the bytes as a struct array but
        # this works for now
        qdata = np.frombuffer(buff_view[bi:bi+body_length],
                              dtype=stype)
        outv_view = qdata['outvar']
        bias_view = qdata['bias']

        cdef Py_ssize_t i
        for i in range(body_shape):
            if ni >= 2*num_int:
                break

            outv_view[i] = self.adj_.second[ni].first
            bias_view[i] = self.adj_.second[ni].second

            ni += 1
            bi += step_size

        return bi

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
        irow = np.empty(ni, dtype=self.itype)
        icol = np.empty(ni, dtype=self.itype)
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


class AdjArrayBQM(cyAdjArrayBQM, BQM):
    __doc__ = cyAdjArrayBQM.__doc__
