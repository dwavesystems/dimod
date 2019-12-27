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
from collections.abc import Mapping
from numbers import Integral

cimport cython

from cython.operator cimport postincrement as inc, dereference as deref

import numpy as np

from dimod.bqm cimport cyShapeableBQM
from dimod.bqm.adjvectorbqm import AdjVectorBQM
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
from dimod.vartypes import as_vartype, Vartype


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
            Creates an empty binary quadratic model.

        AdjArrayBQM(bqm)
            Construct a new bqm that is a copy of the given one.

        AdjArrayBQM(bqm, vartype)
            Construct a new bqm, changing to the appropriate vartype if
            necessary.

        AdjArrayBQM(n, vartype)
            Make a bqm with all zero biases, where n is the number of nodes.

        AdjArrayBQM(M, vartype)
            Where M is a square, array_like_ or a dictionary of the form
            `{(u, v): b, ...}`. Note that when formed with SPIN-variables,
            biases on the diagonal are added to the offset.

    .. _array_like: https://docs.scipy.org/doc/numpy/user/basics.creation.html

    """

    def __cinit__(self, *args, **kwargs):
        self.dtype = dtype
        self.itype = itype
        self.ntype = ntype

        # otherwise these would be None
        self._label_to_idx = dict()
        self._idx_to_label = dict()

        # this should happen implicitly but to make it explicit
        self.offset_ = 0


    def __init__(self, *args, vartype=None):

        if vartype is not None:
            # pass in as a positional argument
            self.__init__(*args, vartype)
            return

        if len(args) == 0:
            raise TypeError("A valid vartype or another bqm must be provided")
        if len(args) == 1:
            # BQM(bqm) or BQM(vartype)
            obj, = args
            if isinstance(obj, BQM):
                self._init_bqm(obj)
            else:
                self._init_number(0, obj)
        elif len(args) == 2:
            # BQM(bqm, vartype), BQM(n, vartype) or BQM(M, vartype)
            obj, vartype = args
            if isinstance(obj, BQM):
                self._init_bqm(obj, vartype)
            elif isinstance(obj, Integral):
                self._init_number(obj, vartype)
            else:
                # make sure linear is NOT a mapping or else it would make
                # another intermediate BQM
                self._init_components([], obj, 0.0, vartype)
        elif len(args) == 3:
            # BQM(linear, quadratic, vartype)
            linear, quadratic, vartype = args
            self._init_components(linear, quadratic, 0.0, vartype)
        elif len(args) == 4:
            # BQM(linear, quadratic, offset, vartype)
            self._init_components(*args)
        else:
            msg = "__init__() takes 4 positional arguments but {} were given"
            raise TypeError(msg.format(len(args)))

    def _init_bqm(self, bqm, vartype=None):
        cdef cyAdjArrayBQM cybqm
        if isinstance(bqm, cyAdjArrayBQM):
            # this would actually work with _init_cybqm but it's faster to do
            # straight copy
            cybqm = bqm
            self.adj_ = cybqm.adj_  # copy
            self.offset_ = cybqm.offset_
            self.vartype = cybqm.vartype

            # shallow copy is OK since everything is hashable
            self._label_to_idx = cybqm._label_to_idx.copy()
            self._idx_to_label = cybqm._idx_to_label.copy()

        else:
            try:
                self._init_cybqm(bqm)
            except TypeError:
                # probably AdjDictBQM or subclass, just like when constructing
                # with maps, it's a lot easier/nicer to pass through
                # AdjVectorBQM
                self._init_bqm(AdjVectorBQM(bqm), vartype=vartype)

        if vartype is not None:
            self.change_vartype(as_vartype(vartype), inplace=True)

    def _init_cybqm(self, cyShapeableBQM bqm):
        """Copy a shapeable BQM into self."""

        # we know how big we'll need to be. Note that num_interactions is O(|V|)
        # for the shapeable bqms. Testing shows it's faster to do it though.
        self.adj_.first.reserve(num_variables(bqm.adj_))
        self.adj_.second.reserve(2*num_interactions(bqm.adj_))  # O(|V|) for bqm
        
        cdef VarIndex vi

        for vi in range(num_variables(bqm.adj_)):
            self.adj_.first.push_back(
                pair[size_t, Bias](self.adj_.second.size(), get_linear(bqm.adj_, vi)))

            span = neighborhood(bqm.adj_, vi)
            self.adj_.second.insert(self.adj_.second.end(), span.first, span.second)

        self.offset_ = bqm.offset_
        self.vartype = bqm.vartype

        # shallow copy is OK since everything is hashable
        self._label_to_idx = bqm._label_to_idx.copy()
        self._idx_to_label = bqm._idx_to_label.copy()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _init_components(self, linear, quadratic, offset, vartype):
        self.vartype = vartype = as_vartype(vartype)

        if isinstance(linear, Mapping) or isinstance(quadratic, Mapping):
            # constructing from dictionaries is a lot easier if you have
            # incremental construction, so we build using one of the shapeable
            # bqms
            self._init_bqm(AdjVectorBQM(linear, quadratic, offset, vartype))
            return

        cdef bool is_spin = self.vartype is Vartype.SPIN

        cdef Bias[:, :] qmatrix = np.atleast_2d(np.asarray(quadratic, dtype=self.dtype))

        cdef Py_ssize_t nvar = qmatrix.shape[0]

        if qmatrix.ndim != 2 or nvar != qmatrix.shape[1]:
            raise ValueError("expected dense to be a 2 dim square array")

        # we know how big linear is going to be, so we can reserve it now. But,
        # because we ignore 0s on the off-diagonal, we can't do the same for
        # quadratic which is where we would get the big savings.
        self.adj_.first.reserve(nvar)
        # self.adj_.second.reserve(nvar*(nvar-1))  # if it was perfectly dense

        cdef VarIndex ui, vi
        cdef Bias qbias
        for ui in range(nvar):
            if is_spin:
                self.adj_.first.push_back(
                    pair[size_t, Bias](self.adj_.second.size(), 0))
                self.offset_ += qmatrix[ui, ui]
            else:
                self.adj_.first.push_back(
                    pair[size_t, Bias](self.adj_.second.size(), qmatrix[ui, ui]))

            for vi in range(nvar):
                if ui == vi:
                    continue

                qbias = qmatrix[ui, vi] + qmatrix[vi, ui]  # add upper and lower

                if qbias:  # ignore 0 off-diagonal
                    self.adj_.second.push_back(pair[VarIndex, Bias](vi, qbias))

        cdef Bias[:] ldata = np.asarray(linear, dtype=self.dtype)
        nvar = ldata.shape[0]

        # handle the case that ldata is larger
        for vi in range(num_variables(self.adj_), nvar):
            self.adj_.first.push_back(
                pair[size_t, Bias](self.adj_.second.size(), 0))

        for vi in range(nvar):
            set_linear(self.adj_, vi, ldata[vi] + get_linear(self.adj_, vi))

    def _init_number(self, int n, vartype):
        # Make a new BQM with n variables all with 0 bias
        self.adj_.first.resize(n)
        self.vartype = as_vartype(vartype)

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

    def change_vartype(self, vartype, inplace=True):
        """Return a binary quadratic model with the specified vartype.

        Args:
            vartype (:class:`.Vartype`/str/set, optional):
                Variable type for the changed model. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            inplace (bool, optional, default=True):
                If True, the binary quadratic model is updated in-place;
                otherwise, a new binary quadratic model is returned.

        Returns:
            :obj:`.AdjArrayBQM`: A binary quadratic model with the specified
            vartype.

        """
        if not inplace:
            return self.copy().change_vartype(vartype, inplace=True)

        vartype = as_vartype(vartype)

        # in place and we are already correct, so nothing to do
        if self.vartype == vartype:
            return self

        cdef Bias lin_mp, lin_offset_mp, quad_mp, quad_offset_mp
        if vartype is Vartype.BINARY:
            lin_mp, lin_offset_mp = 2.0, -1.0
            quad_mp, lin_quad_mp, quad_offset_mp = 4.0, -2.0, 0.5
        elif vartype is Vartype.SPIN:
            lin_mp, lin_offset_mp = 0.5, 0.5
            quad_mp, lin_quad_mp, quad_offset_mp = 0.25, 0.25, 0.125
        else:
            raise ValueError("unkown vartype")

        cdef VarIndex ui, vi
        cdef Bias bias
        cdef NeighborhoodIndex ni

        for ui in range(num_variables(self.adj_)):
            bias = self.adj_.first[ui].second

            self.adj_.first[ui].second = lin_mp * bias
            self.offset_ += lin_offset_mp * bias

            span = neighborhood(self.adj_, ui)
            while span.first != span.second:
                bias = deref(span.first).second

                deref(span.first).second = quad_mp * bias
                self.adj_.first[ui].second += lin_quad_mp * bias
                self.offset_ += quad_offset_mp * bias

                inc(span.first)

        self.vartype = vartype

        return self

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

        if variables is None:
            # in the case that variables is unlabelled we can speed things up
            # by just walking through the range
            for ui in range(num_variables(self.adj_)):
                u = self._idx_to_label.get(ui, ui)

                span = neighborhood(self.adj_, ui)
                while span.first != span.second:
                    vi = deref(span.first).first
                    b = deref(span.first).second
                    if vi > ui:  # have not already visited
                        v = self._idx_to_label.get(vi, vi)
                        yield u, v, as_numpy_scalar(b, self.dtype)

                    inc(span.first)
        elif self.has_variable(variables):
            yield from self.iter_quadratic([variables])
        else:
            seen = set()
            for u in variables:
                ui = self.label_to_idx(u)
                seen.add(u)

                span = neighborhood(self.adj_, ui)
                while span.first != span.second:
                    vi = deref(span.first).first
                    b = deref(span.first).second

                    v = self._idx_to_label.get(vi, vi)

                    if v not in seen:
                        yield u, v, as_numpy_scalar(b, self.dtype)

                    inc(span.first)

    def get_linear(self, object v):
        return as_numpy_scalar(get_linear(self.adj_, self.label_to_idx(v)),
                               self.dtype)

    def get_quadratic(self, u, v, default=None):
        """Get the quadratic bias of (u, v).

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

            default (number, optional):
                Value to return if there is no interactions between `u` and `v`.

        Returns:
            The quadratic bias of (u, v).

        Raises:
            ValueError: If either `u` or `v` is not a variable in the binary
            quadratic model or if `u == v`

            ValueError: If `(u, v)` is not an interaction and `default` is
            `None`.

        """

        if u == v:
            raise ValueError('No interaction between {} and {}'.format(u, v))

        cdef VarIndex ui = self.label_to_idx(u)
        cdef VarIndex vi = self.label_to_idx(v)
        cdef pair[Bias, bool] out = get_quadratic(self.adj_, ui, vi)

        if not out.second:
            if default is None:
                msg = 'No interaction between {!r} and {!r}'
                raise ValueError(msg.format(u, v))

            return self.dtype.type(default)

        return as_numpy_scalar(out.first, self.dtype)

    @classmethod
    def _load(cls, fp, data, offset=0):
        """This method is used by :func:`.load` and should not be invoked
        directly.

        `fp` must be readable and seekable.

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