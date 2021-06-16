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

import collections.abc as abc
import numbers

cimport cython

from libcpp.algorithm cimport copy
from libcpp.iterator cimport inserter
from cython.operator cimport postincrement as inc, dereference as deref
from libcpp.pair cimport pair
from libcpp.vector cimport vector

import numpy as np

from dimod.bqm cimport cyBQM
from dimod.bqm.common cimport NeighborhoodIndex, Integral32plus, Numeric32plus
from dimod.bqm.common import dtype, itype, ntype
from dimod.bqm.utils cimport as_numpy_scalar
from dimod.bqm.utils import coo_sort, cyenergies, cyrelabel
from dimod.bqm.utils import cyrelabel_variables_as_integers
from dimod.core.bqm import BQM, ShapeableBQM
from dimod.utilities import asintegerarrays, asnumericarrays
from dimod.vartypes import as_vartype, Vartype


@cython.embedsignature(True)
cdef class cyAdjVectorBQM:

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
            if hasattr(args[0], 'vartype'):
                self._init_bqm(obj)
            else:
                self._init_number(0, obj)
        elif len(args) == 2:
            # BQM(bqm, vartype), BQM(n, vartype) or BQM(M, vartype)
            obj, vartype = args
            if hasattr(args[0], 'vartype'):
                self._init_bqm(obj, vartype)
            elif isinstance(obj, numbers.Integral):
                self._init_number(obj, vartype)
            else:
                self._init_components({}, obj, 0.0, vartype)
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
        try:
            self._init_cybqm(bqm)
        except TypeError:
            # probably AdjDictBQM or subclass
            self.linear.update(bqm.linear)
            self.quadratic.update(bqm.quadratic)
            self.offset_ = bqm.offset
            self.vartype = bqm.vartype

        if vartype is not None:
            self.change_vartype(as_vartype(vartype), inplace=True)

    def _init_cybqm(self, cyBQM bqm):
        """Construct self from another cybqm."""
        self.bqm_ = cppAdjVectorBQM[VarIndex, Bias](bqm.bqm_)

        self.offset_ = bqm.offset_
        self.vartype = bqm.vartype

        # shallow copy is OK since everything is hashable
        self._label_to_idx = bqm._label_to_idx.copy()
        self._idx_to_label = bqm._idx_to_label.copy()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _init_components(self, linear, quadratic, offset, vartype):
        self.vartype = as_vartype(vartype)
        cdef bint is_spin = self.vartype is Vartype.SPIN

        self.offset_ = offset

        cdef object u, v
        cdef VarIndex ui, vi
        cdef Bias b

        # let's do the (expensive) quadratic first in case we're copying from
        # a dense matrix
        cdef Bias[:, :] qmatrix
        cdef Py_ssize_t nv

        if isinstance(quadratic, abc.Mapping):
            # we could do a bit more to make this faster, but we're accessing
            # a python dict anyway so simpler is probably better
            for (u, v), b in quadratic.items():
                if u == v:
                    if is_spin:
                        self.offset_ += b
                    else:
                        # because we're doing this first on an empty bqm we
                        # can just set
                        self.set_linear(u, b)
                else:
                    ui = self.label_to_idx(self.add_variable(u))
                    vi = self.label_to_idx(self.add_variable(v))
                    # this does two binary searches
                    self.bqm_.set_quadratic(ui, vi, self.bqm_.get_quadratic(ui, vi).first + b)
        elif isinstance(quadratic, abc.Iterator):
            for u, v, b in quadratic:
                if u == v:
                    if is_spin:
                        self.offset_ += b
                    else:
                        # because we're doing this first on an empty bqm we
                        # can just set
                        self.set_linear(u, b)
                else:
                    ui = self.label_to_idx(self.add_variable(u))
                    vi = self.label_to_idx(self.add_variable(v))
                    # this does two binary searches
                    self.bqm_.set_quadratic(ui, vi, self.bqm_.get_quadratic(ui, vi).first + b)
        else:
            qmatrix = np.atleast_2d(np.asarray(quadratic, dtype=self.dtype))

            nvar = qmatrix.shape[0]

            if qmatrix.ndim != 2 or nvar != qmatrix.shape[1]:
                raise ValueError("expected dense to be a 2 dim square array")

            self.bqm_.adj.resize(nvar)

            for ui in range(nvar):
                if is_spin:
                    self.offset_ += qmatrix[ui, ui]
                else:
                    self.bqm_.set_linear(ui, qmatrix[ui, ui])

                for vi in range(ui + 1, nvar):
                    b = qmatrix[ui, vi] + qmatrix[vi, ui]  # add upper and lower

                    if b == 0:  # ignore 0 off-diagonal
                        continue

                    # we'd like to use self.bqm_.set_quadratic(ui, vi, b) but
                    # because we know that we're always adding to the end
                    # of the neighborhood, we can provide a location hint to
                    # insert. We should really do this with an iterator from c++
                    # space, but those are not implemented yet
                    # we use insert because it is the same for map and vector
                    self.bqm_.adj[ui].first.insert(self.bqm_.adj[ui].first.end(),
                                               (vi, b))  # todo: cythonize
                    self.bqm_.adj[vi].first.insert(self.bqm_.adj[vi].first.end(),
                                               (ui, b))  # todo: cythonize

        cdef Bias[:] ldata

        if isinstance(linear, abc.Mapping):
            # again, we could do some fiddling to potentially make this a bit
            # faster, but because we're accessing a dict, probably clearer is
            # better
            for v, b in linear.items():
                # get the index of the variable, ensuring that it exists
                vi = self.label_to_idx(self.add_variable(v))
                self.bqm_.set_linear(vi, self.bqm_.get_linear(vi) + b)
        elif isinstance(linear, abc.Iterator):
            for v, b in linear:
                # get the index of the variable, ensuring that it exists
                vi = self.label_to_idx(self.add_variable(v))
                self.bqm_.set_linear(vi, self.bqm_.get_linear(vi) + b)
        else:
            ldata = np.asarray(linear, dtype=self.dtype)

            if ldata.ndim != 1:
                raise ValueError("expected linear to be a 1 dim array")

            if self._label_to_idx:
                # we've overwritten some labels, so we have to do all the label
                # checking which is slow
                for vi in range(ldata.shape[0]):
                    if self.has_variable(vi):
                        self.set_linear(vi, ldata[vi] + self.get_linear(vi))
                    else:
                        self.set_linear(vi, ldata[vi])
            else:
                if ldata.shape[0] > self.bqm_.num_variables():
                    self.bqm_.adj.resize(ldata.shape[0])  # append

                for vi in range(ldata.shape[0]):
                    self.bqm_.set_linear(vi, ldata[vi] + self.bqm_.get_linear(vi))


    def _init_number(self, int n, vartype):
        # Make a new BQM with n variables all with 0 bias
        self.bqm_.adj.resize(n)
        self.vartype = as_vartype(vartype)

    def __copy__(self):
        cdef cyAdjVectorBQM bqm = type(self)(self.vartype)

        bqm.bqm_ = self.bqm_
        bqm.offset_ = self.offset_

        # everything is immutable, so this is a deep copy
        bqm._label_to_idx = self._label_to_idx.copy()
        bqm._idx_to_label = self._idx_to_label.copy()

        return bqm

    def __deepcopy__(self, memo):
        # all copies are deep
        memo[id(self)] = new = self.__copy__()
        return new

    # todo: support protocol 5, if possible
    def __reduce__(self):
        from dimod.serialization.fileview import FileView, load
        return (load, (FileView(self).readall(),))

    @property
    def num_variables(self):
        """int: Number of variables in the model."""
        return self.bqm_.num_variables()

    @property
    def num_interactions(self):
        """int: Number of interactions in the model."""
        return self.bqm_.num_interactions()

    @property
    def offset(self):
        """Constant energy offset associated with the model."""
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

        if vi < 0 or vi >= self.bqm_.num_variables():
            raise ValueError("{} is not a variable".format(v))

        return vi

    def add_variable(self, object v=None, Bias bias=0.0):
        """Add a variable to the binary quadratic model.

        Args:
            v (hashable, optional):
                A label for the variable. Defaults to the length of the binary
                quadratic model, if that label is available. Otherwise defaults
                to the lowest available positive integer label.

            bias (numeric, optional, default=0):
                The initial bias value for the added variable. If `v` is already
                a variable, then `bias` (if any) is adding to its existing
                linear bias.

        Returns:
            hashable: The label of the added variable.

        Raises:
            TypeError: If the label is not hashable.

        """
        cdef VarIndex vi

        if v is None:
            # we're definitely adding a new variable
            vi = self.bqm_.add_variable()
            v = vi  # python version

            if v in self._label_to_idx:
                # the index is already in use, there must be a missing
                # smaller integer we can use
                for v in range(v):
                    if v not in self._label_to_idx:
                        break
        else:
            try:
                vi = self.label_to_idx(v)
            except ValueError:
                # doesn't exist to make a new one and track it
                vi = self.bqm_.add_variable()
                if vi != v:
                    self._label_to_idx[v] = vi
                    self._idx_to_label[vi] = v

        self.bqm_.set_linear(vi, bias + self.bqm_.get_linear(vi))
        return v

    def iter_linear(self):
        """Iterate over the linear biases of the binary quadratic model.

        Yields:
            tuple: A variable in the binary quadratic model and its linear bias.
        """
        # this should really be done with an iterator, but we don't have them
        # yet so we'll do it manually
        cdef VarIndex vi
        cdef object v
        cdef Bias b

        for vi in range(self.bqm_.num_variables()):
            b = self.bqm_.adj[vi].second
            v = self._idx_to_label.get(vi, vi)
            yield v, as_numpy_scalar(b, self.dtype)

    def iter_quadratic(self, object variables=None):
        """Iterate over the quadratic biases of the binary quadratic model.

        Args:
            variables (iterable):
                Variables in the binary quadratic model. Iterates only over
                interactions of these variables.

        Yields:
            3-tuple: Interaction variables in the binary quadratic model and their
            bias.
        """
        # this would be much easier if interaction_iterator was implemented, but
        # we'll do this by-hand for now
        cdef VarIndex ui, vi  # c indices
        cdef object u, v  # python labels
        cdef Bias b

        # cdef pair[VarIndex, Bias] vi_b  # todo: cythonize

        if variables is None:
            # in the case that variables is unlabelled we can speed things up
            # by just walking through the range
            for ui in range(self.bqm_.num_variables()):
                u = self._idx_to_label.get(ui, ui)

                # we could find the index of the first vi > ui, but this is
                # simpler for now and make for more generic code between
                # vector/map implementations
                for vi_b in self.bqm_.adj[ui].first:
                    vi = vi_b.first
                    b = vi_b.second

                    if vi < ui:
                        continue

                    v = self._idx_to_label.get(vi, vi)

                    yield u, v, as_numpy_scalar(b, self.dtype)
        elif self.has_variable(variables):
            yield from self.iter_quadratic([variables])
        else:
            seen = set()
            for u in variables:
                ui = self.label_to_idx(u)
                seen.add(u)

                for vi_b in self.bqm_.adj[ui].first:
                    vi = vi_b.first
                    b = vi_b.second

                    v = self._idx_to_label.get(vi, vi)

                    if v in seen:
                        continue

                    yield u, v, as_numpy_scalar(b, self.dtype)

    def get_linear(self, object v):
        """Get the linear bias of a specified variable.

        Args:
            v (hashable):
                A variable in the binary quadratic model.

        Returns:
            float: The linear bias of ``v``.

        Raises:
            ValueError: If ``v`` is not a variable in the binary quadratic model.

        """
        return as_numpy_scalar(self.bqm_.get_linear(self.label_to_idx(v)),
                               self.dtype)

    def get_quadratic(self, u, v, default=None):
        """Get the quadratic bias of the specified interaction.

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

            default (number, optional):
                Value to return if there is no interactions between ``u`` and ``v``.

        Returns:
            The quadratic bias of ``(u, v)``.

        Raises:
            ValueError: If either ``u`` or ``v`` is not a variable in the binary
                quadratic model or if ``u == v``

            ValueError: If ``(u, v)`` is not an interaction and `default` is
                `None`.

        """
        if u == v:
            raise ValueError("No interaction between {} and itself".format(u))

        if default is not None:
            try:
                return self.get_quadratic(u, v)
            except ValueError:
                return self.dtype.type(default)

        cdef VarIndex ui = self.label_to_idx(u)
        cdef VarIndex vi = self.label_to_idx(v)
        out = self.bqm_.get_quadratic(ui, vi)  # todo: cythonize

        if not out.second:
            msg = 'No interaction between {!r} and {!r}'
            raise ValueError(msg.format(u, v))

        return as_numpy_scalar(out.first, self.dtype)

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
            :obj:`.AdjVectorBQM`: A binary quadratic model with the specified
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

        for ui in range(self.bqm_.num_variables()):
            bias = self.bqm_.adj[ui].second

            self.bqm_.adj[ui].second = lin_mp * bias
            self.offset_ += lin_offset_mp * bias

            span = self.bqm_.neighborhood(ui)
            while span.first != span.second:
                bias = deref(span.first).second

                deref(span.first).second = quad_mp * bias
                self.bqm_.adj[ui].second += lin_quad_mp * bias
                self.offset_ += quad_offset_mp * bias

                inc(span.first)

        self.vartype = vartype

        return self

    def degree(self, object v):
        """Return degree of the specified variable.

        The degree is the number of variables sharing an interaction with ``v``.

        Args:
            v (hashable):
                Variable in the binary quadratic model.

        Returns:
            Degree of ``v``.
        """

        cdef VarIndex vi = self.label_to_idx(v)
        return self.bqm_.degree(vi)

    def energies(self, samples, dtype=None):
        """Determine the energies of the given samples.

        Args:
            samples_like (samples_like):
                A collection of raw samples. `samples_like` is an extension of
                NumPy's array_like structure. See :func:`.as_samples`.

            dtype (data-type, optional, default=None):
                The desired NumPy data type for the energies. Matches
                :attr:`.dtype` by default.

        Returns:
            :obj:`numpy.ndarray`: The energies.

        """
        return np.asarray(cyenergies(self, samples), dtype=dtype)

    @classmethod
    def from_file(cls, file_like):
        """Construct a BQM from a file-like object.

        See also:
            :meth:`AdjVectorBQM.to_file`: To construct a file-like object.

            :func:`~dimod.serialization.fileview.load`

        """
        from dimod.serialization.fileview import load
        return load(file_like, cls=cls)

    @classmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _from_numpy_vectors(cls,
                            Numeric32plus[::1] linear,
                            Integral32plus[::1] irow,
                            Integral32plus[::1] icol,
                            Numeric32plus[::1] qdata,
                            object vartype):
        """Equivalent of from_numpy_vectors with fused types."""

        if not irow.shape[0] == icol.shape[0] == qdata.shape[0]:
            raise ValueError("quadratic vectors should be equal length")
        cdef Py_ssize_t length = irow.shape[0]

        cdef cyAdjVectorBQM bqm = cls(vartype)

        cdef bint ignore_diagonal = bqm.vartype is Vartype.SPIN

        if length:
            bqm.bqm_ = cppAdjVectorBQM[VarIndex, Bias](
                &irow[0], &icol[0], &qdata[0], length, ignore_diagonal)

        # add the linear
        while bqm.bqm_.num_variables() < linear.shape[0]:
            bqm.bqm_.add_variable()

        cdef Py_ssize_t v
        for v in range(linear.shape[0]):
            bqm.bqm_.set_linear(v, bqm.bqm_.get_linear(v) + linear[v])

        # add the offset
        cdef Py_ssize_t qi
        if ignore_diagonal:
            for qi in range(irow.shape[0]):
                if irow[qi] == icol[qi]:
                    bqm.offset_ += qdata[qi]

        return bqm

    @classmethod
    def from_numpy_vectors(cls, linear, quadratic, offset, vartype,
                           variable_order=None):
        """Create a binary quadratic model from vectors.

        Args:
            linear (array_like):
                A 1D array-like iterable of linear biases.

            quadratic (tuple[array_like, array_like, array_like]):
                A 3-tuple of 1D array_like vectors of the form (row, col, bias).

            offset (numeric, optional):
                Constant offset for the binary quadratic model.

            vartype (:class:`.Vartype`/str/set):
                Variable type for the binary quadratic model. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            variable_order (iterable, optional):
                If provided, labels the variables; otherwise, indices are used.

        Returns:
            A binary quadratic model

        """
        try:
            irow, icol, qdata = quadratic
        except ValueError:
            raise ValueError("quadratic should be a 3-tuple")

        # We need:
        # * numpy ndarrays
        # * contiguous memory
        # * ldata.dtype == qdata.dtype and irow.dtype == icol.dtype
        # * 32 or 64 bit dtypes
        icol, irow = asintegerarrays(
            icol, irow, min_itemsize=4, requirements='C')
        ldata, qdata = asnumericarrays(
            linear, qdata, min_itemsize=4, requirements='C')

        bqm = cls._from_numpy_vectors(ldata, irow, icol, qdata, vartype)

        bqm.offset += offset

        if variable_order is not None:
            if len(variable_order) != len(bqm):
                raise ValueError(
                    "variable_order must be the same length as the BQM")

            bqm.relabel_variables(dict(enumerate(variable_order)))

        return bqm

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
        cdef cyAdjVectorBQM bqm = cls(num_var, data['vartype'])

        # offset, using the vartype it was encoded with
        bqm.offset_ = np.frombuffer(fp.read(dtype.itemsize), dtype)[0]

        # linear
        # annoyingly this does two copies, one into the bytes object returned
        # by read, the other into the array. We could potentially get around
        # this by using readinto and providing the bytesarray, then passing that
        # into np.asarray()
        # TODO: bug, read might actually return fewer bytes
        ldata = np.frombuffer(fp.read(num_var*(ntype.itemsize + dtype.itemsize)),
                              dtype=np.dtype([('nidx', ntype), ('bias', dtype)]))

        # and using the dtypes of the bqm. If they match, this is luckily not a
        # copy! Numpy is cool
        cdef const Bias[:] lbiases = np.asarray(ldata['bias'], dtype=bqm.dtype)
        cdef const NeighborhoodIndex[:] nidxs = np.asarray(ldata['nidx'],
                                                           dtype=bqm.ntype)

        cdef VarIndex ui
        cdef Py_ssize_t nstart, nend, num_bytes, i
        cdef const Bias[:] qbiases
        cdef const VarIndex[:] outvars
        for ui in range(num_var):
            bqm.bqm_.set_linear(ui, <Bias>lbiases[ui])

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
                bqm.bqm_.adj[ui].first.insert(bqm.bqm_.adj[ui].first.end(),
                                          (outvars[i], qbiases[i]))  # todo: cythonize

        return bqm


    def remove_interaction(self, object u, object v):
        """Remove the interaction between variables u and v.

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

        Returns:
            bool: If there was an interaction to remove.

        Raises:
            ValueError: If either u or v is not a variable in the binary
                quadratic model.

        """
        if u == v:
            raise ValueError("No interaction between {} and itself".format(u))

        cdef VarIndex ui = self.label_to_idx(u)
        cdef VarIndex vi = self.label_to_idx(v)
        cdef bint removed = self.bqm_.remove_interaction(ui, vi)

        if not removed:
            raise ValueError('No interaction between {} and {}'.format(u, v))

    def add_linear_equality_constraint(self, object terms, Bias
                                       lagrange_multiplier, Bias constant):
        """Add a linear constraint as a quadratic objective.

        Adds a linear constraint of the form
        :math:`\sum_{i} a_{i} x_{i} + C = 0`
        to the binary quadratic model as a quadratic objective.

        Args:
            terms (iterable/iterator):
                An iterable of 2-tuples, (variable, bias).
                Each tuple is evaluated to the term (bias * variable).
                All terms in the list are summed.
            lagrange_multiplier:
                The coefficient or the penalty strength. This value is
                multiplied by the entire constraint objective and added to the
                bqm (it doesn't appear explicity in the equation above).
            constant:
                The constant value of the constraint, C, in the equation above.

        """
        cdef vector[VarIndex] variables
        cdef vector[Bias] biases

        # can allocate them if we already know the size
        if isinstance(terms, abc.Sized):
            biases.reserve(len(terms))
            variables.reserve(len(terms))

        # put the generator or list into our C++ objects
        cdef Py_ssize_t v
        cdef Bias bias
        for var, bias in terms:
            variables.push_back(self.label_to_idx(var))
            biases.push_back(bias)

        # add the biases to the BQM, not worrying about order or duplication
        cdef Py_ssize_t num_terms = biases.size()

        cdef Py_ssize_t i, j
        cdef Bias lbias, qbias
        self.offset += lagrange_multiplier * constant * constant
        for i in range(num_terms):
            v = variables[i]
            lbias = lagrange_multiplier * biases[i] * (2 * constant + biases[i])
            self.bqm_.set_linear(v, lbias + self.bqm_.get_linear(v))
            for j in range(i + 1, num_terms):
                u = variables[j]
                qbias = 2 * lagrange_multiplier * biases[i] * biases[j]
                self.bqm_.set_quadratic(u, v,
                qbias + self.bqm_.get_quadratic(u, v).first)

    def remove_variable(self, object v=None):
        """Remove a variable and its associated interactions.

        Args:
            v (variable, optional):
                The variable to be removed from the binary quadratic model. If
                not provided, the last variable added is removed.

        Returns:
            variable: The removed variable.

        Raises:
            ValueError: If the binary quadratic model is empty or if ``v`` is not
                a variable.

        """
        if self.bqm_.num_variables() == 0:
            raise ValueError("pop from empty binary quadratic model")

        cdef VarIndex vi  # index of v in the underlying adjacency

        if v is None:
            # just remove the last variable
            vi = self.bqm_.pop_variable()

            # remove the relabels, if any
            v = self._idx_to_label.pop(vi, vi)
            self._label_to_idx.pop(v, None)

            return v

        # in this case we're removing a variable in the middle of the
        # underlying adjacency. We do this by "swapping" the last variable
        # and v, then popping v.

        cdef object last
        cdef VarIndex lasti

        vi = self.label_to_idx(v)

        lasti = self.bqm_.num_variables() - 1
        last = self._idx_to_label.get(lasti, lasti)

        if lasti == vi:
            # equivalent to the None case
            return self.remove_variable()

        # remove all of v's interactions
        for _, u, _ in list(self.iter_quadratic(v)):
            self.remove_interaction(u, v)

        # copy last's to v
        self.bqm_.set_linear(vi, self.bqm_.get_linear(lasti))
        for _, u, b in self.iter_quadratic(last):
            self.set_quadratic(u, v, b)

        # swap last's and v's labels
        self._idx_to_label[vi] = last
        self._label_to_idx[last] = vi
        self._idx_to_label[lasti] = v
        self._label_to_idx[v] = lasti

        # pop last
        self.remove_variable()

        return v

    def relabel_variables(self, mapping, inplace=True):
        """Relabel variables of a binary quadratic model as specified by mapping.

        Args:
            mapping (dict):
                Dict mapping current variable labels to new ones. If an
                incomplete mapping is provided, unmapped variables retain their
                current labels.

            inplace (bool, optional, default=True):
                If True, the binary quadratic model is updated in-place;
                otherwise, a new binary quadratic model is returned.

        Returns:
            Binary quadratic model with relabeled variables. If `inplace`
            is set to True, returns itself.
        """
        return cyrelabel(self, mapping, inplace)

    def relabel_variables_as_integers(self, inplace=True):
            """Relabel as integers the variables of a binary quadratic model.

            Uses the natural labelling of the underlying C++ objects.

            Args:
                inplace (bool, optional, default=True):
                    If True, the binary quadratic model is updated in-place;
                    otherwise, a new binary quadratic model is returned.

            Returns:
                tuple: A 2-tuple containing:

                    A binary quadratic model with the variables relabeled. If
                    `inplace` is set to True, returns itself.

                    dict: Mapping that restores the original labels.
            """
            return cyrelabel_variables_as_integers(self, inplace)

    def set_linear(self, object v, Bias b):
        """Set the linear biase of a variable v.

        Args:
            v (hashable):
                A variable in the binary quadratic model. It is added if not
                already in the model.

            b (numeric):
                The linear bias of v.

        Raises:
            TypeError: If v is not hashable
        """
        cdef VarIndex vi

        # this try-catch it not necessary but it speeds things up in the case
        # that the variable already exists which is the typical case
        try:
            vi = self.label_to_idx(v)
        except ValueError:
            vi = self.label_to_idx(self.add_variable(v))

        self.bqm_.set_linear(vi, b)

    def set_quadratic(self, object u, object v, Bias b):
        """Set the quadratic bias of (u, v).

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

            b (numeric):
                The linear bias of v.

        Raises:
            TypeError: If u or v is not hashable.

        """
        cdef VarIndex ui, vi

        # these try-catchs are not necessary but it speeds things up in the case
        # that the variables already exists which is the typical case
        try:
            ui = self.label_to_idx(u)
        except ValueError:
            ui = self.label_to_idx(self.add_variable(u))
        try:
            vi = self.label_to_idx(v)
        except ValueError:
            vi = self.label_to_idx(self.add_variable(v))

        self.bqm_.set_quadratic(ui, vi, b)

    def to_file(self):
        """View the BQM as a file-like object.

        See also:

            :meth:`AdjVectorBQM.from_file`: To construct a bqm from a file-like
            object.

            :func:`~dimod.serialization.fileview.FileView`

        """
        from dimod.serialization.fileview import FileView
        return FileView(self)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def to_numpy_vectors(self, variable_order=None,
                         dtype=None, index_dtype=None,
                         sort_indices=False, sort_labels=True,
                         return_labels=False):
        """Convert binary quadratic model to NumPy vectors.

        Args:
            variable_order (iterable, optional, default=None):
                Variable order for the vector output. By default uses
                the order of the binary quadratic model.

            dtype (data-type, optional, default=None):
                Desired NumPy data type for the linear biases.

            index_dtype (data-type, optional, default=None):
                Desired NumPy data type for the indices.

            sort_indices (Boolean, optional, default=False):
                If True, sorts index vectors of variables and interactions.

            sort_labels (Boolean, optional, default=True):
                If True, sorts vectors based on variable labels.

            return_labels (Boolean, optional, default=False):
                If True, returns a list of variable labels.

        Returns:
            tuple: A tuple containing:

                Array of linear biases.

                3-tuple of arrays ``u``, ``v``, and ``b``, where the first two
                are variables that form interactions and the third is the
                quadratic bias of the interaction.

                Offset.

                Optionally, variable labels.
        """

        cdef Py_ssize_t nv = self.bqm_.num_variables()
        cdef Py_ssize_t ni = self.bqm_.num_interactions()

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
            span = self.bqm_.neighborhood(vi)

            while span.first != span.second and deref(span.first).first < vi:
                irow_view[qi] = vi
                icol_view[qi] = deref(span.first).first
                qdata_view[qi] = deref(span.first).second

                qi += 1
                inc(span.first)

        # at this point we have the arrays but they are index-order, NOT the
        # label-order. So we need to do some fiddling
        cdef long[:] reindex
        cdef long ri
        if variable_order is not None or (sort_labels and self._label_to_idx):
            if variable_order is None:
                variable_order = [self._idx_to_label.get(v, v) for v in range(nv)]
                if sort_labels:
                    try:
                        variable_order.sort()
                    except TypeError:
                        # can't sort unlike types in py3
                        pass

            # ok, using the variable_order, calculate the re-index
            reindex = np.full(nv, -1, dtype=np.int_)
            for ri, v in enumerate(variable_order):
                vi = self.label_to_idx(v)
                reindex[vi] = ri

                ldata_view[ri] = self.bqm_.get_linear(vi)

            for qi in range(ni):
                irow_view[qi] = reindex[irow_view[qi]]
                icol_view[qi] = reindex[icol_view[qi]]

            labels = variable_order

        else:
            # the fast case! We don't need to do anything except construct the
            # linear
            for vi in range(nv):
                ldata_view[vi] = self.bqm_.get_linear(vi)

            if return_labels:
                labels = [self._idx_to_label.get(v, v) for v in range(nv)]

        if sort_indices:
            coo_sort(irow, icol, qdata)

        ret = [np.asarray(ldata, dtype),
               (np.asarray(irow, index_dtype),
                np.asarray(icol, index_dtype),
                np.asarray(qdata, dtype)),
               self.offset]

        if return_labels:
            ret.append(labels)

        return tuple(ret)


class AdjVectorBQM(cyAdjVectorBQM, ShapeableBQM):
    __doc__ = cyAdjVectorBQM.__doc__
