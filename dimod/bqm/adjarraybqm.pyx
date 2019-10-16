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

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from numbers import Integral

cimport cython

from cython.operator cimport postincrement as inc, dereference as deref

import numpy as np

from dimod.bqm.adjmapbqm cimport cyAdjMapBQM
from dimod.bqm.cppbqm cimport (num_variables,
                               num_interactions,
                               get_linear,
                               get_quadratic,
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
            self.invars_.resize(obj)
        elif isinstance(obj, tuple):
            self.__init__(cyAdjMapBQM(obj, vartype))  # via the map version
        elif hasattr(obj, "to_adjarray"):

            # this is not very elegent...
            other = obj.to_adjarray()
            self.invars_ = other.invars_
            self.outvars_ = other.outvars_
            self.offset_ = other.offset_
            self._label_to_idx = other._label_to_idx
            self._idx_to_label = other._idx_to_label
        else:
            # assume it's dense

            D = np.atleast_2d(np.asarray(obj, dtype=self.dtype))

            num_variables = D.shape[0]

            if D.ndim != 2 or num_variables != D.shape[1]:
                raise ValueError("expected dense to be a 2 dim square array")

            self.invars_.resize(num_variables)

            # we could grow the vectors going through it one at a time, but
            # in the interest of future-proofing we will go through once,
            # resize the outvars_ then go through it again to fill

            # figure out the degree of each variable and consequently the
            # number of interactions
            num_interactions = 0  # 2x num_interactions because count degree
            for ui in range(num_variables):
                degree = 0
                for vi in range(num_variables):
                    if ui != vi and (D[vi, ui] or D[ui, vi]):
                        degree += 1

                if ui < num_variables - 1:
                    self.invars_[ui + 1].first = degree + self.invars_[ui].first

                num_interactions += degree

            self.outvars_.resize(num_interactions)

            # todo: fix this, we're assigning twice
            for ui in range(num_variables):
                degree = 0
                for vi in range(num_variables):
                    if ui == vi:
                        self.invars_[ui].second = D[ui, vi]
                    elif D[vi, ui] or D[ui, vi]:
                        self.outvars_[self.invars_[ui].first + degree].first = vi
                        self.outvars_[self.invars_[ui].first + degree].second = D[vi, ui] + D[ui, vi]
                        degree += 1

    @property
    def num_variables(self):
        return num_variables(self.invars_, self.outvars_)

    @property
    def num_interactions(self):
        """int: The number of interactions in the model."""
        return num_interactions(self.invars_, self.outvars_)

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

        if vi < 0 or vi >= num_variables(self.invars_, self.outvars_):
            raise ValueError("{} is not a variable".format(v))

        return vi

    def iter_linear(self):
        cdef VarIndex vi
        cdef object v
        cdef Bias b

        for vi in range(num_variables(self.invars_, self.outvars_)):
            v = self._idx_to_label.get(vi, vi)
            b = self.invars_[vi].second

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
            for ui in range(num_variables(self.invars_, self.outvars_)):
                it_eit = neighborhood(self.invars_, self.outvars_, ui)
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

                it_eit = neighborhood(self.invars_, self.outvars_, ui)
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
        return as_numpy_scalar(get_linear(self.invars_,
                                          self.outvars_,
                                          self.label_to_idx(v)),
                               self.dtype)

    def get_quadratic(self, object u, object v):

        if u == v:
            raise ValueError('No interaction between {} and {}'.format(u, v))
        cdef VarIndex ui = self.label_to_idx(u)
        cdef VarIndex vi = self.label_to_idx(v)

        cdef pair[Bias, bool] out = get_quadratic(self.invars_, self.outvars_,
                                                  ui, vi)

        if not out.second:
            raise ValueError('No interaction between {} and {}'.format(u, v))

        return as_numpy_scalar(out.first, self.dtype)

    def set_linear(self, object v, Bias b):
        set_linear(self.invars_, self.outvars_, self.label_to_idx(v), b)

    def set_quadratic(self, object u, object v, Bias b):

        if u == v:
            raise ValueError('No interaction between {} and {}'.format(u, v))
        cdef VarIndex ui = self.label_to_idx(u)
        cdef VarIndex vi = self.label_to_idx(v)

        cdef bool isset = set_quadratic(self.invars_, self.outvars_, ui, vi, b)

        if not isset:
            raise ValueError('No interaction between {} and {}'.format(u, v))

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
            energies_view[row] = energy(self.invars_,
                                        self.outvars_,
                                        samples[row, :])

        return energies

    def to_lists(self):
        """Dump to two lists, mostly for testing"""
        return list(self.invars_), list(self.outvars_)


class AdjArrayBQM(cyAdjArrayBQM, BQM):
    __doc__ = cyAdjArrayBQM.__doc__
