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

from libcpp cimport bool
from libcpp.map cimport map
from libcpp.pair cimport pair
from libcpp.vector cimport vector

from dimod.bqm.common cimport VarIndex, Bias

ctypedef map[VarIndex, Bias].const_iterator NeighborIterator


cdef class cyAdjMapBQM:
    """A binary quadratic model where the neighborhoods are c++ maps.

    Can be created in several ways:

        AdjMapBQM(vartype)
            Creates an empty binary quadratic model.

        AdjMapBQM(bqm)
            Creates a BQM from another BQM. See `copy` and `cls` kwargs below.

        AdjMapBQM(bqm, vartype)
            Creates a BQM from another BQM, changing to the appropriate
            `vartype` if necessary.

        AdjMapBQM(n, vartype)
            Make a BQM with `n` variables, indexed linearly from zero, setting
            all biases to zero.

        AdjMapBQM(quadratic, vartype)
            Creates a BQM from quadratic biases given as a square array_like_
            or a dictionary of the form `{(u, v): b, ...}`. Note that when
            formed with SPIN-variables, biases on the diagonal are added to the
            offset.

        AdjMapBQM(linear, quadratic, vartype)
            Creates a BQM from linear and quadratic biases, where `linear` is a
            one-dimensional array_like_ or a dictionary of the form
            `{v: b, ...}`, and `quadratic` is a square array_like_ or a
            dictionary of the form `{(u, v): b, ...}`. Note that when formed
            with SPIN-variables, biases on the diagonal are added to the offset.

        AdjMapBQM(linear, quadratic, offset, vartype)
            Creates a BQM from linear and quadratic biases, where `linear` is a
            one-dimensional array_like_ or a dictionary of the form
            `{v: b, ...}`, and `quadratic` is a square array_like_ or a
            dictionary of the form `{(u, v): b, ...}`, and `offset` is a
            numerical offset. Note that when formed with SPIN-variables, biases
            on the diagonal are added to the offset.

    .. _array_like: https://docs.scipy.org/doc/numpy/user/basics.creation.html

    """
    cdef vector[pair[map[VarIndex, Bias], Bias]] adj_

    cdef Bias offset_

    cdef readonly object vartype
    """The variable type, :class:`.Vartype.SPIN` or :class:`.Vartype.BINARY`."""

    cdef readonly object dtype
    """The data type of the linear biases, int8."""

    cdef readonly object itype
    """The data type of the indices, uint32."""

    cdef readonly object ntype
    """The data type of the neighborhood indices, varies by platform."""

    # these are not public because the user has no way to access the underlying
    # variable indices
    cdef object _label_to_idx
    cdef object _idx_to_label

    cdef VarIndex label_to_idx(self, object) except *
