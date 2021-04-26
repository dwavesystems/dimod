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
from dimod.bqm.cppbqm cimport AdjVectorBQM as cppAdjVectorBQM
from dimod.bqm.common cimport VarIndex, Bias

cdef class cyAdjVectorBQM:
    """A binary quadratic model where the neighborhoods are C++ vectors.

    Can be created in several ways:

        AdjVectorBQM(vartype)
            Creates an empty binary quadratic model (BQM).

        AdjVectorBQM(bqm)
            Creates a BQM from another BQM. See `copy` and `cls` kwargs below.

        AdjVectorBQM(bqm, vartype)
            Creates a BQM from another BQM, changing to the appropriate
            `vartype` if necessary.

        AdjVectorBQM(n, vartype)
            Creates a BQM with `n` variables, indexed linearly from zero,
            setting all biases to zero.

        AdjVectorBQM(quadratic, vartype)
            Creates a BQM from quadratic biases given as a square array_like_
            or a dictionary of the form `{(u, v): b, ...}`. Note that when
            formed with SPIN-variables, biases on the diagonal are added to the
            offset.

        AdjVectorBQM(linear, quadratic, vartype)
            Creates a BQM from linear and quadratic biases, where `linear` is a
            one-dimensional array_like_ or a dictionary of the form
            `{v: b, ...}`, and `quadratic` is a square array_like_ or a
            dictionary of the form `{(u, v): b, ...}`. Note that when formed
            with SPIN-variables, biases on the diagonal are added to the offset.

        AdjVectorBQM(linear, quadratic, offset, vartype)
            Creates a BQM from linear and quadratic biases, where `linear` is a
            one-dimensional array_like_ or a dictionary of the form
            `{v: b, ...}`, and `quadratic` is a square array_like_ or a
            dictionary of the form `{(u, v): b, ...}`, and `offset` is a
            numerical offset. Note that when formed with SPIN-variables, biases
            on the diagonal are added to the offset.

    Notes:

        The AdjVectorBQM is implemented using an adjacency structure where the
        neighborhoods are implemented as C++ vectors.

        Advantages:

        - Supports incremental construction
        - Fast iteration over the biases

        Disadvantages:

        - Only supports float64 biases

        Intended Use:

        - When performance is important and the use case requires incremental
          construction
        - This should be the default BQM type for large problems where
          arbitrary types are not needed

    Examples:

        >>> import numpy as np
        >>> from dimod import AdjVectorBQM

        Construct from dicts.

        >>> AdjVectorBQM({'a': -1.0}, {('a', 'b'): 1.0}, 'SPIN')
        AdjVectorBQM({a: -1.0, b: 0.0}, {('a', 'b'): 1.0}, 0.0, 'SPIN')

        Incremental Construction.

        >>> bqm = AdjVectorBQM('SPIN')
        >>> bqm.add_variable('a')
        'a'
        >>> bqm.add_variable()
        1
        >>> bqm.set_quadratic('a', 1, 3.0)
        >>> bqm
        AdjVectorBQM({a: 0.0, 1: 0.0}, {('a', 1): 3.0}, 0.0, 'SPIN')

    .. _array_like: https://numpy.org/doc/stable/user/basics.creation.html

    """
    cdef cppAdjVectorBQM[VarIndex, Bias] bqm_

    cdef Bias offset_

    cdef readonly object vartype
    """Variable type, :class:`.Vartype.SPIN` or :class:`.Vartype.BINARY`."""

    cdef readonly object dtype
    """Data type of the linear biases, float64."""

    cdef readonly object itype
    """Data type of the indices, uint32."""

    cdef readonly object ntype
    """Data type of the neighborhood indices, varies by platform."""

    # these are not public because the user has no way to access the underlying
    # variable indices
    cdef object _label_to_idx
    cdef object _idx_to_label

    cdef VarIndex label_to_idx(self, object) except *
