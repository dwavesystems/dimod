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
from dimod.bqm.cppbqm cimport AdjArrayBQM as cppAdjArrayBQM
from dimod.bqm.common cimport VarIndex, Bias


cdef class cyAdjArrayBQM:
    """A binary quadratic model structured as two C++ vectors.

    Can be created in several ways:

        AdjArrayBQM(vartype)
            Creates an empty binary quadratic model.

        AdjArrayBQM(bqm)
            Creates a BQM from another BQM. See `copy` and `cls` kwargs below.

        AdjArrayBQM(bqm, vartype)
            Creates a BQM from another BQM, changing to the specified
            `vartype` if necessary.

        AdjArrayBQM(n, vartype)
            Creates a BQM with `n` variables, indexed linearly from zero,
            setting all biases to zero.

        AdjArrayBQM(quadratic, vartype)
            Creates a BQM from quadratic biases given as a square array_like_
            or a dictionary of the form `{(u, v): b, ...}`. Note that when
            formed with SPIN-variables, biases on the diagonal are added to the
            offset.

        AdjArrayBQM(linear, quadratic, vartype)
            Creates a BQM from linear and quadratic biases, where `linear` is a
            one-dimensional array_like_ or a dictionary of the form
            `{v: b, ...}`, and `quadratic` is a square array_like_ or a
            dictionary of the form `{(u, v): b, ...}`. Note that when formed
            with SPIN-variables, biases on the diagonal are added to the offset.

        AdjArrayBQM(linear, quadratic, offset, vartype)
            Creates a BQM from linear and quadratic biases, where `linear` is a
            one-dimensional array_like_ or a dictionary of the form
            `{v: b, ...}`, and `quadratic` is a square array_like_ or a
            dictionary of the form `{(u, v): b, ...}`, and `offset` is a
            numerical offset. Note that when formed with SPIN-variables, biases
            on the diagonal are added to the offset.

    Notes:

        The AdjArrayBQM is implemented using two C++ vectors. The first
        vector contains the linear biases and the index of the start of each
        variable's neighborhood in the second vector. The second
        vector contains the neighboring variables and their associated quadratic
        biases. The vectors, once initialized, are not resized.

        Advantages:

        - Very fast iteration over the biases

        Disadvantages:

        - Does not support incremental construction
        - Only supports float64 biases

        Intended Use:

        - When performance is important and the BQM can be treated as read-only

    Examples:

        >>> import numpy as np
        >>> from dimod import AdjArrayBQM

        Construct from a NumPy array.

        >>> AdjArrayBQM(np.triu(np.ones((2, 2))), 'BINARY')
        AdjArrayBQM({0: 1.0, 1: 1.0}, {(0, 1): 1.0}, 0.0, 'BINARY')

        Construct from dicts.

        >>> AdjArrayBQM({'a': -1}, {('a', 'b'): 1}, 'SPIN')
        AdjArrayBQM({a: -1.0, b: 0.0}, {('a', 'b'): 1.0}, 0.0, 'SPIN')

    .. _array_like: https://docs.scipy.org/doc/numpy/user/basics.creation.html

    """
    cdef cppAdjArrayBQM[VarIndex, Bias] bqm_

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
