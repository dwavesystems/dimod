# Copyright 2020 D-Wave Systems Inc.
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
from collections.abc import Sequence, Set

from dimod.bqm.adjvectorbqm import AdjVectorBQM
from dimod.core.bqm import BQM
from dimod.vartypes import as_vartype

__all__ = ['as_bqm']


def as_bqm(*args, cls=None, copy=False):
    """Convert the input to a binary quadratic model.

    Converts the following input formats to a binary quadratic model (BQM):

        as_bqm(vartype)
            Creates an empty binary quadratic model.

        as_bqm(bqm)
            Creates a BQM from another BQM. See `copy` and `cls` kwargs below.

        as_bqm(bqm, vartype)
            Creates a BQM from another BQM, changing to the appropriate
            `vartype` if necessary. See `copy` and `cls` kwargs below.

        as_bqm(n, vartype)
            Creates a BQM with `n` variables, indexed linearly from zero,
            setting all biases to zero.

        as_bqm(quadratic, vartype)
            Creates a BQM from quadratic biases given as a square array_like_
            or a dictionary of the form `{(u, v): b, ...}`. Note that when
            formed with SPIN-variables, biases on the diagonal are added to the
            offset.

        as_bqm(linear, quadratic, vartype)
            Creates a BQM from linear and quadratic biases, where `linear` is a
            one-dimensional array_like_ or a dictionary of the form
            `{v: b, ...}`, and `quadratic` is a square array_like_ or a
            dictionary of the form `{(u, v): b, ...}`. Note that when formed
            with SPIN-variables, biases on the diagonal are added to the offset.

        as_bqm(linear, quadratic, offset, vartype)
            Creates a BQM from linear and quadratic biases, where `linear` is a
            one-dimensional array_like_ or a dictionary of the form
            `{v: b, ...}`, and `quadratic` is a square array_like_ or a
            dictionary of the form `{(u, v): b, ...}`, and `offset` is a
            numerical offset. Note that when formed with SPIN-variables, biases
            on the diagonal are added to the offset.

    Args:
        *args:
            See above.

        cls (type/list, optional):
            Class of the returned BQM. If given as a list,
            the returned BQM is of one of the types in the list. Default is
            :class:`.AdjVectorBQM`.

        copy (bool, optional, default=False):
            If False, a new BQM is only constructed when
            necessary.

    Returns:
        A binary quadratic model.

    .. _array_like: https://numpy.org/doc/stable/user/basics.creation.html

    """

    if cls is None:
        if isinstance(args[0], BQM):
            cls = type(args[0])
        else:
            cls = AdjVectorBQM
    elif isinstance(cls, (Sequence, Set)):  # want Collection, but not in 3.5
        classes = tuple(cls)
        if not classes:
            raise ValueError("cls kwarg should be a type or a list of types")
        if type(args[0]) in classes:
            cls = type(args[0])
        else:
            # otherwise just pick the first one
            cls = classes[0]

    if isinstance(args[0], cls) and not copy:
        # this is the only case (currently) in which copy matters
        if len(args) == 1:
            return args[0]
        elif len(args) == 2:
            bqm, vartype = args
            if bqm.vartype is as_vartype(vartype):
                return bqm
            # otherwise we're doing a copy
        # otherwise we don't have a well-formed bqm input so pass off the check
        # to cls(*args)

    return cls(*args)
