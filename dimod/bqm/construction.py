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

    Many formats can be converted to a binary quadratic model:

        as_bqm(vartype)
            Creates an empty binary quadratic model.

        as_bqm(bqm)
            Creates a bqm from another bqm. See `copy` and `cls` kwargs below.

        as_bqm(bqm, vartype)
            Creates a bqm from another bqm, changing to the appropriate vartype
            if necessary. See `copy` and `cls` kwargs below.

        as_bqm(n, vartype)
            Make a bqm with all zero biases, where n is the number of nodes.

        as_bqm(Q, vartype)
            Where Q is a square array_like_ or a dictionary of the form
            `{(u, v): b, ...}`. Note that when formed with SPIN-variables,
            biases on the diagonal are added to the offset.

        as_bqm(L, Q, vartype)
            Where L is a one dimensional array_like_ or a dictionary of the form
            `{v: b, ...}`.
            And where Q is a square array_like_ or a dictionary of the form
            `{(u, v): b, ...}`. Note that when formed with SPIN-variables,
            biases on the diagonal are added to the offset.

        as_bqm(L, Q, offset, vartype)
            Where L is a one dimensional array_like_ or a dictionary of the form
            `{v: b, ...}`.
            And where Q is a square array_like_ or a dictionary of the form
            `{(u, v): b, ...}`. Note that when formed with SPIN-variables,
            biases on the diagonal are added to the offset.
            And where offset is a numerical offset.

    Args:
        *args:
            See above.

        cls (type/list, optional, default=:class:`.AdjVectorBQM`):
            The class of the returned binary quadratic model. If given as a list
            the returned bqm will be of one of the types in the list.

        copy (bool, optional, default=False):
            If False, a new binary quadratic model is only constructed when
            necessary.

    Returns:
        A binary quadratic model.

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
