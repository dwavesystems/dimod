# Copyright 2018 D-Wave Systems Inc.
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
"""
Enumeration of valid variable types for binary quadratic models.

Examples:

    :class:`.Vartype` is an :py:class:`~enum.Enum`. Each vartype has a value and
    a name.

    >>> vartype = dimod.SPIN
    >>> vartype.name
    'SPIN'
    >>> vartype.value == {-1, +1}
    True

    >>> vartype = dimod.BINARY
    >>> vartype.name
    'BINARY'
    >>> vartype.value == {0, 1}
    True

    The :func:`.as_vartype` function allows the user to provide several
    convenient forms.

    >>> from dimod import as_vartype

    >>> as_vartype(dimod.SPIN) is dimod.SPIN
    True
    >>> as_vartype('SPIN') is dimod.SPIN
    True
    >>> as_vartype({-1, 1}) is dimod.SPIN
    True

    >>> as_vartype(dimod.BINARY) is dimod.BINARY
    True
    >>> as_vartype('BINARY') is dimod.BINARY
    True
    >>> as_vartype({0, 1}) is dimod.BINARY
    True

"""
import enum
import typing

__all__ = ['as_vartype',
           'Vartype', 'ExtendedVartype',
           'SPIN', 'BINARY', 'DISCRETE',
           ]


class Vartype(enum.Enum):
    """An :py:class:`~enum.Enum` over the types of variables for the binary
    quadratic model.

    Attributes:
        SPIN (:class:`.Vartype`): Vartype for spin-valued models; variables of
           the model are either -1 or 1.
        BINARY (:class:`.Vartype`): Vartype for binary models; variables of the
           model are either 0 or 1.

    """
    SPIN = frozenset({-1, 1})
    BINARY = frozenset({0, 1})


# unfortunately one cannot subclass/extend Enum and the assumption that there
# are only two is baked into too many places to just add DISCRETE to Vartype
class ExtendedVartype(enum.Enum):
    DISCRETE = int


SPIN = Vartype.SPIN
BINARY = Vartype.BINARY
DISCRETE = ExtendedVartype.DISCRETE


VartypeLike = typing.Union[Vartype, str, frozenset]
ExtendedVartypeLike = typing.Union[VartypeLike, ExtendedVartype]


def as_vartype(vartype: ExtendedVartypeLike, extended: bool = False):
    """Cast various inputs to a valid vartype object.

    Args:
        vartype (:class:`.Vartype`/str/set):
            Variable type. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        extended (bool, optional, default=False):
            If `True`, vartype can also be:

            * :class:`.ExtendedVartype.DISCRETE`, ``'DISCRETE'``

    Returns:
        :class:`.Vartype`: Either :class:`.Vartype.SPIN` or
        :class:`.Vartype.BINARY`. If `extended` is True, can also
        be :class:`.ExtendedVartype.DISCRETE`.

    See also:
        :func:`~dimod.decorators.vartype_argument`

    """
    if isinstance(vartype, Vartype):
        return vartype

    try:
        if isinstance(vartype, str):
            vartype = Vartype[vartype]  # raises KeyError
        elif isinstance(vartype, frozenset):
            vartype = Vartype(vartype)  # raise ValueError
        else:
            # raises ValueError if not a bqm
            # raises TypeError if not iterable
            vartype = Vartype(frozenset(vartype))
    except (ValueError, KeyError, TypeError):
        # avoid the "During handling of the above exception..." message that
        # removes the full traceback
        pass
    else:
        return vartype

    if not extended:
        raise TypeError(("expected input vartype to be one of: "
                         "Vartype.SPIN, 'SPIN', {-1, 1}, "
                         "Vartype.BINARY, 'BINARY', or {0, 1}."))

    if isinstance(vartype, ExtendedVartype):
        return vartype

    try:
        if isinstance(vartype, str):
            vartype = ExtendedVartype[vartype]
        else:
            vartype = ExtendedVartype(vartype)
    except (ValueError, KeyError, TypeError):
        # avoid the "During handling of the above exception..." message
        # that removes the full traceback
        pass
    else:
        return vartype

    raise TypeError(("expected input vartype to be one of: "
                     "Vartype.SPIN, 'SPIN', {-1, 1}, "
                     "Vartype.BINARY, 'BINARY', {0, 1}, "
                     "ExtendedVartype.DISCRETE, or 'DISCRETE'."))
