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
from __future__ import annotations

import enum
import typing
import warnings

from collections.abc import Container, Iterable

import dimod.typing

__all__ = ['as_vartype',
           'Vartype', 'ExtendedVartype',
           'SPIN', 'BINARY', 'DISCRETE', 'INTEGER', 'REAL',
           ]


class Integers(Container):
    """Container for testing integer membership."""
    def __contains__(self, item):
        try:
            return int(item) == item
        except TypeError:
            return False

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __eq__(self, other):
        # All Integers objects are equal
        return type(self) is type(other)


class Real(Container):
    """Container for testing real membership."""
    def __contains__(self, item):
        try:
            return float(item) == item
        except TypeError:
            return False

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __eq__(self, other):
        # All Real objects are equal
        return type(self) is type(other)


class Vartype(enum.Enum):
    """An :py:class:`~enum.Enum` over the types of variables for quadratic models.

    Attributes:
        SPIN: Vartype for spin-valued binary quadratic models and variables of
           quadratic models that have values that are either -1 or 1.
        BINARY: Vartype for binary quadratic models and variables of quadratic
           models that have values that are either 0 or 1.
        INTEGER: Vartype for variables in quadratic models
            that have values of type int.
        REAL: Vartype for variables in quadratic models
            that have values of type float.

    """
    SPIN = frozenset({-1, 1})
    BINARY = frozenset({0, 1})
    INTEGER = DISCRETE = Integers()  # DISCRETE is an alias for INTEGER
    REAL = Real()

    # Dev note:
    # This allows us to treat QMs and BQMs in the same way, i.e. we can
    # do bqm.vartype(v).
    # It would be better to make BQM.vartype a method, but that would be an
    # enormous backward compatibility break.
    # I tried instead making a proxy object, but that doesn't allow syntax like
    # bqm.vartype is SPIN
    def __call__(self, v: typing.Optional[dimod.typing.Variable] = None) -> Vartype:
        return self


# Deprecated alias
ExtendedVartype = Vartype


SPIN = Vartype.SPIN
BINARY = Vartype.BINARY
INTEGER = Vartype.INTEGER
DISCRETE = Vartype.DISCRETE
REAL = Vartype.REAL


# when we drop 3.7 we can just use typing.Literal
VartypeLike = typing.Union[Vartype, str, frozenset]
"""Objects that can be interpreted as a variable type.

This includes:

* :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
* :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
* :class:`~dimod.Vartype.INTEGER`, ``'INTEGER'``
* :class:`~dimod.Vartype.REAL`, ``'REAL'``

"""


def _vartype_miss(vartype, extended):
    if extended:
        candidates = ("Vartype.SPIN, 'SPIN', {-1, 1}, "
                      "Vartype.BINARY, 'BINARY', {0, 1}, "
                      "Vartype.INTEGER, 'INTEGER', "
                      "Vartype.REAL, or 'REAL'")
    else:
        candidates = ("Vartype.SPIN, 'SPIN', {-1, 1}, "
                      "Vartype.BINARY, 'BINARY', or {0, 1}")
    return TypeError(f"expected input vartype to be one of: {candidates!s}; "
                     f"received {vartype!r}.")


# In the future we may wish to make extended default to True. However, for
# now even raising a PendingDeprecationWarning would be a big change.
def as_vartype(vartype: VartypeLike, extended: bool = False) -> Vartype:
    """Cast various inputs to a valid vartype object.

    Args:
        vartype (:class:`.Vartype`/str/set):
            Variable type. Accepted input values:

            * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        extended (bool, optional, default=False):
            If `True`, vartype can also be:

            * :class:`~dimod.Vartype.INTEGER`, ``'INTEGER'``
            * :class:`~dimod.Vartype.REAL`, ``'REAL'``


    Returns:
        :class:`.Vartype`: Either :class:`~dimod.Vartype.SPIN` or
        :class:`~dimod.Vartype.BINARY`. If `extended` is True, can also
        be :class:`~dimod.Vartype.INTEGER` or :class:`~dimod.Vartype.REAL`

    See also:
        :func:`~dimod.decorators.vartype_argument`

    """
    # can be done with singledispatch, but this function needs to be
    # performant and if branches are faster
    if isinstance(vartype, Vartype):
        pass  # already what we want
    elif isinstance(vartype, str):
        try:
            vartype = Vartype[vartype]
        except KeyError:
            raise _vartype_miss(vartype, extended) from None
    elif isinstance(vartype, frozenset):
        try:
            vartype = Vartype(vartype)
        except ValueError:
            raise _vartype_miss(vartype, extended) from None
    elif isinstance(vartype, Iterable):  # not frozenset or str
        try:
            vartype = Vartype(frozenset(vartype))
        except ValueError:
            raise _vartype_miss(vartype, extended) from None
    else:
        raise _vartype_miss(vartype, extended)

    if not extended and vartype != SPIN and vartype != BINARY:
        raise _vartype_miss(vartype, extended)

    return vartype
