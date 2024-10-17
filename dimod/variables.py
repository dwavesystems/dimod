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
A class and utilities for encoding variable objects.

The :class:`Variables` class is intended to be used as an attribute of other
classes, such as :class:`.DiscreteQuadraticModel` and :class:`~dimod.SampleSet`.

The requirements for the class are:
    *   Have a minimal memory footprint when the variables are labeled `[0, n)`
    *   Behave like a list for iteration and getting items
    *   Behave like a set for determining if it contains an item
    *   Constant time for finding the index of an item

"""

import collections.abc as abc
import io
import typing
import warnings

from numbers import Integral, Number
from operator import eq
from pprint import PrettyPrinter

from dimod.cyvariables import cyVariables
from dimod.typing import Variable


__all__ = ['Variables']


def serialize_variable(v):
    # want to handle things like numpy numbers and fractions that do not
    # serialize so easy
    if isinstance(v, Integral):
        return int(v)
    elif isinstance(v, Number):
        return float(v)
    elif isinstance(v, str):
        return v
    elif isinstance(v, abc.Collection):
        return tuple(iter_serialize_variables(v))
    else:
        return v


def iter_serialize_variables(variables):
    yield from map(serialize_variable, variables)


def deserialize_variable(v):
    if isinstance(v, abc.Collection) and not isinstance(v, str):
        return tuple(iter_deserialize_variables(v))
    else:
        return v


def iter_deserialize_variables(variables):
    yield from map(deserialize_variable, variables)


class Variables(cyVariables, abc.Set[Variable], abc.Sequence[Variable]):
    """Set-like and list-like variables tracking.

    Args:
        iterable (iterable[:class:`~dimod.typing.Variable`], optional):
            An :term:`iterable` of labels. Duplicate labels are ignored.
            All labels must be :term:`hashable`.

    Examples:

        The ``Variables`` object encodes an ordered set of variables.

        >>> variables = dimod.variables.Variables(['a', 'b', 0, 1])
        >>> print(variables)
        Variables(['a', 'b', 0, 1])

        ``Variables`` is a :term:`sequence`.

        >>> variables[0]  # O(1) element access using integer indices
        'a'
        >>> len(variables)  # defines a length
        4

        Unlike most other sequence types, ``Variables`` provides `O(1)`
        lookup for the index of an element.

        >>> 0 in variables  # O(1) lookup
        True
        >>> variables.index('b')  # O(1) lookup
        1

        Therefore ``Variables`` is also a set.

        >>> variables & [0, 'a', 'f']
        Variables([0, 'a'])

        ``Variables`` inherits from :class:`~collections.abc.Sequence` and
        :class:`~collections.abc.Set` so it inherits all of the provided
        mixin methods.

        >>> list(reversed(variables))
        [1, 0, 'b', 'a']
        >>> variables.count('b')
        1
        >>> variables.count('hello')
        0

    """
    def __eq__(self, other: object) -> bool:
        if isinstance(other, abc.Sequence):
            return len(self) == len(other) and all(map(eq, self, other))
        elif isinstance(other, abc.Set):
            return not (self ^ other)
        else:
            return False

    def __ne__(self, other: object) -> bool:
        return not (self == other)

    def __repr__(self) -> str:
        stream = io.StringIO()
        stream.write(type(self).__name__)
        stream.write('(')
        if self:
            if self.is_range and len(self) > 10:
                # 10 is arbitrary, but the idea is we want to truncate
                # longer variables that are integer-labelled
                stream.write(repr(range(len(self))))
            else:
                stream.write('[')
                iterator = iter(self)
                stream.write(repr(next(iterator)))
                for v in iterator:
                    stream.write(', ')
                    stream.write(repr(v))
                stream.write(']')
        stream.write(')')
        return stream.getvalue()

    @property
    def is_range(self) -> bool:
        """Return True if the variables are labeled `[0,n)`."""
        return self._is_range()

    def to_serializable(self) -> list[typing.Union[int, float, str, tuple]]:
        """Return an object that is JSON-serializable.

        Returns:
            A list of JSON-serializable objects. Handles some common cases like
            NumPy scalars.

        """
        return list(iter_serialize_variables(self))


# register the various objects with prettyprint
def _pprint_variables(printer, variables, stream, indent, *args, **kwargs):
    if not variables or variables.is_range:
        stream.write(repr(variables))
    else:
        indent += stream.write(type(variables).__name__)
        indent += stream.write('(')
        printer._pprint_list(variables, stream, indent, *args, **kwargs)
        indent += stream.write(')')


try:
    PrettyPrinter._dispatch[Variables.__repr__] = _pprint_variables
except AttributeError:
    # we're using some internal stuff in PrettyPrinter so let's silently fail
    # for that
    pass
