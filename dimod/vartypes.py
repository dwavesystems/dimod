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
# ================================================================================================
"""
Enumeration of valid variable types for binary quadratic models.

Examples:
    This example shows easy access to different Vartypes, which are in the main
    namespace.

    >>> vartype = dimod.SPIN
    >>> print(vartype)
    Vartype.SPIN
    >>> vartype = dimod.BINARY
    >>> print(vartype)
    Vartype.BINARY
    >>> vartype = dimod.Vartype.SPIN
    >>> print(vartype)
    Vartype.SPIN
    >>> isinstance(vartype, dimod.Vartype)
    True

    This example shows access by value or name.

    >>> print(dimod.Vartype({0, 1}))
    Vartype.BINARY
    >>> print(dimod.Vartype['SPIN'])
    Vartype.SPIN

    This example uses the `.value` parameter to validate.

    >>> sample = {'u': -1, 'v': 1}
    >>> vartype = dimod.Vartype.SPIN
    >>> all(val in vartype.value for val in sample.values())
    True

"""
import enum

__all__ = ['Vartype', 'SPIN', 'BINARY']


class Vartype(enum.Enum):
    """An :py:class:`~enum.Enum` over the types of variables for the binary quadratic model.

    Attributes:
        SPIN (:class:`.Vartype`): Vartype for spin-valued models; variables of
           the model are either -1 or 1.
        BINARY (:class:`.Vartype`): Vartype for binary models; variables of the
           model are either 0 or 1.

    """
    SPIN = frozenset({-1, 1})
    BINARY = frozenset({0, 1})


SPIN = Vartype.SPIN
BINARY = Vartype.BINARY
