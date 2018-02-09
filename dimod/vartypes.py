"""
Vartype
--------

Vartype is an enumeration of the valid types of binary variables
for binary quadratic models.

Examples:
    >>> vartype = dimod.Vartype.SPIN
    >>> print(vartype)
    Vartype.SPIN
    >>> isinstance(vartype, dimod.Vartype)
    True

    Access can also be by value or name.

    >>> print(dimod.Vartype({0, 1}))
    Vartype.BINARY
    >>> print(dimod.Vartype['SPIN'])
    Vartype.SPIN

    To check correctness, use the `.value` parameter.

    >>> sample = {'u': -1, 'v': 1}
    >>> vartype = dimod.Vartype.SPIN
    >>> all(val in vartype.value for val in sample.values())
    True

    The different Vartypes are also in the main namespace
    for easy access.

    >>> vartype = dimod.SPIN
    >>> print(vartype)
    Vartype.SPIN
    >>> vartype = dimod.BINARY
    >>> print(vartype)
    Vartype.BINARY

"""
import enum

__all__ = ['Vartype', 'SPIN', 'BINARY']


class Vartype(enum.Enum):
    """An :py:class:`~enum.Enum` over the types of variables for the binary quadratic model.

    Attributes:
        SPIN (:class:`.Vartype`): The vartype for spin-valued models. That
            is the variables of the model are either -1 or 1.
        BINARY (:class:`.Vartype`): The vartype for binary models. That is
            the variables of the model are either 0 or 1.

    """
    SPIN = frozenset({-1, 1})
    BINARY = frozenset({0, 1})

SPIN = Vartype.SPIN
BINARY = Vartype.BINARY
