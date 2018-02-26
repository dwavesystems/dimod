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
