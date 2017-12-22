import enum


class VARTYPES(enum.Enum):
    SPIN = frozenset((-1, 1))
    BINARY = frozenset((0, 1))
    UNDEFINED = None
SPIN = VARTYPES.SPIN
BINARY = VARTYPES.BINARY
UNDEFINED = VARTYPES.UNDEFINED
