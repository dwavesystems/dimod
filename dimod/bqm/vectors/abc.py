try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import numpy as np


class Vector(abc.MutableSequence):
    """Allows for nicer isinstance checking for the Vector classes"""
    __slots__ = ()

    def __str__(self):
        return str(list(self))

    def __repr__(self):
        return 'vector(%s, dtype=%r)' % (self, np.asarray(self).dtype.name)

    def __ne__(self, other):
        return not (self == other)

    def __eq__(self, other):
        return np.array_equal(self, other)


class ViewError(Exception):
    """"""
