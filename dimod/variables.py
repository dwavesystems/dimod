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
from numbers import Integral, Number

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from operator import eq

from dimod.decorators import lockable_method
from dimod.utilities import iter_safe_relabels

__all__ = ['Variables']


class CallableDict(abc.Callable, dict):
    """Dict that can be accessed like a function."""
    __slots__ = ()

    def __call__(self, v):
        if v not in self:
            raise ValueError('missing element {!r}'.format(v))
        return self[v]


def iter_serialize_variables(variables):
    # want to handle things like numpy numbers and fractions that do not
    # serialize so easy
    for v in variables:
        if isinstance(v, Integral):
            yield int(v)
        elif isinstance(v, Number):
            yield float(v)
        elif isinstance(v, str):
            yield v
        # we want Collection, but that's not available in py3.5
        elif isinstance(v, (abc.Sequence, abc.Set)):
            yield tuple(iter_serialize_variables(v))
        else:
            yield v


def iter_deserialize_variables(variables):
    # convert list back into tuples
    for v in variables:
        # we want Collection, but that's not available in py3.5
        if isinstance(v, (abc.Sequence, abc.Set)) and not isinstance(v, str):
            yield tuple(iter_deserialize_variables(v))
        else:
            yield v


class Variables(abc.Sequence, abc.Set):
    """set-like and list-like variable tracking.

    Args:
        iterable: An iterable of variable labels.

    """
    __slots__ = ('_label', 'index', '_writeable')

    def __init__(self, iterable):
        self.index = index = CallableDict()

        def _iter():
            idx = 0
            for v in iterable:
                if v in index:
                    continue
                index[v] = idx
                idx += 1
                yield v
        self._label = list(_iter())

    def __getitem__(self, i):
        return self._label[i]

    # support python2 pickle
    def __getstate__(self):
        return {'_label': self._label, 'index': self.index}

    def __len__(self):
        return len(self._label)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._label)

    # support python2 pickle
    def __setstate__(self, state):
        for attr, obj in state.items():
            setattr(self, attr, obj)

    def __str__(self):
        return str(self._label)

    def __iter__(self):
        return iter(self._label)

    def __contains__(self, v):
        # we can speed this up because we're keeping a dict
        try:
            return v in self.index
        except TypeError:
            # unhashable objects
            return False

    def __ne__(self, other):
        return not (self == other)

    def __eq__(self, other):
        if isinstance(other, abc.Sequence):
            return len(self) == len(other) and all(map(eq, self, other))
        elif isinstance(other, abc.Set):
            return not (self ^ other)
        else:
            return False

    @property
    def is_writeable(self):
        return getattr(self, '_writeable', True)

    @is_writeable.setter
    def is_writeable(self, b):
        self._writeable = bool(b)

    # index method is overloaded by __init__

    def count(self, v):
        # everything is unique
        return int(v in self)

    def to_serializable(self):
        return list(iter_serialize_variables(self))

    @lockable_method
    def relabel(self, mapping):

        for submap in iter_safe_relabels(mapping, self):

            label = self._label
            index = self.index

            for old, new in submap.items():
                if old not in self:
                    continue

                label[index[old]] = new
                index[new] = index[old]
                del index[old]
