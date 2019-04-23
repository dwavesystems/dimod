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
try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from operator import eq

from dimod.utilities import resolve_label_conflict

from six import PY3
from six.moves import map


class CallableDict(abc.Callable, dict):
    """Dict that can be accessed like a function."""
    __slots__ = ()

    def __call__(self, v):
        if v not in self:
            raise ValueError('missing element {!r}'.format(v))
        return self[v]


class Variables(abc.Sequence, abc.Set):
    """set-like and list-like variable tracking.

    Args:
        iterable: An iterable of variable labels.

    """
    __slots__ = '_label', 'index'

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

    def __len__(self):
        return len(self._label)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self._label)

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

    # index method is overloaded by __init__

    def count(self, v):
        # everything is unique
        return int(v in self)

    def relabel(self, mapping):

        # put the new labels into a set for fast lookup
        try:
            new_labels = set(mapping.values())
        except TypeError:
            # when new labels are not hashable
            raise ValueError("mapping targets must be hashable objects")

        if PY3:
            old_labels = mapping.keys()
        else:
            # we actually want old_labels to be a keysview like in python3
            # but since we don't get that in python2 we just use the mapping
            # itself and treat it as a set rather than a dict.
            old_labels = mapping

        for v in new_labels:
            if v in self and v not in old_labels:
                msg = ("A variable cannot be relabeled {!r} without also "
                       "relabeling the existing variable of the same name")
                raise ValueError(msg.format(v))

        if any(v in new_labels for v in old_labels):
            old_to_intermediate, intermediate_to_new = resolve_label_conflict(mapping, old_labels, new_labels)

            self.relabel(old_to_intermediate)
            self.relabel(intermediate_to_new)

            return

        label = self._label
        index = self.index

        for old, new in mapping.items():
            if old not in self:
                continue

            label[index[old]] = new
            index[new] = index[old]
            del index[old]
