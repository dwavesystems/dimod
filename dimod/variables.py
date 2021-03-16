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

import collections.abc as abc
import io
import warnings

from numbers import Integral, Number
from operator import eq
from pprint import PrettyPrinter

from dimod.decorators import lockable_method
from dimod.utilities import iter_safe_relabels

__all__ = ['Variables']


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
    """Set-like and list-like variables tracking.

    Args:
        iterable (iterable):
            An iterable of labels. Duplicate labels are ignored. All labels
            must be hashable.

    """
    __slots__ = ('_idx_to_label',
                 '_label_to_idx',
                 '_stop',
                 )

    def __init__(self, iterable=None):
        self._idx_to_label = dict()
        self._label_to_idx = dict()
        self._stop = 0

        if iterable is not None:
            for v in iterable:
                self._append(v, permissive=True)

    def __contains__(self, v):
        try:
            in_range = (isinstance(v, Number)
                        and v == int(v)
                        and 0 <= v < self._stop)
        except AttributeError:
            in_range = False

        try:
            return (in_range and v not in self._idx_to_label
                    or v in self._label_to_idx)
        except TypeError:
            # unhashable
            return False

    def __eq__(self, other):
        if isinstance(other, abc.Sequence):
            return len(self) == len(other) and all(map(eq, self, other))
        elif isinstance(other, abc.Set):
            return not (self ^ other)
        else:
            return False

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError("index must be an integer.")

        given = idx  # for error message

        # handle negative indexing
        if idx < 0:
            idx = self._stop + idx

        if idx >= self._stop:
            raise IndexError('index {} out of range'.format(given))

        return self._idx_to_label.get(idx, idx)

    def __len__(self):
        return self._stop

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        stream = io.StringIO()
        stream.write(type(self).__name__)
        stream.write('(')
        if self:
            if self.is_range and len(self) > 10:
                # 10 is arbitrary, but the idea is we want to truncate
                # longer variables that are integer-labelled
                stream.write(repr(range(self._stop)))
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
    def is_range(self):
        return not self._label_to_idx

    def _append(self, v=None, *, permissive=False):
        """Append a new variable.

        This method is semi-public. it is intended to be used by
        classes that have :class:`.Variables` as an attribute, not by the
        the user.
        """

        if v is None:
            # handle the easy case
            if self.is_range:
                self._stop += 1
                return

            # we need to pick a new label
            v = self._stop

            if v not in self:
                # it's free, so we can stop
                self._stop += 1
                return

            # there must be a free integer available
            v = 0
            while v in self:
                v += 1

        elif v in self:
            if permissive:
                return
            else:
                raise ValueError('{!r} is already a variable'.format(v))

        idx = self._stop

        if idx != v:
            self._label_to_idx[v] = idx
            self._idx_to_label[idx] = v

        self._stop += 1

    def _relabel(self, mapping):
        """Relabel the variables in-place.

        This method is semi-public. it is intended to be used by
        classes that have :class:`.Variables` as an attribute, not by the
        the user.
        """
        for submap in iter_safe_relabels(mapping, self):
            for old, new in submap.items():
                if old == new:
                    continue

                idx = self._label_to_idx.pop(old, old)

                if new != idx:
                    self._label_to_idx[new] = idx
                    self._idx_to_label[idx] = new  # overwrites old idx
                else:
                    self._idx_to_label.pop(idx, None)

    def _relabel_as_integers(self):
        """Relabel the variables as integers in-place.

        This method is semi-public. it is intended to be used by
        classes that have :class:`.Variables` as an attribute, not by the
        the user.
        """
        mapping = self._idx_to_label.copy()
        self._idx_to_label.clear()
        self._label_to_idx.clear()
        return mapping

    def count(self, v):
        # everything is unique
        return int(v in self)

    def index(self, v, permissive=False):
        """Return the index of v.

        Args:
            v (hashable):
                A variable.

            permissive (bool, optional, default=False):
                If True, the variable will be inserted, guaranteeing an index
                can be returned.

        Returns:
            int: The index of the given variable.

        Raises:
            ValueError: If the variable is not present and `permissive` is
            False.

        """
        if permissive:
            self._append(v, permissive=True)
        elif v not in self:
            raise ValueError('unknown variable {!r}'.format(v))
        return self._label_to_idx.get(v, v)

    def to_serializable(self):
        """Return an object that (should be) json-serializable.

        Returns:
            list: A list of (hopefully) json-serializable objects. Handles some
            common cases like NumPy scalars.
            See :func:`iter_serialize_variables`.

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
