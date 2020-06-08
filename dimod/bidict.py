# Copyright 2020 D-Wave Systems Inc.
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

from collections.abc import Mapping, MutableMapping


__all__ = ['BiDict']


class BiDict(MutableMapping):
    """A BiDict is a dictionary with fast reverse lookup.

    All dictionary methods are supported. In addition there is an `inverse`
    attribute, which it itself a `BiDict`, storing the values as keys
    and vice versa.

    Attributes:

        inverse:
            Another `BiDict`, equivalent to one constructed by
            `{value: key for key, value in d}`.

    Examples:

        >>> from dimod.bidict import BiDict

        `BiDict` can be constructed and interacted with just like a normal
        dictionary, but it also has an `inverse` attribute, which is itself
        another (inversed) `BiDict`.

        >>> bidict = BiDict(a=1, b=2)
        >>> bidict['a']
        1
        >>> bidict.inverse[1]
        'a'
        >>> bidict.inverse.inverse['a']
        1

        Note that this means that the values must also be unique.

        >>> bidict = BiDict(a=1)
        >>> bidict['b'] = 1
        >>> print(bidict)
        {'b': 1}
        >>> print(bidict.inverse)
        {1: 'b'}

    """
    __slots__ = ('_forward', 'inverse')

    def __init__(self, *args, **kwargs):
        self._forward = forward = dict(*args, **kwargs)
        self.inverse = inverse = BiDict.__init_inverse__(self)

        if len(forward) != len(inverse):
            # forward contained some non-unique values, so do one more pass
            forward.clear()
            forward.update((value, key) for key, value in inverse.items())

    @classmethod
    def __init_inverse__(cls, forward):
        obj = cls.__new__(cls)  # skip making dicts
        obj._forward = dict((value, key) for key, value in forward.items())
        obj.inverse = forward
        return obj

    def __delitem__(self, key):
        value = self._forward.pop(key)  # raises exception correctly
        self.inverse._forward.pop(value)

    def __getitem__(self, key):
        return self._forward[key]

    def __iter__(self):
        return iter(self._forward)

    def __len__(self):
        return len(self._forward)

    def __repr__(self):
        return '{!s}({!s})'.format(type(self).__name__, self)

    def __setitem__(self, key, value):
        try:
            oldval = self._forward[key]
        except KeyError:
            pass  # not present
        else:
            del self.inverse._forward[oldval]

        try:
            oldkey = self.inverse[value]
        except KeyError:
            pass  # not present
        else:
            del self._forward[oldkey]

        self._forward[key] = value
        self.inverse._forward[value] = key

    def __str__(self):
        return str(dict(self))

    # we don't need to overwrite these mixins but we do for performance

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented

        if len(self) != len(other):
            return False

        return all(other[key] == value for key, value in self.items())

    def clear(self):
        self._forward.clear()
        self.inverse._forward.clear()

    def items(self):
        return self._forward.items()

    def keys(self):
        return self._forward.keys()

    def values(self):
        return self._forward.values()
