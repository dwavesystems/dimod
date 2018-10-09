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
from collections import Sequence, Container, Mapping, ItemsView, ValuesView

import numpy as np

from six.moves import zip


class VariableIndexView(Sequence, Container):
    __slots__ = '_label', '_index'

    def __init__(self, iterable):
        self._index = index = {}

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
        return v in self._index

    def index(self, v):
        # we can speed this up because we're keeping a dict
        try:
            return self._index[v]
        except KeyError:
            raise ValueError('{!r} is not in {}'.format(v, self.__class__.__name__))

    def count(self, v):
        # everything is unique
        return int(v in self)

    def __eq__(self, other):
        return len(self) == len(other) and all(self[idx] == other[idx] for idx in range(len(self)))


class IndexView(Mapping):
    __slots__ = '_variables', '_data'

    def __init__(self, variables, data):
        self._variables = variables
        self._data = data

    def __getitem__(self, v):
        return self._data[self._variables.index(v)]

    def __iter__(self):
        return iter(self._variables)

    def __len__(self):
        return len(self._variables)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self._variables, self._data)

    def __str__(self):
        return str(dict(self))

    def values(self):
        return IndexValuesView(self)

    def items(self):
        return IndexItemsView(self)


class SampleView(IndexView):
    """View each row of the samples record as if it was a dict."""
    def __repr__(self):
        return str(self)


class IndexItemsView(ItemsView):
    """Faster read access to the numpy array"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return zip(self._mapping._variables, self._mapping._data.flat)


class IndexValuesView(ValuesView):
    """Faster read access to the numpy array"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return iter(self._mapping._data.flat)
