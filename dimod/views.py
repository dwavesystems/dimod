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
try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from operator import eq

import numpy as np

from six.moves import zip, map

from dimod.utilities import resolve_label_conflict


class CallableDict(abc.Callable, dict):
    """Dict that can be accessed like a function."""
    __slots__ = ()

    def __call__(self, v):
        try:
            return self[v]
        except KeyError:
            raise ValueError('missing element {!r}'.format(v))


class Variables(abc.Sequence, abc.Container):
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
        return v in self.index

    def __ne__(self, other):
        return not (self == other)

    def __eq__(self, other):
        return (isinstance(other, abc.Sequence) and
                len(self) == len(other) and
                all(map(eq, self, other)))

    # index method is overloaded by __init__

    def count(self, v):
        # everything is unique
        return int(v in self)

    def relabel(self, mapping):
        try:
            old_labels = set(mapping)
            new_labels = set(mapping.values())
        except TypeError:
            raise ValueError("mapping targets must be hashable objects")

        for v in new_labels:
            if v in self and v not in old_labels:
                raise ValueError(('A variable cannot be relabeled "{}" without also relabeling '
                                  "the existing variable of the same name").format(v))

        if any(v in new_labels for v in old_labels):
            old_to_intermediate, intermediate_to_new = resolve_label_conflict(mapping, old_labels, new_labels)

            self.relabel(old_to_intermediate)
            self.relabel(intermediate_to_new)

            return

        label = self._label
        index = self.index

        for old, new in mapping.items():
            label[index[old]] = new

            index[new] = index[old]
            del index[old]


class VariableArrayView(abc.Mapping):
    """Create a mapping out of :class:`dimod.views.Variables' and :obj:`numpy.ndarray`."""
    __slots__ = '_variables', '_data'

    def __init__(self, variables, data):

        if not isinstance(variables, Variables):
            raise TypeError("variables should be a Variables object")
        if not isinstance(data, np.ndarray):
            raise TypeError("data should be a numpy 1 dimensional array")
        if data.ndim != 1:
            raise ValueError("data should be a numpy 1 dimensional array")
        if len(variables) != len(data):
            raise ValueError("variables and data should match length")

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
        return ArrayValuesView(self)

    def items(self):
        return ArrayItemsView(self)


class ArrayValuesView(abc.ValuesView):
    """Faster read access to the numpy array"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return iter(self._mapping._data.flat)


class ArrayItemsView(abc.ItemsView):
    """Faster read access to the numpy array"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return zip(self._mapping._variables, self._mapping._data.flat)


class SampleView(VariableArrayView):
    """View each row of the samples record as if it was a dict."""
    __slots__ = ()

    def __repr__(self):
        return str(self)


class NeighbourView(abc.Mapping):
    __slots__ = '_variables', '_ineighbours', '_qdata'

    def __init__(self, variables, ineighbours, qdata):
        if not isinstance(variables, Variables):
            raise TypeError
        if not isinstance(ineighbours, dict):
            raise TypeError
        if not isinstance(qdata, np.ndarray):
            raise TypeError
        self._variables = variables
        self._ineighbours = ineighbours
        self._qdata = qdata

    def __getitem__(self, v):
        return self._qdata[self._ineighbours[self._variables.index(v)]]

    def __setitem__(self, v, bias):
        self._qdata[self._ineighbours[self._variables.index(v)]] = bias

    def __iter__(self):
        variables = self._variables
        for idx in self._ineighbours:
            yield variables[idx]

    def __len__(self):
        return len(self._ineighbours)

    def __repr__(self):
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self._variables, self._ineighbours)

    def __str__(self):
        return str(dict(self))


class AdjacencyView(abc.Mapping):
    __slots__ = '_variables', '_iadj', '_qdata'

    def __init__(self, variables, iadj, qdata):
        if not isinstance(variables, Variables):
            raise TypeError
        if not isinstance(iadj, dict):
            raise TypeError
        if not isinstance(qdata, np.ndarray):
            raise TypeError
        self._variables = variables
        self._iadj = iadj
        self._qdata = qdata

    def __getitem__(self, v):
        return NeighbourView(self._variables, self._iadj[self._variables.index(v)], self._qdata)

    def __iter__(self):
        return iter(self._variables)

    def __len__(self):
        return len(self._variables)

    def __repr__(self):
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self._variables, self._iadj)

    def __str__(self):
        return str({key: dict(val) for key, val in self.items()})


class LinearView(VariableArrayView):
    __slots__ = ()

    def __setitem__(self, v, bias):
        self._data[self._variables.index(v)] = bias


class QuadraticView(abc.Mapping):
    __slots__ = '_bqm',

    def __init__(self, bqm):
        self._bqm = bqm

    def __getitem__(self, interaction):
        u, v = interaction
        return self._bqm.adj[u][v]

    def __setitem__(self, interaction, bias):
        u, v = interaction
        self._bqm.adj[u][v] = bias

    def __iter__(self):
        bqm = self._bqm
        variables = bqm.variables
        for r, c in zip(bqm.irow, bqm.icol):
            yield variables[r], variables[c]

    def __len__(self):
        return len(self._bqm.qdata)

    def __str__(self):
        return str(dict(self))
