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

from collections import OrderedDict

from six.moves import zip


class IndexView(abc.Mapping):
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


class IndexItemsView(abc.ItemsView):
    """Faster read access to the numpy array"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return zip(self._mapping._variables, self._mapping._data.flat)


class IndexValuesView(abc.ValuesView):
    """Faster read access to the numpy array"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return iter(self._mapping._data.flat)

class Bias(object):
    """Biases and data associated with a variable or interaction."""

    __slots__ = 'value', '_data'

    def __init__(self, value, data=None):
        self.value = value

        # we only create the dictionary when it's needed. Otherwise we would use a lot
        # of memory on empty dictionaries.
        if data is not None:
            self._data = data

    def __repr__(self):
        if hasattr(self, '_data'):
            return '{}({!r}, data={!r})'.format(self.__class__.__name__, self.value, self.data)
        else:
            return '{}({!r})'.format(self.__class__.__name__, self.value)

    def copy(self):
        if hasattr(self, '_data'):
            # make a shallow copy of data.
            return type(self)(self.value, self.data.copy())
        else:
            return type(self)(self.value)

    @property
    def data(self):
        """dict: data associated with a variable or interaction."""
        try:
            return self._data
        except AttributeError:
            pass

        self._data = data = {}
        return data
    

class LinearView(abc.MutableMapping):
    """Acts as a dictionary `{v: bias, ...}` for the linear biases.

    The linear biases are stored in a dict-of-dicts format, where 'self loops'
    store the linear biases.
    So `{v: bias}` is stored `._adj = {v: {v: Bias(bias)}}`.

    """

    __slots__ = '_adj',

    def __init__(self, bqm):
        self._adj = bqm._adj

    def __delitem__(self, v):
        if v not in self:
            raise KeyError
        if len(self._adj[v]) > 1:
            raise ValueError("there are interactions associated with {} that must be deleted first".format(v))
        del self._adj[v]

    def __getitem__(self, v):
        return self._adj[v][v].value

    def __iter__(self):
        return iter(self._adj)

    def __len__(self):
        return len(self._adj)

    def __setitem__(self, v, bias):
        adj = self._adj
        if v in adj:
            adj[v][v].value = bias
        else:
            adj[v] = {v: Bias(bias)}

    def __str__(self):
        return str(dict(self))

    def items(self):
        return LinearItemsView(self)


class LinearItemsView(abc.ItemsView):
    """Faster items iteration for LinearView."""

    __slots__ = ()

    def __iter__(self):
        for v, neighbours in self._mapping._adj.items():
            yield v, neighbours[v].value


class QuadraticView(abc.MutableMapping):
    """Acts as a dictionary `{(u, v): bias, ...}` for the quadratic biases.

    The quadratic biases are stored in a dict-of-dicts format. So `{(u, v): bias}` is stored as
    `._adj = {u: {v: Bias(bias)}, v: {u: Bias(bias)}}`.

    """

    __slots__ = '_adj',

    def __init__(self, bqm):
        self._adj = bqm._adj

    def __delitem__(self, interaction):
        u, v = interaction
        if u == v:
            raise KeyError('{} is not an interaction'.format(interaction))
        adj = self._adj
        del adj[v][u]
        del adj[u][v]

    def __getitem__(self, interaction):
        u, v = interaction
        if u == v:
            raise KeyError("{} is not a neighbour of itself".format(u))
        return self._adj[u][v].value

    def __iter__(self):
        seen = set()
        adj = self._adj
        for u, neigh in adj.items():

            for v in neigh:
                if u == v:
                    # not adjacent to itself
                    continue

                if v not in seen:
                    yield (u, v)

            seen.add(u)

    def __len__(self):
        # -1 comes from self loops
        return sum(len(neighbours) - 1 for neighbours in self._adj.values()) // 2

    def __setitem__(self, interaction, bias):
        u, v = interaction
        if u == v:
            raise KeyError('{} cannot have an interaction with itself'.format(u))

        adj = self._adj
        
        # we don't know what type we want the biases, so we require that the variables already
        # exist before we can add an interaction between them
        if u not in adj:
            raise KeyError('{} is not already a variable in the binary quadratic model'.format(u))
        if v not in adj:
            raise KeyError('{} is not already a variable in the binary quadratic model'.format(v))

        if v in adj[u]:
            # adj[u][v] is adj[v][u]
            adj[u][v].value = bias
        else:
            adj[u][v] = adj[v][u] = Bias(bias)

    def __str__(self):
        return str(dict(self))

    def items(self):
        return QuadraticItemsView(self)


class QuadraticItemsView(abc.ItemsView):
    """Faster items iteration"""
    __slots__ = ()

    def __iter__(self):
        adj = self._mapping._adj
        for u, v in self._mapping:
            yield (u, v), adj[u][v].value

class NeighbourView(abc.Mapping):
    """Acts as a dictionary `{u: bias, ...}` for the neighbours of a variable `v`.

    See Also:
        :class:`AdjacencyView`

    """

    __slots__ = '_adj', '_var'

    def __init__(self, adj, v):
        self._adj = adj
        self._var = v

    def __getitem__(self, v):
        u = self._var
        if u == v:
            raise KeyError("{} is not a neighbour of itself".format(u))
        return self._adj[u][v].value

    def __setitem__(self, u, bias):
        v = self._var
        if u == v:
            raise KeyError("{} is not a neighbour of itself".format(u))
        self._adj[u][v].value = bias

    def __iter__(self):
        v = self._var
        for u in self._adj[v]:
            if u != v:
                yield u

    def __len__(self):
        return len(self._adj[self._var]) - 1  # ignore self

    def __str__(self):
        return str(dict(self))

class AdjacencyView(abc.Mapping):
    """Acts as a dict-of-dicts `{u: {v: bias}, v: {u: bias}}` for the quadratic biases.

    The quadratic biases are stored in a dict-of-dicts format. So `{u: {v: bias}, v: {u: bias}}`
    is stored as `._adj = {u: {v: Bias(bias)}, v: {u: Bias(bias)}}`.

    """

    __slots__ = '_adj',

    def __init__(self, bqm):
        self._adj = bqm._adj

    def __getitem__(self, v):
        if v not in self._adj:
            raise KeyError('{} is not a variable'.format(v))
        return NeighbourView(self._adj, v)

    def __iter__(self):
        return iter(self._adj)

    def __len__(self):
        return len(self._adj)
