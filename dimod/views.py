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

class Data(object):
    __slots__ = 'bias', '_data'

    def __init__(self, bias, data=None):
        self.bias = bias

        if data is not None:
            self._data = data

    def __repr__(self):
        if hasattr(self, '_data'):
            return '{}({!r}, data={!r})'.format(self.__class__.__name__, self.bias, self.data)
        else:
            return '{}({!r})'.format(self.__class__.__name__, self.bias)

    def copy(self):
        if hasattr(self, '_data'):
            return type(self)(self.bias, self.data.copy())
        else:
            return type(self)(self.bias)

    @property
    def data(self):
        try:
            return self._data
        except AttributeError:
            pass

        self._data = data = {}
        return data
    

class LinearView(abc.MutableMapping):
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
        return self._adj[v][v].bias

    def __iter__(self):
        return iter(self._adj)

    def __len__(self):
        return len(self._adj)

    def __setitem__(self, v, bias):
        adj = self._adj
        if v in adj:
            adj[v][v].bias = bias
        else:
            adj[v] = {v: Data(bias)}

    def __str__(self):
        return str(dict(self))

    def items(self):
        return LinearItemsView(self)

class OrderedLinearView(LinearView):
    def __setitem__(self, v, bias):
        adj = self._adj
        if v in adj:
            adj[v][v].bias = bias
        else:
            adj[v] = OrderedDict(v=Data(bias))


class LinearItemsView(abc.ItemsView):
    """Faster items iteration"""
    __slots__ = ()

    def __iter__(self):
        for v, neighbours in self._mapping._adj.items():
            yield v, neighbours[v].bias


class QuadraticView(abc.MutableMapping):
    __slots__ = '_adj',

    def __init__(self, bqm):
        self._adj = bqm._adj

    def __delitem__(self, interaction):
        u, v = interaction
        if u == v:
            raise KeyError
        adj = self._adj
        del adj[v][u]
        del adj[u][v]

    def __getitem__(self, interaction):
        u, v = interaction
        if u == v:
            raise KeyError
        return self._adj[u][v].bias

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
        return sum(len(neighbours) - 1 for neighbours in self._adj.values()) // 2

    def __setitem__(self, interaction, bias):
        u, v = interaction
        if u == v:
            raise KeyError

        adj = self._adj
        
        if u not in adj:
            raise NotImplementedError
        if v not in adj:
            raise NotImplementedError

        adj[u][v] = adj[v][u] = Data(bias)

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
            yield (u, v), adj[u][v].bias

class NeighbourView(abc.Mapping):
    __slots__ = '_adj', '_var'

    def __init__(self, adj, v):
        self._adj = adj
        self._var = v

    def __getitem__(self, v):
        u = self._var
        if u == v:
            raise KeyError
        return self._adj[u][v].bias

    def __setitem__(self, u, bias):
        v = self._var
        if u == v:
            raise KeyError
        self._adj[u][v].bias = bias

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
    __slots__ = '_adj',

    def __init__(self, bqm):
        self._adj = bqm._adj

    def __getitem__(self, v):
        if v not in self._adj:
            raise KeyError
        return NeighbourView(self._adj, v)

    def __iter__(self):
        return iter(self._adj)

    def __len__(self):
        return len(self._adj)
