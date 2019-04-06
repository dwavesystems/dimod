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


class BQMView(object):
    __slots__ = '_adj',

    def __init__(self, bqm):
        self._adj = bqm._adj

    # support python2 pickle
    def __getstate__(self):
        return {'_adj': self._adj}

    # support python2 pickle
    def __setstate__(self, state):
        self._adj = state['_adj']


class LinearView(BQMView, abc.MutableMapping):
    """Acts as a dictionary `{v: bias, ...}` for the linear biases.

    The linear biases are stored in a dict-of-dicts format, where 'self loops'
    store the linear biases.
    So `{v: bias}` is stored `._adj = {v: {v: Bias(bias)}}`.
    If v is not in ._adj[v] then the bias is treated as 0.

    """

    def __delitem__(self, v):
        if v not in self:
            raise KeyError
        adj = self._adj
        if len(adj[v]) - (v in adj[v]) > 0:
            raise ValueError("there are interactions associated with {} that must be deleted first".format(v))
        del adj[v]

    def __getitem__(self, v):
        # developer note: we could try to match the type with other biases in
        # the bqm, but I think it is better to just use python int 0 as it
        # is most likely to be compatible with other numeric types.
        return self._adj[v].get(v, 0)

    def __iter__(self):
        return iter(self._adj)

    def __len__(self):
        return len(self._adj)

    def __setitem__(self, v, bias):
        adj = self._adj
        if v in adj:
            adj[v][v] = bias
        else:
            adj[v] = {v: bias}

    def __str__(self):
        return str(dict(self))

    def items(self):
        return LinearItemsView(self)


class LinearItemsView(abc.ItemsView):
    """Faster items iteration for LinearView."""

    __slots__ = ()

    def __iter__(self):
        for v, neighbours in self._mapping._adj.items():
            # see note in LinearView.__getitem__
            yield v, neighbours.get(v, 0)


class QuadraticView(BQMView, abc.MutableMapping):
    """Acts as a dictionary `{(u, v): bias, ...}` for the quadratic biases.

    The quadratic biases are stored in a dict-of-dicts format. So `{(u, v): bias}` is stored as
    `._adj = {u: {v: Bias(bias)}, v: {u: Bias(bias)}}`.

    """

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
            raise KeyError('{} cannot have an interaction with itself'.format(u))
        return self._adj[u][v]

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
        # remove the self-loops
        return sum(len(neighbours) - (v in neighbours)
                   for v, neighbours in self._adj.items()) // 2

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

        adj[u][v] = adj[v][u] = bias

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
            yield (u, v), adj[u][v]


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
            raise KeyError('{} cannot have an interaction with itself'.format(u))
        return self._adj[u][v]

    def __setitem__(self, u, bias):
        v = self._var
        if u == v:
            raise KeyError('{} cannot have an interaction with itself'.format(u))
        adj = self._adj
        if u not in adj:
            raise KeyError('{} is not an interaction'.format((u, v)))
        adj[v][u] = adj[u][v] = bias

    def __iter__(self):
        v = self._var
        for u in self._adj[v]:
            if u != v:
                yield u

    def __len__(self):
        v = self._var
        neighbours = self._adj[v]
        return len(neighbours) - (v in neighbours)  # ignore self

    def __str__(self):
        return str(dict(self))


class AdjacencyView(BQMView, abc.Mapping):
    """Acts as a dict-of-dicts `{u: {v: bias}, v: {u: bias}}` for the quadratic biases.

    The quadratic biases are stored in a dict-of-dicts format. So `{u: {v: bias}, v: {u: bias}}`
    is stored as `._adj = {u: {v: Bias(bias)}, v: {u: Bias(bias)}}`.

    """

    def __getitem__(self, v):
        if v not in self._adj:
            raise KeyError('{} is not a variable'.format(v))
        return NeighbourView(self._adj, v)

    def __iter__(self):
        return iter(self._adj)

    def __len__(self):
        return len(self._adj)
