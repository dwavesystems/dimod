try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import numpy as np

from dimod.bqm.vectors.abc import Vector
from dimod.variables import Variables, MutableVariables
from dimod.views import VariableArrayView


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


class LinearView(abc.MutableMapping):
    __slots__ = '_variables', '_data'

    def __init__(self, variables, data):

        if not isinstance(variables, MutableVariables):
            raise TypeError("variables should be a MutableVariables object")
        if not isinstance(data, Vector):
            raise TypeError("data should be a Vector")
        if len(variables) != len(data):
            raise ValueError("variables and data should match length")

        self._variables = variables
        self._data = data

    def __getitem__(self, v):
        try:
            return self._data[self._variables.index(v)]
        except ValueError:
            raise KeyError('missing element {}'.format(v))

    def __iter__(self):
        return iter(self._variables)

    def __len__(self):
        return len(self._variables)

    def __contains__(self, v):
        return v in self._variables

    def __repr__(self):
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self._variables, self._data)

    def __str__(self):
        return str(dict(self))

    def __setitem__(self, v, bias):
        if v in self:
            self._data[self._variables.index(v)] = bias
        else:
            self._variables.append(v)
            self._data.append(bias)

    def __delitem__(self, v):
        idx = self._variables.index(v)
        del self._variables[idx]
        del self._data[idx]


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
