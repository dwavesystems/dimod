try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import numpy as np


class BQMView(abc.Mapping):
    __slots__ = '_bqm',

    def __init__(self, bqm):
        self._bqm = bqm

    def __repr__(self):
        return '{}({!r})'.format(self.__class__.__name__, self._bqm)

    def __str__(self):
        return str(dict(self))


class LinearView(BQMView):
    __slots__ = ()

    def __getitem__(self, v):
        ldata = self._bqm._ldata
        variables = self._bqm._variables
        try:
            return ldata[variables.index(v)]
        except ValueError:
            raise KeyError('missing element {}'.format(v))

    def __setitem__(self, v, bias):
        ldata = self._bqm._ldata
        variables = self._bqm._variables
        try:
            ldata[variables.index(v)] = bias
        except ValueError:
            raise KeyError('missing element {}'.format(v))

    def __iter__(self):
        return iter(self._bqm._variables)

    def __len__(self):
        return len(self._bqm._variables)

    def __contains__(self, v):
        return v in self._bqm._variables

    def items(self):
        return LinearItemsView(self)

    def values(self):
        return LinearValuesView(self)


class LinearItemsView(abc.ItemsView):
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return zip(self._mapping._bqm._variables, self._mapping._bqm.ldata.flat)


class LinearValuesView(abc.ValuesView):
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return iter(self._mapping._bqm.ldata.flat)


class NeighbourView(abc.Mapping):
    __slots__ = '_ineighbours', '_qdata'

    def __init__(self, ineighbours, qdata):
        self._ineighbours = ineighbours
        self._qdata = qdata

    def __getitem__(self, v):
        return self._qdata[self._ineighbours[v]]

    def __setitem__(self, v, bias):
        self._qdata[self._ineighbours[v]] = bias

    def __iter__(self):
        return iter(self._ineighbours)

    def __len__(self):
        return len(self._ineighbours)

    def __repr__(self):
        return '{}({!r}, {!r})'.format(self.__class__.__name__, self._variables, self._ineighbours)

    def __str__(self):
        return str(dict(self))


class AdjacencyView(BQMView):
    __slots__ = ()

    def __len__(self):
        return len(self._bqm._variables)

    def __iter__(self):
        return iter(self._bqm._variables)

    def __contains__(self, v):
        return v in self._bqm._variables

    def __getitem__(self, v):
        return NeighbourView(self._bqm._iadj[v], self._bqm._qdata)


class QuadraticView(BQMView):
    __slots__ = ()

    def __getitem__(self, interaction):
        u, v = interaction
        iadj = self._bqm._iadj
        qdata = self._bqm._qdata
        return qdata[iadj[u][v]]

    def __setitem__(self, interaction, bias):
        u, v = interaction
        iadj = self._bqm._iadj
        qdata = self._bqm._qdata
        qdata[iadj[u][v]] = bias

    def __iter__(self):
        bqm = self._bqm
        variables = bqm._variables
        for ir, ic in zip(bqm._irow, bqm._icol):
            yield variables[ir], variables[ic]

    def __len__(self):
        return len(self._bqm._qdata)

    def items(self):
        return QuadraticItemsView(self)

    def values(self):
        return QuadraticValuesView(self)


class QuadraticItemsView(abc.ValuesView):
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return zip(self._mapping, self._mapping._bqm.qdata.flat)


class QuadraticValuesView(abc.ValuesView):
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return iter(self._mapping._bqm.qdata.flat)
