# Copyright 2019 D-Wave Systems Inc.
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
import abc

try:
    from collections.abc import KeysView, Mapping, MutableMapping
except ImportError:
    from collections import KeysView, Mapping, MutableMapping

import numpy as np

from six import add_metaclass

from dimod.vartypes import as_vartype

__all__ = ['BQM', 'ShapeableBQM']


# todo: there is a lot of duplication between dimod/views/bqm.py and
# dimod/core/bqm.py. For now we'll maintain both but this needs to resolved
# at some point


class BQMView:
    __slots__ = ['_bqm']

    def __init__(self, bqm):
        self._bqm = bqm

    # support python2 pickle
    def __getstate__(self):
        return {'_bqm': self._bqm}

    # support python2 pickle
    def __setstate__(self, state):
        self._bqm = state['_bqm']


class Adjacency(BQMView, Mapping):
    def __getitem__(self, v):
        if not self._bqm.has_variable(v):
            raise KeyError('{} is not a variable'.format(v))
        return Neighbour(self._bqm, v)

    def __iter__(self):
        return self._bqm.iter_variables()

    def __len__(self):
        return self._bqm.num_variables


class ShapeableAdjacency(Adjacency):
    def __getitem__(self, v):
        if not self._bqm.has_variable(v):
            raise KeyError('{} is not a variable'.format(v))
        return ShapeableNeighbour(self._bqm, v)


class Neighbour(Mapping):
    __slots__ = ['_bqm', '_var']

    def __init__(self, bqm, v):
        self._bqm = bqm
        self._var = v

    def __getitem__(self, v):
        return self._bqm.get_quadratic(self._var, v)

    def __iter__(self):
        return self._bqm.iter_neighbors(self._var)

    def __len__(self):
        return self._bqm.degree(self._var)

    def __setitem__(self, v, bias):
        self._bqm.set_quadratic(self._var, v, bias)


class ShapeableNeighbour(Neighbour, MutableMapping):
    def __delitem__(self, v):
        self._bqm.remove_interaction(self._var, v)


class Linear(BQMView, Mapping):
    __slots__ = ['_bqm']

    def __init__(self, bqm):
        self._bqm = bqm

    def __getitem__(self, v):
        return self._bqm.get_linear(v)

    def __iter__(self):
        return self._bqm.iter_variables()

    def __len__(self):
        return len(self._bqm)

    def __setitem__(self, v, bias):
        # inherits its ability to reshape the bqm from the `.set_linear` method
        self._bqm.set_linear(v, bias)


class ShapeableLinear(Linear, MutableMapping):
    def __delitem__(self, v):
        try:
            self._bqm.remove_variable(v)
        except ValueError:
            raise KeyError(repr(v))


class Quadratic(BQMView, Mapping):
    def __getitem__(self, uv):
        return self._bqm.get_quadratic(*uv)

    def __iter__(self):
        return self._bqm.iter_interactions()

    def __len__(self):
        return self._bqm.num_interactions

    def __setitem__(self, uv, bias):
        # inherits its ability to reshape the bqm from the `.set_linear` method
        u, v = uv
        self._bqm.set_quadratic(u, v, bias)


class ShapeableQuadratic(Quadratic, MutableMapping):
    def __delitem__(self, uv):
        try:
            self._bqm.remove_interaction(*uv)
        except ValueError:
            raise KeyError(repr(uv))


@add_metaclass(abc.ABCMeta)
class BQM:
    @abc.abstractmethod
    def __init__(self, obj):
        pass

    def __eq__(self, other):
        return (self.vartype == other.vartype
                and self.shape == other.shape  # not necessary but fast to check
                and self.offset == other.offset
                and self.adj == other.adj)

    def __ne__(self, other):
        return not self == other

    @abc.abstractproperty
    def num_interactions(self):
        """int: The number of interactions in the model."""
        pass

    @abc.abstractproperty
    def num_variables(self):
        """int: The number of variables in the model."""
        pass

    @abc.abstractmethod
    def change_vartype(self, vartype, inplace=True):
        """Return a binary quadratic model with the specified vartype."""
        pass

    @abc.abstractmethod
    def copy(self):
        """Return a copy."""
        pass

    @abc.abstractmethod
    def degree(self, v):
        pass

    @abc.abstractmethod
    def get_linear(self, u, v):
        pass

    @abc.abstractmethod
    def get_quadratic(self, u, v):
        pass

    @abc.abstractmethod
    def iter_linear(self):
        pass

    @abc.abstractmethod
    def iter_quadratic(self, variables=None):
        pass

    @abc.abstractmethod
    def set_linear(self, u, v):
        pass

    @abc.abstractmethod
    def set_quadratic(self, u, v, bias):
        pass

    # mixins

    def __len__(self):
        """The number of variables in the binary quadratic model."""
        return self.num_variables

    @property
    def adj(self):
        return Adjacency(self)

    @property
    def linear(self):
        return Linear(self)

    @property
    def quadratic(self):
        return Quadratic(self)

    @property
    def shape(self):
        """2-tuple: (num_variables, num_interactions)."""
        return self.num_variables, self.num_interactions

    @property
    def variables(self):
        return KeysView(self.linear)

    def degrees(self, array=False, dtype=np.int):
        if array:
            return np.fromiter((self.degree(v) for v in self.iter_variables()),
                               count=len(self), dtype=dtype)
        return {v: self.degree(v) for v in self.iter_variables()}

    def has_variable(self, v):
        """Return True if v is a variable in the binary quadratic model."""
        try:
            self.get_linear(v)
        except (ValueError, TypeError):
            return False
        return True

    def iter_variables(self):
        """Iterate over the variables of the binary quadratic model.

        Yields:
            hashable: A variable in the binary quadratic model.

        """
        for v, _ in self.iter_linear():
            yield v

    def iter_interactions(self):
        """Iterate over the interactions of the binary quadratic model.

        Yields:
            interaction: An interaction in the binary quadratic model.

        """
        for u, v, _ in self.iter_quadratic():
            yield u, v

    def iter_neighbors(self, u):
        """Iterate over the neighbors of a variable in the bqm.

        Yields:
            variable: The neighbors of `v`.

        """
        for _, v, _ in self.iter_quadratic(u):
            yield v

    @classmethod
    def shapeable(cls):
        return issubclass(cls, ShapeableBQM)

    def to_coo(self):
        """The BQM as 4 numpy vectors, the offset and a list of variables."""
        nv = self.num_variables
        ni = self.num_interactions

        try:
            dtype = self.dtype
        except AttributeError:
            dtype = np.float
        try:
            itype = self.itype
        except AttributeError:
            itype = np.uint32

        ldata = np.empty(nv, dtype=dtype)
        irow = np.empty(ni, dtype=itype)
        icol = np.empty(ni, dtype=itype)
        qdata = np.empty(ni, dtype=dtype)

        labels = list(self.iter_variables())
        label_to_idx = {v: i for i, v in enumerate(labels)}

        for v, bias in self.linear.items():
            ldata[label_to_idx[v]] = bias
        qi = 0
        for (u, v), bias in self.quadratic.items():
            irow[qi] = label_to_idx[u]
            icol[qi] = label_to_idx[v]
            qdata[qi] = bias
            qi += 1

        # we want to make sure the COO format is sorted
        swaps = irow > icol
        if swaps.any():
            # in-place
            irow[swaps], icol[swaps] = icol[swaps], irow[swaps]

        # sort lexigraphically
        order = np.lexsort((irow, icol))
        if not (order == range(len(order))).all():
            # copy
            irow = irow[order]
            icol = icol[order]
            qdata = qdata[order]

        return ldata, (irow, icol, qdata), self.offset, labels


class ShapeableBQM(BQM):
    @abc.abstractmethod
    def add_variable(self, v=None):
        """Add a variable to the binary quadratic model.

        Args:
            label (hashable, optional):
                A label for the variable. Defaults to the length of the binary
                quadratic model, if that label is available. Otherwise defaults
                to the lowest available positive integer label.

        Returns:
            hashable: The label of the added variable.

        Raises:
            TypeError: If the label is not hashable.

        """
        pass

    @abc.abstractmethod
    def remove_interaction(self, u, v):
        pass

    @abc.abstractmethod
    def remove_variable(self):
        pass

    # mixins

    @property
    def adj(self):
        return ShapeableAdjacency(self)

    @property
    def linear(self):
        return ShapeableLinear(self)

    @property
    def quadratic(self):
        return ShapeableQuadratic(self)
