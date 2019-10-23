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

from six import add_metaclass


# todo: there is a lot of duplication between dimid/views/bqm.py and
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

    @abc.abstractproperty
    def num_interactions(self):
        """int: The number of interactions in the model."""
        pass

    @abc.abstractproperty
    def num_variables(self):
        """int: The number of variables in the model."""
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
