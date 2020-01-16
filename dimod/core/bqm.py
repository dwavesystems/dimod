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
import io

from collections.abc import KeysView, Mapping, MutableMapping
from pprint import PrettyPrinter

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel as LegacyBQM
from dimod.vartypes import as_vartype, Vartype

__all__ = ['BQM', 'ShapeableBQM']


# todo: there is a lot of duplication between dimod/views/bqm.py and
# dimod/core/bqm.py. For now we'll maintain both but this needs to resolved
# at some point


class BQMView(Mapping):
    __slots__ = ['_bqm']

    def __init__(self, bqm):
        self._bqm = bqm

    # support python2 pickle
    def __getstate__(self):
        return {'_bqm': self._bqm}

    # support python2 pickle
    def __setstate__(self, state):
        self._bqm = state['_bqm']

    def __repr__(self):
        # want the repr to make clear that it's not the correct item
        return "<{!s}: {!s}>".format(type(self).__name__, self)

    def __str__(self):
        # let's just print the whole (potentially massive) thing for now, in
        # the future we'd like to do something a bit more clever (like hook into
        # dimod's Formatter)
        stream = io.StringIO()
        stream.write('{')
        last = len(self) - 1
        for i, (key, value) in enumerate(self.items()):
            stream.write('{!s}: {!s}'.format(key, value))
            if i != last:
                stream.write(', ')
        stream.write('}')
        return stream.getvalue()


class Adjacency(BQMView):
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


class Neighbour(BQMView):
    __slots__ = ['_var']

    def __init__(self, bqm, v):
        super().__init__(bqm)
        self._var = v

    def __getitem__(self, v):
        try:
            return self._bqm.get_quadratic(self._var, v)
        except ValueError as e:
            raise KeyError(*e.args)

    def __iter__(self):
        return self._bqm.iter_neighbors(self._var)

    def __len__(self):
        return self._bqm.degree(self._var)

    def __setitem__(self, v, bias):
        self._bqm.set_quadratic(self._var, v, bias)


class ShapeableNeighbour(Neighbour, MutableMapping):
    def __delitem__(self, v):
        self._bqm.remove_interaction(self._var, v)


class Linear(BQMView):
    __slots__ = ['_bqm']

    def __init__(self, bqm):
        self._bqm = bqm

    def __getitem__(self, v):
        try:
            return self._bqm.get_linear(v)
        except ValueError as e:
            raise KeyError(*e.args)

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


class Quadratic(BQMView):
    def __getitem__(self, uv):
        try:
            return self._bqm.get_quadratic(*uv)
        except ValueError as e:
            raise KeyError(*e.args)

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


class BQM(metaclass=abc.ABCMeta):
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

    def __repr__(self):
        return "{!s}({!s}, {!s}, {!r}, {!r})".format(type(self).__name__,
                                                     self.linear,
                                                     self.quadratic,
                                                     self.offset,
                                                     self.vartype.name)

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

    @classmethod
    def from_ising(cls, h, J, offset=0):
        """Create a binary quadratic model from an Ising problem.

        Args:
            h (dict/list):
                Linear biases of the Ising problem. If a dict, should be of the
                form `{v: bias, ...}` where v is a spin-valued variable and `bias`
                is its associated bias. If a list, it is treated as a list of
                biases where the indices are the variable labels.

            J (dict[(variable, variable), bias]):
                Quadratic biases of the Ising problem.

            offset (optional, default=0.0):
                Constant offset applied to the model.

        Returns:
            A spin-valued binary quadratic model.

        """
        return cls(h, J, offset, Vartype.SPIN)

    @classmethod
    def from_qubo(cls, Q, offset=0):
        """Create a binary quadratic model from a QUBO problem.

        Args:
            Q (dict):
                Coefficients of a quadratic unconstrained binary optimization
                (QUBO) problem. Should be a dict of the form `{(u, v): bias, ...}`
                where `u`, `v`, are binary-valued variables and `bias` is their
                associated coefficient.

            offset (optional, default=0.0):
                Constant offset applied to the model.

        Returns:
            A binary-valued binary quadratic model.

        """
        return cls({}, Q, offset, Vartype.BINARY)

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

    def to_numpy_vectors(self, variable_order=None,
                         dtype=np.float, index_dtype=np.intc,
                         sort_indices=False, sort_labels=True,
                         return_labels=False):
        """The BQM as 4 numpy vectors, the offset and a list of variables."""
        num_variables = self.num_variables
        num_interactions = self.num_interactions

        irow = np.empty(num_interactions, dtype=index_dtype)
        icol = np.empty(num_interactions, dtype=index_dtype)
        qdata = np.empty(num_interactions, dtype=dtype)

        if variable_order is None:
            variable_order = list(self.iter_variables())

            if sort_labels:
                try:
                    variable_order.sort()
                except TypeError:
                    # can't sort unlike types in py3
                    pass

        try:
            ldata = np.fromiter((self.linear[v] for v in variable_order),
                                count=num_variables, dtype=dtype)
        except KeyError:
            msg = "provided 'variable_order' does not match binary quadratic model"
            raise ValueError(msg)

        label_to_idx = {v: idx for idx, v in enumerate(variable_order)}

        # we could speed this up a lot with cython
        for idx, ((u, v), bias) in enumerate(self.quadratic.items()):
            irow[idx] = label_to_idx[u]
            icol[idx] = label_to_idx[v]
            qdata[idx] = bias

        if sort_indices:
            # row index should be less than col index, this handles
            # upper-triangular vs lower-triangular
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

        ret = [ldata, (irow, icol, qdata), ldata.dtype.type(self.offset)]

        if return_labels:
            ret.append(variable_order)

        return tuple(ret)


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


# register the various objects with prettyprint
def _pprint_bqm(printer, bqm, stream, indent, *args, **kwargs):
    clsname = type(bqm).__name__
    stream.write(clsname)
    indent += len(clsname)
    bqmtup = (bqm.linear, bqm.quadratic, bqm.offset, bqm.vartype.name)
    printer._pprint_tuple(bqmtup, stream, indent, *args, **kwargs)


try:
    PrettyPrinter._dispatch[BQMView.__repr__] = PrettyPrinter._pprint_dict
    PrettyPrinter._dispatch[BQM.__repr__] = _pprint_bqm
except AttributeError:
    # we're using some internal stuff in PrettyPrinter so let's silently fail
    # for that
    pass


# register the legacy BQM as a subclass of BQM and ShapeableBQM. This actually
# causes some issues because they don't have identical APIs, but in the mean
# time it does things like allow AdjArrayBQM(LegacyBQM(...)) to work. Relatively
# soon we'll replace LegacyBQM with AdjDictBQM
BQM.register(LegacyBQM)
ShapeableBQM.register(LegacyBQM)
