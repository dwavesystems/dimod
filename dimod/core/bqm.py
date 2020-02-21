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
import functools

from collections.abc import Container, KeysView, Mapping, MutableMapping
from numbers import Number
from pprint import PrettyPrinter

import numpy as np

from dimod.sampleset import as_samples
from dimod.vartypes import as_vartype, Vartype

__all__ = ['BQM', 'ShapeableBQM']


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
        try:
            return (self.vartype == other.vartype
                    and self.shape == other.shape  # not necessary but fast to check
                    and self.offset == other.offset
                    and self.linear == other.linear
                    and self.adj == other.adj)
        except AttributeError:
            return False

    def __ne__(self, other):
        return not (self == other)

    @abc.abstractproperty
    def num_interactions(self):
        """int: The number of interactions in the model."""
        pass

    @abc.abstractproperty
    def num_variables(self):
        """int: The number of variables in the model."""
        pass

    @abc.abstractproperty
    def vartype(self):
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
    def get_linear(self, v):
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
    def relabel_variables(self, mapping, inplace=True):
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
    def base(self):
        """The base bqm, itself if not a view."""
        return self

    @property
    def binary(self):
        if self.vartype is Vartype.BINARY:
            return self

        try:
            return self._binary
        except AttributeError:
            pass

        # this may be kept around even if self.vartype is changed, but that's
        # covered by the above check
        self._binary = binary = BinaryView(self)
        return binary

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
    def spin(self):
        if self.vartype is Vartype.SPIN:
            return self

        try:
            return self._spin
        except AttributeError:
            pass

        # this may be kept around even if self.vartype is changed, but that's
        # covered by the above check
        self._spin = spin = SpinView(self)
        return spin

    @property
    def variables(self):
        return KeysView(self.linear)

    def add_offset(self, offset):
        """Add specified value to the offset of a binary quadratic model."""
        self.offset += offset

    def remove_offset(self):
        """Set the binary quadratic model's offset to zero."""
        # maintain type
        self.offset -= self.offset

    def degrees(self, array=False, dtype=np.int):
        if array:
            return np.fromiter((self.degree(v) for v in self.iter_variables()),
                               count=len(self), dtype=dtype)
        return {v: self.degree(v) for v in self.iter_variables()}

    @classmethod
    def empty(cls, vartype):
        """Create a new empty binary quadratic model."""
        return cls(vartype)

    def energies(self, samples_like, dtype=None):
        """Determine the energies of the given samples.

        Args:
            samples_like (samples_like):
                A collection of raw samples. `samples_like` is an extension of
                NumPy's array_like structure. See :func:`.as_samples`.

            dtype (:class:`numpy.dtype`, optional):
                The data type of the returned energies.

        Returns:
            :obj:`numpy.ndarray`: The energies.

        """
        samples, labels = as_samples(samples_like)

        ldata, (irow, icol, qdata), offset \
            = self.to_numpy_vectors(variable_order=labels, dtype=dtype)

        energies = samples.dot(ldata) + (samples[:, irow]*samples[:, icol]).dot(qdata) + offset
        return np.asarray(energies, dtype=dtype)  # handle any type promotions

    def energy(self, sample, dtype=None):
        energy, = self.energies(sample, dtype=dtype)
        return energy

    def flip_variable(self, v):
        """Flip variable v in a binary quadratic model.

        Args:
            v (variable):
                Variable in the binary quadratic model.

        """
        for u in self.adj[v]:
            self.spin.adj[v][u] *= -1
        self.spin.linear[v] *= -1

    @classmethod
    def from_coo(cls, obj, vartype=None):
        """Deserialize a binary quadratic model from a COOrdinate_ format encoding.

        .. _COOrdinate: https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)

        Args:
            obj: (str/file):
                Either a string or a `.read()`-supporting `file object`_ that represents
                linear and quadratic biases for a binary quadratic model. This data
                is stored as a list of 3-tuples, (i, j, bias), where :math:`i=j`
                for linear biases.

            vartype (:class:`.Vartype`/str/set, optional):
                Variable type for the binary quadratic model. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

                If not provided, the vartype must be specified with a header in the
                file.

        .. _file object: https://docs.python.org/3/glossary.html#term-file-object

        .. note:: Variables must use index lables (numeric lables). Binary quadratic
            models created from COOrdinate format encoding have offsets set to
            zero.

        .. note:: This method will be deprecated in the future. The preferred
            pattern is to use :func:`~dimod.serialization.coo.load` or
            :func:`~dimod.serialization.coo.loads` directly.

        """
        import dimod.serialization.coo as coo

        if isinstance(obj, str):
            return coo.loads(obj, cls=cls, vartype=vartype)

        return coo.load(obj, cls=cls, vartype=vartype)

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
    def from_networkx_graph(cls, G, vartype=None, node_attribute_name='bias',
                            edge_attribute_name='bias'):
        """Create a binary quadratic model from a NetworkX graph.

        Args:
            G (:obj:`networkx.Graph`):
                A NetworkX graph with biases stored as node/edge attributes.

            vartype (:class:`.Vartype`/str/set, optional):
                Variable type for the binary quadratic model. Accepted input
                values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

                If not provided, the `G` should have a vartype attribute. If
                `vartype` is provided and `G.vartype` exists then the argument
                overrides the property.

            node_attribute_name (hashable, optional, default='bias'):
                Attribute name for linear biases. If the node does not have a
                matching attribute then the bias defaults to 0.

            edge_attribute_name (hashable, optional, default='bias'):
                Attribute name for quadratic biases. If the edge does not have a
                matching attribute then the bias defaults to 0.

        Returns:
            Binary quadratic model

        .. note:: This method will be deprecated in the future. The preferred
            pattern is to use the :func:`.from_networkx_graph` function.

        """
        from dimod.converters import from_networkx_graph  # avoid circular import
        return from_networkx_graph(G, vartype, node_attribute_name,
                                   edge_attribute_name, cls=cls)

    @classmethod
    def from_numpy_matrix(cls, mat, variable_order=None, offset=0.0,
                          interactions=None):
        """Create a binary quadratic model from a NumPy array.

        Args:
            mat (:class:`numpy.ndarray`):
                Coefficients of a quadratic unconstrained binary optimization
                (QUBO) model formatted as a square NumPy 2D array.

            variable_order (list, optional):
                If provided, labels the QUBO variables; otherwise, row/column
                indices are used. If `variable_order` is longer than the array,
                extra values are ignored.

            offset (optional, default=0.0):
                Constant offset for the binary quadratic model.

            interactions (iterable, optional, default=[]):
                Any additional 0.0-bias interactions to be added to the binary
                quadratic model. Only works for shapeable binary quadratic
                models.

        Returns:
            Binary quadratic model with vartype set to :class:`.Vartype.BINARY`.

        .. note:: This method will be deprecated in the future. The preferred
            pattern is to use the constructor directly.

        """
        bqm = cls(mat, Vartype.BINARY)
        bqm.offset = offset

        if variable_order is not None:
            bqm.relabel_variables(dict(enumerate(variable_order)))

        if interactions is not None:
            for u, v in interactions:
                bqm.add_interaction(u, v, 0.0)

        return bqm

    @classmethod
    def from_numpy_vectors(cls, linear, quadratic, offset, vartype, variable_order=None):
        """Create a binary quadratic model from vectors.

        Args:
            linear (array_like):
                A 1D array-like iterable of linear biases.

            quadratic (tuple[array_like, array_like, array_like]):
                A 3-tuple of 1D array_like vectors of the form (row, col, bias).

            offset (numeric, optional):
                Constant offset for the binary quadratic model.

            vartype (:class:`.Vartype`/str/set):
                Variable type for the binary quadratic model. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            variable_order (iterable, optional):
                If provided, labels the variables; otherwise, indices are used.

        Returns:
            A binary quadratic model

        """

        try:
            heads, tails, values = quadratic
        except ValueError:
            raise ValueError("quadratic should be a 3-tuple")

        if not len(heads) == len(tails) == len(values):
            raise ValueError("row, col, and bias should be of equal length")

        if variable_order is None:
            variable_order = list(range(len(linear)))

        linear = {v: float(bias) for v, bias in zip(variable_order, linear)}
        quadratic = {(variable_order[u], variable_order[v]): float(bias)
                     for u, v, bias in zip(heads, tails, values)}

        return cls(linear, quadratic, offset, vartype)

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

    def normalize(self, bias_range=1, quadratic_range=None,
                  ignored_variables=None, ignored_interactions=None,
                  ignore_offset=False):
        """Normalizes the biases of the binary quadratic model such that they
        fall in the provided range(s), and adjusts the offset appropriately.

        If `quadratic_range` is provided, then `bias_range` will be treated as
        the range for the linear biases and `quadratic_range` will be used for
        the range of the quadratic biases.

        Args:
            bias_range (number/pair):
                Value/range by which to normalize the all the biases, or if
                `quadratic_range` is provided, just the linear biases.

            quadratic_range (number/pair):
                Value/range by which to normalize the quadratic biases.

            ignored_variables (iterable, optional):
                Biases associated with these variables are not scaled.

            ignored_interactions (iterable[tuple], optional):
                As an iterable of 2-tuples. Biases associated with these
                interactions are not scaled.

            ignore_offset (bool, default=False):
                If True, the offset is not scaled.

        """

        def parse_range(r):
            if isinstance(r, Number):
                return -abs(r), abs(r)
            return r

        def min_and_max(iterable):
            if not iterable:
                return 0, 0
            return min(iterable), max(iterable)

        if ignored_variables is None:
            ignored_variables = set()
        elif not isinstance(ignored_variables, Container):
            ignored_variables = set(ignored_variables)

        if ignored_interactions is None:
            ignored_interactions = set()
        elif not isinstance(ignored_interactions, Container):
            ignored_interactions = set(ignored_interactions)

        if quadratic_range is None:
            linear_range, quadratic_range = bias_range, bias_range
        else:
            linear_range = bias_range

        lin_range, quad_range = map(parse_range, (linear_range,
                                                  quadratic_range))

        lin_min, lin_max = min_and_max([v for k, v in self.linear.items()
                                        if k not in ignored_variables])
        quad_min, quad_max = min_and_max([v for (a, b), v in self.quadratic.items()
                                          if ((a, b) not in ignored_interactions
                                              and (b, a) not in
                                              ignored_interactions)])

        inv_scalar = max(lin_min / lin_range[0], lin_max / lin_range[1],
                         quad_min / quad_range[0], quad_max / quad_range[1])

        if inv_scalar != 0:
            self.scale(1 / inv_scalar, ignored_variables=ignored_variables,
                       ignored_interactions=ignored_interactions,
                       ignore_offset=ignore_offset)

    def scale(self, scalar, ignored_variables=None, ignored_interactions=None,
              ignore_offset=False):
        """Multiply all the biases by the specified scalar.

        Args:
            scalar (number):
                Value by which to scale the energy range of the binary
                quadratic model.

            ignored_variables (iterable, optional):
                Biases associated with these variables are not scaled.

            ignored_interactions (iterable[tuple], optional):
                As an iterable of 2-tuples. Biases associated with these
                interactions are not scaled.

            ignore_offset (bool, default=False):
                If True, the offset is not scaled.

        """

        if ignored_variables is None:
            ignored_variables = set()
        elif not isinstance(ignored_variables, Container):
            ignored_variables = set(ignored_variables)

        if ignored_interactions is None:
            ignored_interactions = set()
        elif not isinstance(ignored_interactions, Container):
            ignored_interactions = set(ignored_interactions)

        linear = self.linear
        for v in linear:
            if v in ignored_variables:
                continue
            linear[v] *= scalar

        quadratic = self.quadratic
        for u, v in quadratic:
            if (u, v) in ignored_interactions or (v, u) in ignored_interactions:
                continue
            quadratic[(u, v)] *= scalar

        if not ignore_offset:
            self.offset *= scalar

    @classmethod
    def shapeable(cls):
        return issubclass(cls, ShapeableBQM)

    def to_coo(self, fp=None, vartype_header=False):
        """Serialize the binary quadratic model to a COOrdinate_ format encoding.

        .. _COOrdinate: https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)

        Args:
            fp (file, optional):
                `.write()`-supporting `file object`_ to save the linear and quadratic biases
                of a binary quadratic model to. The model is stored as a list of 3-tuples,
                (i, j, bias), where :math:`i=j` for linear biases. If not provided,
                returns a string.

            vartype_header (bool, optional, default=False):
                If true, the binary quadratic model's variable type as prepended to the
                string or file as a header.

        .. _file object: https://docs.python.org/3/glossary.html#term-file-object

        .. note:: Variables must use index lables (numeric lables). Binary quadratic
            models saved to COOrdinate format encoding do not preserve offsets.

        .. note:: This method will be deprecated in the future. The preferred
            pattern is to use :func:`~dimod.serialization.coo.dump` or
            :func:`~dimod.serialization.coo.dumps` directly.

        """
        import dimod.serialization.coo as coo

        if fp is None:
            return coo.dumps(self, vartype_header)
        else:
            coo.dump(self, fp, vartype_header)

    def to_ising(self):
        """Converts a binary quadratic model to Ising format.

        If the binary quadratic model's vartype is not :class:`.Vartype.SPIN`,
        values are converted.

        Returns:
            tuple: 3-tuple of form (`linear`, `quadratic`, `offset`), where
            `linear` is a dict of linear biases, `quadratic` is a dict of
            quadratic biases, and `offset` is a number that represents the
            constant offset of the binary quadratic model.

        """
        bqm = self.spin
        return dict(bqm.linear), dict(bqm.quadratic), bqm.offset

    def to_networkx_graph(self, node_attribute_name='bias',
                          edge_attribute_name='bias'):
        """Convert a binary quadratic model to NetworkX graph format.

        Args:
            node_attribute_name (hashable, optional, default='bias'):
                Attribute name for linear biases.

            edge_attribute_name (hashable, optional, default='bias'):
                Attribute name for quadratic biases.

        Returns:
            :class:`networkx.Graph`: A NetworkX graph with biases stored as
            node/edge attributes.

        .. note:: This method will be deprecated in the future. The preferred
            pattern is to use :func:`.to_networkx_graph`.

        """
        from dimod.converters import to_networkx_graph  # avoid circular import
        return to_networkx_graph(self, node_attribute_name, edge_attribute_name)

    def to_numpy_matrix(self, variable_order=None):
        """Convert a binary quadratic model to NumPy 2D array.

        Args:
            variable_order (list, optional):
                If provided, indexes the rows/columns of the NumPy array. If
                `variable_order` includes any variables not in the binary
                quadratic model, these are added to the NumPy array.

        Returns:
            :class:`numpy.ndarray`: The binary quadratic model as a NumPy 2D
            array. Note that the binary quadratic model is converted to
            :class:`~.Vartype.BINARY` vartype.

        .. note:: This method will be deprecated in the future. The preferred
            pattern is to use :meth:`.to_dense`.

        """
        num_variables = self.num_variables
        M = np.zeros((num_variables, num_variables), dtype=self.base.dtype)

        if variable_order is None:
            variable_order = range(num_variables)
        elif len(variable_order) != num_variables:
            raise ValueError("variable_order does not include all variables")

        label_to_idx = {v: i for i, v in enumerate(self.variables)}

        # do the dense thing
        for ui, u in enumerate(variable_order):
            try:
                M[ui, ui] = self.binary.linear[u]
            except KeyError:
                raise ValueError(("if 'variable_order' is not provided, binary "
                                  "quadratic model must be "
                                  "index labeled [0, ..., N-1]"))

            for vi, v in enumerate(variable_order[ui+1:], start=ui+1):
                M[ui, vi] = self.binary.quadratic.get((u, v), 0.0)

        return M

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

    def to_qubo(self):
        """Convert a binary quadratic model to QUBO format.

        If the binary quadratic model's vartype is not :class:`.Vartype.BINARY`,
        values are converted.

        Returns:
            tuple: 2-tuple of form (`biases`, `offset`), where `biases` is a
            dict in which keys are pairs of variables and values are the
            associated linear or quadratic bias and `offset` is a number that
            represents the constant offset of the binary quadratic model.

        """
        qubo = dict(self.binary.quadratic)
        qubo.update(((v, v), bias) for v, bias in self.binary.linear.items())
        return qubo, self.binary.offset


class ShapeableBQM(BQM):
    @abc.abstractmethod
    def add_variable(self, v=None, bias=0):
        """Add a variable to the binary quadratic model.

        Args:
            v (hashable, optional):
                A label for the variable. Defaults to the length of the binary
                quadratic model, if that label is available. Otherwise defaults
                to the lowest available positive integer label.

            bias (numeric, optional, default=0):
                The initial bias value for the added variable. If `v` is already
                a variable, then `bias` (if any) is adding to its existing
                linear bias.

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

    def add_variables_from(self, linear):
        """Add variables and/or linear biases to a binary quadratic model.

        Args:
            linear (dict/iterable):
                A collection of variables in their associated linear biases.
                If a dict, should be of the form `{v: bias, ...}` where `v` is
                a variable and `bias` is its associated linear bias. Otherwise
                should be an iterable of `(v, bias)` pairs.

        """
        if isinstance(linear, Mapping):
            for v, bias in linear.items():
                self.add_variable(v, bias)
        else:
            try:
                for v, bias in linear:
                    self.add_variable(v, bias)
            except TypeError:
                raise TypeError("expected 'linear' to be a dict or an iterable"
                                " of 2-tuples.")

    def add_interaction(self, u, v, bias):
        """Add an interaction and/or quadratic bias to a binary quadratic model.

        Args:
            u (variable):
                One of the pair of variables to add to the model. Can be any
                python object that is a valid dict key.

            v (variable):
                One of the pair of variables to add to the model. Can be any
                python object that is a valid dict key.

            bias (bias):
                Quadratic bias associated with u, v. If u, v is already in the
                model, this value is added to the current quadratic bias.

        """
        self.set_quadratic(u, v, bias + self.get_quadratic(u, v, default=0))

    def add_interactions_from(self, quadratic):
        """Add interactions and/or quadratic biases to a binary quadratic model.

        Args:
            quadratic (dict/iterable):
                A collection of interactions and their associated quadratic
                bias. If a dict, should be of the form `{(u, v): bias, ...}`
                where `u` and `v` are variables in the model and `bias` is
                there associated quadratic bias. Otherwise, whould be an
                iterable of `(u, v, bias)` triplets.

        """
        if isinstance(quadratic, Mapping):
            for (u, v), bias in quadratic.items():
                self.add_interaction(u, v, bias)
        else:
            try:
                for u, v, bias in quadratic:
                    self.add_interaction(u, v, bias)
            except TypeError:
                raise TypeError("expected 'quadratic' to be a dict or an "
                                "iterable of 3-tuples.")

    def contract_variables(self, u, v):
        """Enforce u, v being the same variable in a binary quadratic model.

        The resulting variable is labeled 'u'. Values of interactions between
        `v` and variables that `u` interacts with are added to the
        corresponding interactions of `u`.

        Args:
            u (variable):
                Variable in the binary quadratic model.

            v (variable):
                Variable in the binary quadratic model.

        """
        if not self.has_variable(u):
            msg = "{} is not a variable in the binary quadratic model"
            raise ValueError(msg.format(u))
        if not self.has_variable(v):
            msg = "{} is not a variable in the binary quadratic model"
            raise ValueError(msg.format(v))

        if self.vartype is Vartype.BINARY:
            # the quadratic bias becomes linear
            self.set_linear(u, (self.get_linear(u) + self.get_linear(v)
                                + self.get_quadratic(u, v, default=0)))
        elif self.vartype is Vartype.SPIN:
            # the quadratic bias becomes an offset
            self.set_linear(u, self.get_linear(u) + self.get_linear(v))
            self.offset += self.get_quadratic(u, v)

        self.remove_interaction(u, v)

        # add all of v's interactions to u's
        for _, w, b in self.iter_quadratic(v):
            self.set_quadratic(u, w, self.get_quadratic(u, w, default=0) + b)

        # finally remove v
        self.remove_variable(v)

    def fix_variable(self, v, value):
        """Remove a variable by fixing its value.

        Args:
            v (variable):
                Variable in the binary quadratic model to be fixed.

            value (int):
                Value assigned to the variable. Values must match the
                :class:`.Vartype` of the binary quadratic model.

        """

        if value not in self.vartype.value:
            raise ValueError("expected value to be in {}, received {} "
                             "instead".format(self.vartype.value, value))

        try:
            for u, bias in self.adj[v].items():
                self.linear[u] += bias*value
        except KeyError:
            raise ValueError('{} is not a variable'.format(v))

        self.offset += value*self.linear[v]
        self.remove_variable(v)

    def fix_variables(self, fixed):
        """Fix the value of the variables and remove them.

        Args:
            fixed (dict/iterable):
                A dictionary or an iterable of 2-tuples of variable assignments.

        """
        if isinstance(fixed, Mapping):
            fixed = fixed.items()
        for v, val in fixed:
            self.fix_variable(v, val)

    def remove_variables_from(self, variables):
        """Remove the given variables from the binary quadratic model."""
        for v in variables:
            self.remove_variable(v)

    def remove_interactions_from(self, interactions):
        """Remove the given interactions from the binary quadratic model."""
        for u, v in interactions:
            self.remove_interaction(u, v)

    def update(self, other):
        """Update the binary quadratic model, adding biases from another."""

        if self.vartype is Vartype.SPIN:
            bqm = other.spin
        elif self.vartype is Vartype.BINARY:
            bqm = other.binary
        else:
            raise ValueError("unknown vartype")

        self.add_variables_from(bqm.linear)
        self.add_interactions_from(bqm.quadratic)
        self.add_offset(bqm.offset)


class VartypeView(BQM):

    def __init__(self, bqm):
        self._bqm = bqm

    @property
    def base(self):
        return self._bqm

    @property
    def num_interactions(self):
        return self._bqm.num_interactions

    @property
    def num_variables(self):
        return self._bqm.num_variables

    def change_vartype(self, *args, **kwargs):
        msg = '{} can only be {}-valued'.format(type(self).__name__,
                                                self.vartype.name)
        raise NotImplementedError(msg)

    def degree(self, *args, **kwargs):
        return self._bqm.degree(*args, **kwargs)

    def iter_linear(self):
        for v, _ in self._bqm.iter_linear():
            yield v, self.get_linear(v)

    def iter_quadratic(self, variables=None):
        for u, v, _ in self._bqm.iter_quadratic(variables):
            yield u, v, self.get_quadratic(u, v)

    def relabel_variables(self, *args, **kwargs):
        return self._bqm.relabel_variables(*args, **kwargs)


class BinaryView(VartypeView):
    @property
    def binary(self):
        return self

    @property
    def offset(self):
        bqm = self._bqm
        return (bqm.offset
                - sum(b for _, b in bqm.iter_linear())
                + sum(b for _, _, b in bqm.iter_quadratic()))

    @offset.setter
    def offset(self, bias):
        # just add the difference
        self._bqm.offset += bias - self.offset

    @property
    def spin(self):
        return self._bqm.spin

    @property
    def vartype(self):
        return Vartype.BINARY

    def copy(self):
        return self._bqm.change_vartype(Vartype.BINARY, inplace=False)

    def get_linear(self, v):
        bqm = self._bqm
        return 2 * bqm.get_linear(v) - 2 * sum(b for _, _, b in bqm.iter_quadratic(v))

    def get_quadratic(self, u, v, default=None):
        try:
            return 4 * self._bqm.get_quadratic(u, v)
        except ValueError as err:
            if default is None:
                raise err
        return default

    def set_linear(self, v, bias):
        bqm = self._bqm

        # need the difference
        delta = bias - self.get_linear(v)

        bqm.set_linear(v, bqm.get_linear(v) + delta / 2)
        bqm.offset += delta / 2

    def set_quadratic(self, u, v, bias):
        bqm = self._bqm

        # need the difference
        delta = bias - self.get_quadratic(u, v, default=0)

        # if it doesn't exist and BQM is not shapeable, this will fail
        bqm.set_quadratic(u, v, bias / 4)  # this one is easy

        # the other values get the delta
        bqm.set_linear(u, bqm.get_linear(u) + delta / 4)
        bqm.set_linear(v, bqm.get_linear(v) + delta / 4)

        bqm.offset += delta / 4


class SpinView(VartypeView):
    @property
    def binary(self):
        return self._bqm.binary

    @property
    def offset(self):
        bqm = self._bqm
        return (bqm.offset
                + sum(b for _, b in bqm.iter_linear()) / 2
                + sum(b for _, _, b in bqm.iter_quadratic()) / 4)

    @offset.setter
    def offset(self, bias):
        # just add the difference
        self._bqm.offset += bias - self.offset

    @property
    def spin(self):
        return self

    @property
    def vartype(self):
        return Vartype.SPIN

    def copy(self):
        return self._bqm.change_vartype(Vartype.SPIN, inplace=False)

    def get_linear(self, v):
        bqm = self._bqm
        return (bqm.get_linear(v) / 2
                + sum(b for _, _, b in bqm.iter_quadratic(v)) / 4)

    def get_quadratic(self, u, v, default=None):
        try:
            return self._bqm.get_quadratic(u, v) / 4
        except ValueError as err:
            if default is None:
                raise err
        return default

    def set_linear(self, v, bias):
        bqm = self._bqm

        # need the difference
        delta = bias - self.get_linear(v)

        bqm.set_linear(v, bqm.get_linear(v) + 2 * delta)
        bqm.offset -= delta

    def set_quadratic(self, u, v, bias):
        bqm = self._bqm

        # need the difference
        delta = bias - self.get_quadratic(u, v, default=0)

        # if it doesn't exist and BQM is not shapeable, this will fail
        bqm.set_quadratic(u, v, 4 * bias)  # this one is easy

        # the other values get the delta
        bqm.set_linear(u, bqm.get_linear(u) - 2 * delta)
        bqm.set_linear(v, bqm.get_linear(v) - 2 * delta)

        bqm.offset += delta


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
