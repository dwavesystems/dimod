# Copyright 2021 D-Wave Systems Inc.
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

import copy
import io
import warnings

from collections.abc import Iterable, Iterator, Mapping, Sequence, MutableMapping, Container
from functools import cached_property
from numbers import Integral, Number
from typing import Hashable, Union, Tuple, Optional, Any

import numpy as np

try:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    ArrayLike = Any
    DTypeLike = Any

from dimod.binary.cybqm import cyBQM_float32, cyBQM_float64
from dimod.binary.pybqm import pyBQM
from dimod.decorators import forwarding_method
from dimod.typing import Bias, Variable
from dimod.variables import Variables
from dimod.vartypes import as_vartype, Vartype

__all__ = ['BinaryQuadraticModel',
           'BQM',
           'DictBQM',
           'Float32BQM',
           'Float64BQM',
           ]


class BQMView:
    __slots__ = ['_bqm']

    def __init__(self, bqm):
        self._bqm = bqm

    def __repr__(self):
        # let's just print the whole (potentially massive) thing for now, in
        # the future we'd like to do something a bit more clever (like hook
        # into dimod's Formatter)
        stream = io.StringIO()
        stream.write('{')
        last = len(self) - 1
        for i, (key, value) in enumerate(self.items()):
            stream.write(f'{key!r}: {value!r}')
            if i != last:
                stream.write(', ')
        stream.write('}')
        return stream.getvalue()


class Neighborhood(Mapping, BQMView):
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
        for v, _ in self._bqm.iter_neighborhood(self._var):
            yield v

    def __len__(self):
        return self._bqm.degree(self._var)

    def __setitem__(self, v, bias):
        self._bqm.set_quadratic(self._var, v, bias)


class Adjacency(Mapping, BQMView):
    """Quadratic biases as a nested dict of dicts.

    Accessed like a dict of dicts, where the keys of the outer dict are all
    of the model's variables (e.g. `v`) and the values are the neighborhood of
    `v`. Each neighborhood is a dict where the keys are the neighbors of `v`
    and the values are their associated quadratic biases.
    """
    def __getitem__(self, v):
        return Neighborhood(self._bqm, v)

    def __iter__(self):
        return iter(self._bqm.variables)

    def __len__(self):
        return self._bqm.num_variables


class Linear(MutableMapping, BQMView):
    """Linear biases as a mapping.

    Accessed like a dict, where keys are the variables of the binary quadratic
    model and values are the linear biases.
    """
    def __delitem__(self, v):
        try:
            self._bqm.remove_variable(v)
        except ValueError:
            raise KeyError(repr(v))

    def __getitem__(self, v):
        try:
            return self._bqm.get_linear(v)
        except ValueError as e:
            raise KeyError(*e.args)

    def __iter__(self):
        yield from self._bqm.variables

    def __len__(self):
        return self._bqm.num_variables

    def __setitem__(self, v, bias):
        self._bqm.set_linear(v, bias)


class Quadratic(MutableMapping, BQMView):
    """Quadratic biases as a flat mapping.

    Accessed like a dict, where keys are 2-tuples of varables, which represent
    an interaction and values are the quadratic biases.
    """
    def __delitem__(self, uv):
        try:
            self._bqm.remove_interaction(*uv)
        except ValueError:
            raise KeyError(repr(uv))

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented

        try:
            return (len(self) == len(other) and
                    all(self[key] == value for key, value in other.items()))
        except KeyError:
            return False

    def __getitem__(self, uv):
        try:
            return self._bqm.get_quadratic(*uv)
        except ValueError as e:
            raise KeyError(*e.args)

    def __iter__(self):
        for u, v, _ in self._bqm.iter_quadratic():
            yield u, v

    def __len__(self):
        return self._bqm.num_interactions

    def __setitem__(self, uv, bias):
        self._bqm.set_quadratic(*uv, bias)


class BinaryQuadraticModel:
    """TODO"""

    _DATA_CLASSES = {
        np.dtype(np.float32): cyBQM_float32,
        np.dtype(np.float64): cyBQM_float64,
        np.dtype(np.object_): pyBQM,
    }

    def __init__(self, *args, vartype=None, dtype=np.float64):

        if vartype is not None:
            args = [*args, vartype]

        # developer note: I regret ever setting up this construction system
        # but it's gotten out there so we're stuck with it
        # I would like to reestablish kwarg construction at some point
        if len(args) == 0:
            raise TypeError("A valid vartype or another bqm must be provided")
        if len(args) == 1:
            # BQM(bqm) or BQM(vartype)
            if isinstance(args[0], BinaryQuadraticModel):
                raise NotImplementedError
                self._init_bqm(args[0], dtype=dtype)
            else:
                self._init_empty(vartype=args[0], dtype=dtype)
        elif len(args) == 2:
            # BQM(bqm, vartype), BQM(n, vartype) or BQM(M, vartype)
            if isinstance(args[0], BinaryQuadraticModel):
                raise NotImplementedError
                self._init_bqm(args[0], args[1], dtype=dtype)
            elif isinstance(args[0], Integral):
                self._init_empty(vartype=args[1], dtype=dtype)
                self.resize(args[0])
            else:
                self._init_components([], args[0], 0.0, args[1], dtype=dtype)
        elif len(args) == 3:
            # BQM(linear, quadratic, vartype)
            self._init_components(args[0], args[1], 0.0, args[2], dtype=dtype)
        elif len(args) == 4:
            # BQM(linear, quadratic, offset, vartype)
            self._init_components(*args, dtype=dtype)
        else:
            msg = "__init__() takes 4 positional arguments but {} were given."
            raise TypeError(msg.format(len(args)))

    def _init_components(self, linear, quadratic, offset, vartype, dtype):
        self.data = type(self)._DATA_CLASSES[np.dtype(dtype)](vartype)

        vartype = self.data.vartype

        if isinstance(quadratic, (Mapping, Iterator)):
            if isinstance(quadratic, Mapping):
                quadratic = ((u, v, bias) for (u, v), bias in quadratic.items())

            for u, v, bias in quadratic:
                if u == v:
                    if vartype is Vartype.BINARY:
                        self.add_linear(u, bias)
                    elif vartype is Vartype.SPIN:
                        self.offset += bias
                    else:
                        raise RuntimeError(f"unexpected vartype: {vartype}")
                else:
                    self.add_quadratic(u, v, bias)

        else:
            if self.dtype == np.dtype('O') and not hasattr(quadratic, 'dtype'):
                dt = np.dtype('O')
            else:
                dt = None

            quadratic = np.asarray(quadratic, order='C', dtype=dt)
            diag = np.diagonal(quadratic)
            if diag.any():
                if vartype is Vartype.SPIN:
                    self.offset += diag.sum()
                elif vartype is Vartype.BINARY:
                    self.add_linear_from_array(diag)
                else:
                    raise RuntimeError(f"unexpected vartype: {vartype}")

                # zero out the diagonal
                new_quadratic = np.array(quadratic, copy=True)
                np.fill_diagonal(new_quadratic, 0)

                self.add_quadratic_from_dense(new_quadratic)
            else:
                self.add_quadratic_from_dense(quadratic)

        if isinstance(linear, (Iterator, Mapping)):
            self.add_linear_from(linear)
        else:
            self.add_linear_from_array(linear)

        self.offset += offset

    def _init_empty(self, vartype, dtype):
        self.data = type(self)._DATA_CLASSES[np.dtype(dtype)](vartype)

    def __eq__(self, other):
        # todo: performance
        try:
            return (self.vartype == other.vartype
                    and self.shape == other.shape  # redundant, fast to check
                    and self.offset == other.offset
                    and self.linear == other.linear
                    and self.adj == other.adj)
        except AttributeError:
            return False

    def __len__(self):
        return self.num_variables

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return "{!s}({!s}, {!s}, {!r}, {!r})".format(type(self).__name__,
                                                     self.linear,
                                                     self.quadratic,
                                                     self.offset,
                                                     self.vartype.name)

    @cached_property
    def adj(self) -> Adjacency:
        return Adjacency(self)

    @property
    def dtype(self) -> np.dtype:
        """Data-type of the model's biases."""
        return self.data.dtype

    @cached_property
    def linear(self) -> Linear:
        return Linear(self)

    @property
    def num_variables(self) -> int:
        return self.data.num_variables()

    @property
    def num_interactions(self) -> int:
        return self.data.num_interactions()

    @cached_property
    def quadratic(self) -> Quadratic:
        return Quadratic(self)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.num_variables, self.num_interactions

    @cached_property
    def variables(self) -> Variables:
        """Alias for :meth:`add_quadratic_from`."""
        return self.data.variables

    @property
    def offset(self) -> np.number:
        """Constant energy offset associated with the model."""
        return self.data.offset

    @offset.setter
    def offset(self, offset):
        self.data.offset = offset

    @property
    def vartype(self) -> Vartype:
        """The model's variable type.

        One of :class:`.Vartype.SPIN` or :class:`.Vartype.BINARY`.
        """
        return self.data.vartype

    @classmethod
    def shapeable(cls) -> bool:
        """Returns True if the binary quadratic model is shapeable.

        This method is deprecated. All BQMs are shapeable.
        """
        name = cls.__name__
        warnings.warn(
            f"{name}.shapeable() is deprecated. All BQMs are shapeable.",
            DeprecationWarning,
            stacklevel=2)
        return True

    @forwarding_method
    def add_linear(self, v: Variable, bias: Bias):
        return self.data.add_linear

    def add_linear_from(self, linear: Union[Iterable, Mapping]):
        """Add variables and linear biases to a binary quadratic model.

        Args:
            linear:
                A collection of variables in their associated linear biases.
                If a dict, should be of the form `{v: bias, ...}` where `v` is
                a variable and `bias` is its associated linear bias.
                Otherwise, should be an iterable of `(v, bias)` pairs.

        """
        if isinstance(linear, Mapping):
            iterator = linear.items()
        elif isinstance(linear, Iterator):
            iterator = linear
        else:
            raise TypeError(
                "expected 'linear' to be a dict or an iterable of 2-tuples.")

        for v, bias in iterator:
            self.add_linear(v, bias)

    def add_linear_from_array(self, linear: Sequence):
        ldata = np.asarray(linear)

        # cython has trouble with readonly buffers as of 0.29.22, in the
        # future we can remove this
        if not ldata.flags.writeable:
            ldata = np.array(ldata, copy=True)

        self.data.add_linear_from_array(np.asarray(ldata))

    add_variables_from = add_linear_from
    """Alias for :meth:`add_linear_from`."""

    def add_offset(self, bias):
        self.offset += bias

    @forwarding_method
    def add_quadratic(self, u: Variable, v: Variable, bias: Bias):
        return self.data.add_quadratic

    def add_interaction(self, *args, **kwargs):
        """Alias for :meth:`.add_quadratic`."""
        return self.add_quadratic(*args, **kwargs)

    def add_quadratic_from(self, quadratic: Union[Mapping, Iterable]):
        """Add quadratic biases to the binary quadratic model.

        Args:
            quadratic:
                Collection of interactions and their associated quadratic
                bias. If a dict, should be of the form `{(u, v): bias, ...}`
                where `u` and `v` are variables in the model and `bias` is
                the associated quadratic bias. Otherwise, should be an
                iterable of `(u, v, bias)` triplets.
                If a variable is not present in the model, it is added.
                If the interaction already exists, the bias is added.

        Raises:
            ValueError:
                If any self-loops are given. E.g. `(u, u, bias)` is not a valid
                triplet.

        """
        add_quadratic = self.data.add_quadratic

        if isinstance(quadratic, Mapping):
            for (u, v), bias in quadratic.items():
                add_quadratic(u, v, bias)
        else:
            for u, v, bias in quadratic:
                add_quadratic(u, v, bias)

    add_interactions_from = add_quadratic_from
    """Alias for :meth:`add_quadratic_from`."""

    def add_quadratic_from_dense(self, quadratic: ArrayLike):
        """Add quadratic biases from a square 2d array-like.

        Args:
            quadratic:
                An square 2d `array-like`_ of quadratic biases.

        .. _`array_like`:  https://numpy.org/doc/stable/user/basics.creation.html

        """
        quadratic = np.asarray(quadratic, order='C')

        # cython has trouble with readonly buffers as of 0.29.22, in the
        # future we can remove this
        if not quadratic.flags.writeable:
            quadratic = np.array(quadratic, copy=True)

        self.data.add_quadratic_from_dense(quadratic)

    @forwarding_method
    def add_variable(self, v: Optional[Variable] = None, bias: Bias = 0):
        return self.data.add_variable

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
        if u not in self.variables:
            raise ValueError(f"unknown variable: {u}")
        if v not in self.variables:
            raise ValueError(f"unknown variable: {v}")

        self.add_linear(u, self.get_linear(v))

        if self.vartype is Vartype.BINARY:
            self.add_linear(u, self.get_quadratic(u, v, default=0))
        elif self.vartype is Vartype.SPIN:
            self.offset += self.get_quadratic(u, v, default=0)

        else:
            raise RuntimeError(f"unknown vartype: {self.vartype}")

        self.remove_interaction(u, v)

        # add all of v's interactions to u's
        for w, b in self.iter_neighborhood(v):
            self.add_quadratic(u, w, b)

        # finally remove v
        self.remove_variable(v)

    def copy(self, deep=False):
        """Return a copy."""
        return copy.deepcopy(self)

    def change_vartype(self, vartype, inplace=True):
        """Return a binary quadratic model with the specified vartype.

        Args:
            vartype (:class:`.Vartype`/str/set, optional):
                Variable type for the changed model. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            inplace (bool, optional, default=True):
                If True, the binary quadratic model is updated in-place;
                otherwise, a new binary quadratic model is returned.

        Returns:
            :obj:`.BQM`: A binary quadratic model with the specified
            vartype.

        """
        if not inplace:
            return self.copy().change_vartype(vartype, inplace=True)
        self.data.change_vartype(vartype)
        return self

    @forwarding_method
    def degree(self, v: Variable):
        return self.data.degree

    @classmethod
    def empty(cls, vartype):
        """Create a new binary quadratic model with no variables and no offset.
        """
        return cls(vartype)

    def energies(self, samples_like, dtype: Optional[DTypeLike] = None):
        return self.data.energies(samples_like, dtype=dtype)

    def energy(self, sample, dtype=None):
        """Determine the energy of the given sample.

        Args:
            samples_like (samples_like):
                Raw sample. `samples_like` is an extension of
                NumPy's array_like structure. See :func:`.as_samples`.

            dtype (data-type, optional, default=None):
                Desired NumPy data type for the energy. Matches
                :attr:`.dtype` by default.

        Returns:
            The energy.

        """
        energy, = self.energies(sample, dtype=dtype)
        return energy

    def fix_variable(self, v: Hashable, value: int):
        """Remove a variable by fixing its value.

        Args:
            v: Variable in the binary quadratic model to be fixed.

            value: Value assigned to the variable. Values must match the
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

    def fix_variables(self, fixed: Union[Mapping, Iterable]):
        """Fix the value of the variables and remove them.

        Args:
            fixed: A dictionary or an iterable of 2-tuples of variable
                assignments.

        """
        if isinstance(fixed, Mapping):
            fixed = fixed.items()
        for v, val in fixed:
            self.fix_variable(v, val)

    # def flip_variable(self, v: Hashable):
    #     """Flip variable `v` in a binary quadratic model."""
    #     for u in self.adj[v]:
    #         self.spin.adj[v][u] *= -1
    #     self.spin.linear[v] *= -1

    @classmethod
    def from_coo(cls, obj, vartype=None):
        """Deserialize a BQM from a Coordinate format encoding.

        Args:
            obj: (str/file):
                Either a string or a `.read()`-supporting file object that
                represents linear and quadratic biases for a binary quadratic
                model.

        Note:
            This method is deprecated, use
            :func:`~dimod.serialization.coo.load` or
            :func:`~dimod.serialization.coo.loads` instead.

        """
        warnings.warn('BinaryQuadraticModel.from_coo() is deprecated since '
                      'dimod 0.10.0, '
                      'use dimod.serialization.coo.load(bqm) or '
                      'dimod.serialization.coo.loads(bqm, fp) instead.',
                      DeprecationWarning, stacklevel=2)

        import dimod.serialization.coo as coo

        if isinstance(obj, str):
            return coo.loads(obj, cls=cls, vartype=vartype)

        return coo.load(obj, cls=cls, vartype=vartype)

    @classmethod
    def from_ising(cls, h: Union[Mapping, Sequence],
                   J: Mapping,
                   offset: Number = 0):
        """Create a binary quadratic model from an Ising problem.

        Args:
            h: Linear biases of the Ising problem. If a dict, should be of the
                form `{v: bias, ...}` where v is a spin-valued variable and
                `bias` is its associated bias. If a list, it is treated as a
                list of biases where the indices are the variable labels.

            J: Quadratic biases of the Ising problem.

            offset (optional, default=0.0): Constant offset.

        Returns:
            A spin-valued binary quadratic model.

        """
        return cls(h, J, offset, Vartype.SPIN)

    @classmethod
    def from_numpy_matrix(cls, mat, variable_order=None, offset=0,
                          interactions=None):
        warnings.warn('BQM.from_numpy_matrix(M) is deprecated since dimod '
                      '0.10.0, use BQM(M, "BINARY") instead.',
                      DeprecationWarning, stacklevel=2)
        bqm = cls(mat, Vartype.BINARY)
        bqm.offset = offset

        if variable_order is not None:
            bqm.relabel_variables(dict(enumerate(variable_order)))

        if interactions is not None:
            for u, v in interactions:
                bqm.add_interaction(u, v, 0.0)

        return bqm

    @classmethod
    def from_numpy_vectors(cls, linear, quadratic, offset, vartype, *,
                           variable_order=None, dtype=np.float64):
        obj = super().__new__(cls)
        data_cls = cls._DATA_CLASSES[np.dtype(dtype)]
        obj.data = data_cls.from_numpy_vectors(
            linear, quadratic, offset, vartype,
            variable_order=variable_order)
        return obj

    @classmethod
    def from_qubo(cls, Q: Mapping, offset: Number = 0):
        """Create a binary quadratic model from a QUBO problem.

        Args:
            Q: Coefficients of a quadratic unconstrained binary optimization
                (QUBO) problem. Should be a dict of the form
                `{(u, v): bias, ...}` where `u`, `v`, are binary-valued
                variables and `bias` is their associated coefficient.

            offset (optional, default=0.0):
                Constant offset applied to the model.

        Returns:
            A binary-valued binary quadratic model.

        """
        return cls({}, Q, offset, Vartype.BINARY)

    def has_variable(self, v):
        warnings.warn('bqm.has_variable(v) is deprecated since dimod 0.10.0, '
                      'use v in bqm.variables instead.', 
                      DeprecationWarning, stacklevel=2)
        return v in self.data.variables

    @forwarding_method
    def iter_neighborhood(self, v: Variable):
        return self.data.iter_neighborhood

    def iter_neighbors(self, u):
        warnings.warn('bqm.iter_neighbors(v) is deprecated since dimod 0.10.0,'
                      ' use (v for v, _ in bqm.iter_neighborhood(v)) instead.',
                      DeprecationWarning, stacklevel=2)
        for v, _ in self.iter_neighborhood(u):
            yield v

    def iter_quadratic(self, variables=None):
        if variables is None:
            yield from self.data.iter_quadratic()
        else:
            warnings.warn('passing variables to bqm.iter_quadratic() is '
                          'deprecated since dimod 0.10.0, use '
                          'bqm.iter_neighborhood(v) instead.',
                          DeprecationWarning, stacklevel=2)

            # single variable case
            if isinstance(variables, Hashable) in self.variables:
                u = variables
                for v, bias in self.iter_neighborhood(u):
                    yield u, v, bias
            else:
                seen = set()
                for u in variables:
                    seen.add(u)

                    for v, bias in self.iter_neighborhood(u):
                        if v in seen:
                            continue

                        yield u, v, bias

    def iter_variables(self):
        """Iterate over the variables of the binary quadratic model.

        Yields:
            hashable: A variable in the binary quadratic model.

        Note:
            This method is deprecated, use `iter(bqm.variables)` instead.

        """
        warnings.warn('bqm.iter_variables() is deprecated since dimod 0.10.0, '
                      'use iter(bqm.variables) instead.', stacklevel=2)
        return iter(self.variables)

    @forwarding_method
    def get_linear(self, v: Variable):
        return self.data.get_linear

    @forwarding_method
    def get_quadratic(self, u: Variable, v: Variable,
                      default: Optional[Bias] = None):
        return self.data.get_quadratic

    def normalize(self, bias_range=1, quadratic_range=None,
                  ignored_variables=None, ignored_interactions=None,
                  ignore_offset=False):
        """Normalizes the biases of the binary quadratic model to fall in the
        provided range(s), and adjusts the offset appropriately.

        If ``quadratic_range`` is provided, ``bias_range`` is used for the linear
        biases and ``quadratic_range`` for the quadratic biases.

        Args:
            bias_range (number/pair):
                Value/range that the biases of the BQM is scaled to fit
                within. If ``quadratic_range`` is provided, this range is
                used to fit the linear biases.

            quadratic_range (number/pair):
                The BQM is scaled so that the quadratic biases fit within
                this range.

            ignored_variables (iterable, optional):
                Biases associated with these variables are not scaled.

            ignored_interactions (iterable[tuple], optional):
                Biases associated with these interactions, formatted as an iterable of 2-tuples, are not scaled.

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

            return 1.0 / inv_scalar
        else:
            return 1.0

    def relabel_variables(self, mapping, inplace=True):
        if not inplace:
            return self.copy().relabel_variables(mapping, inplace=True)

        self.data.relabel_variables(mapping)
        return self

    def relabel_variables_as_integers(self, inplace=True):
        if not inplace:
            return self.copy().relabel_variables_as_integers(inplace=True)

        mapping = self.data.relabel_variables_as_integers()
        return self, mapping

    @forwarding_method
    def remove_interaction(self, u: Variable, v: Variable):
        return self.data.remove_interaction

    def remove_interactions_from(self, interactions: Iterable):
        """Remove the given interactions from the binary quadratic model.

        Args:
            interactions: an iterabble of 2-tuples of interactions in
                the binary quadratic model.

        """
        for u, v in interactions:
            self.remove_interaction(u, v)

    @forwarding_method
    def remove_variable(self, v: Optional[Variable] = None) -> Variable:
        return self.data.remove_variable

    @forwarding_method
    def resize(self, n: int):
        return self.data.resize

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
                Biases associated with these interactions, formatted as an
                iterable of 2-tuples, are not scaled.

            ignore_offset (bool, default=False):
                If True, the offset is not scaled.

        """
        # this could be cythonized for performance

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

    @forwarding_method
    def set_linear(self, v: Variable, bias: Bias):
        return self.data.set_linear

    @forwarding_method
    def set_quadratic(self, u: Variable, v: Variable, bias: Bias):
        return self.data.set_quadratic

    def to_coo(self, fp=None, vartype_header: bool = False):
        """Serialize the BQM to a COOrdinate format encoding.

        If `fp` is provided, the serialized bqm will be written to a file,
        otherwise it will be returned as a string.

        Note:
            This method is deprecated, use
            :func:`~dimod.serialization.coo.dump` or
            :func:`~dimod.serialization.coo.dumps` instead.
        """
        warnings.warn('bqm.to_coo() is deprecated since dimod 0.10.0, '
                      'use dimod.serialization.coo.dump(bqm) or '
                      'dimod.serialization.coo.dumps(bqm, fp) instead.',
                      DeprecationWarning, stacklevel=2)

        import dimod.serialization.coo as coo

        if fp is None:
            return coo.dumps(self, vartype_header)
        else:
            coo.dump(self, fp, vartype_header)

    def to_ising(self):
        """Convert a binary quadratic model to Ising format.

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

    def to_numpy_matrix(self, variable_order=None):
        warnings.warn('bqm.to_numpy_matrix() is deprecated since dimod 0.10.0',
                      DeprecationWarning, stacklevel=2)

        if variable_order is None:
            if self.variables ^ range(self.num_variables):
                raise ValueError("variable_order must be provided for BQMs "
                                 "that are not index-labelled")
            variable_order = range(self.num_variables)
        elif self.variables ^ variable_order:
            raise ValueError("variable_order does not match the BQM variables")

        if self.vartype is Vartype.SPIN:
            bqm = self.change_vartype(Vartype.BINARY, inplace=False)
        elif self.vartype is Vartype.BINARY:
            bqm = self
        else:
            raise RuntimeError("unexpected vartype")

        ldata, (irow, icol, qdata), _ = bqm.to_numpy_vectors(variable_order)

        # make sure it's upper triangular
        idx = irow > icol
        if idx.any():
            irow[idx], icol[idx] = icol[idx], irow[idx]

        dense = np.zeros((self.num_variables, self.num_variables),
                         dtype=self.dtype)
        dense[irow, icol] = qdata

        # set the linear
        np.fill_diagonal(dense, ldata)

        return dense

    def to_numpy_vectors(self, variable_order=None, *,
                         dtype=None, index_dtype=None,
                         sort_indices=False, sort_labels=True,
                         return_labels=False):

        if dtype is not None:
            warnings.warn(
                "The 'dtype' keyword argument is deprecated since dimod 0.10.0"
                " and does nothing",
                DeprecationWarning, stacklevel=2)
        if index_dtype is not None:
            warnings.warn(
                "The 'index_dtype' keyword argument is deprecated since dimod "
                "0.10.0 and does nothing",
                DeprecationWarning, stacklevel=2)

        return self.data.to_numpy_vectors(
            variable_order=variable_order,
            sort_indices=sort_indices,
            sort_labels=sort_labels,
            return_labels=return_labels,
            )

    def update(self, other):
        try:
            self.data.update(other.data)
        except AttributeError:
            raise TypeError
        except TypeError:
            pass
        else:
            return

        raise NotImplementedError


BQM = BinaryQuadraticModel


class DictBQM(BQM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=object, **kwargs)


class Float32BQM(BQM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=np.float32, **kwargs)


class Float64BQM(BQM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, dtype=np.float64, **kwargs)
