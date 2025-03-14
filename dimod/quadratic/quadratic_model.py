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

r"""Quadratic models are problems of the form:

.. math::

    E(x) = \sum_i a_i x_i + \sum_{i \le j} b_{i, j} x_i x_j + c

where :math:`\{ x_i\}_{i=1, \dots, N}` can be binary\ [#]_ or integer
variables and :math:`a_{i}, b_{ij}, c` are real values.

.. [#]
    For binary variables, the range of the quadratic-term summation is
    :math:`i < j` because :math:`x^2 = x` for binary values :math:`\{0, 1\}`
    and :math:`s^2 = 1` for spin values :math:`\{-1, 1\}`.
"""

from __future__ import annotations

import collections.abc
import tempfile
import typing

from copy import deepcopy
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np

try:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    ArrayLike = typing.Any
    DTypeLike = typing.Any

from dimod.decorators import forwarding_method, unique_variable_labels
from dimod.quadratic.cyqm import cyQM_float32, cyQM_float64
from dimod.serialization.fileview import (
    SpooledTemporaryFile,
    _BytesIO,
    LinearSection,
    NeighborhoodSection,
    OffsetSection,
    VariablesSection,
    VartypesSection,
    load,
    read_header,
    write_header,
    )
from dimod.sym import Eq, Ge, Le, Comparison
from dimod.typing import Variable, Bias, VartypeLike
from dimod.variables import Variables
from dimod.vartypes import Vartype, as_vartype
from dimod.views.quadratic import QuadraticViewsMixin

if TYPE_CHECKING:
    # avoid circular imports
    from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel


__all__ = ['QuadraticModel', 'QM', 'Integer', 'Integers', 'IntegerArray', 'Real', 'Reals']


QM_MAGIC_PREFIX = b'DIMODQM'


Vartypes = typing.Union[collections.abc.Mapping[Variable, Vartype],
                        collections.abc.Iterable[tuple[Variable, VartypeLike]]]


class QuadraticModel(QuadraticViewsMixin):
    r"A quadratic model."

    _DATA_CLASSES = {
        np.dtype(np.float32): cyQM_float32,
        np.dtype(np.float64): cyQM_float64,
    }

    DEFAULT_DTYPE = np.float64
    """The default dtype used to construct the class."""

    def __init__(self,
                 linear: typing.Optional[collections.abc.Mapping[Variable, Bias]] = None,
                 quadratic: typing.Optional[collections.abc.Mapping[tuple[Variable, Variable], Bias]] = None,
                 offset: Bias = 0,
                 vartypes: typing.Optional[Vartypes] = None,
                 *,
                 dtype: typing.Optional[DTypeLike] = None):
        dtype = np.dtype(self.DEFAULT_DTYPE) if dtype is None else np.dtype(dtype)
        self.data = self._DATA_CLASSES[np.dtype(dtype)]()

        if vartypes is not None:
            if isinstance(vartypes, collections.abc.Mapping):
                vartypes = vartypes.items()
            for v, vartype in vartypes:
                self.add_variable(vartype, v)
                self.set_linear(v, 0)

        # todo: in the future we can support more types for construction, but
        # let's keep it simple for now
        if linear is not None:
            for v, bias in linear.items():
                self.add_linear(v, bias)
        if quadratic is not None:
            for (u, v), bias in quadratic.items():
                self.add_quadratic(u, v, bias)
        self.offset += offset

    def __deepcopy__(self, memo: dict[int, typing.Any]) -> 'QuadraticModel':
        new = type(self).__new__(type(self))
        new.data = deepcopy(self.data, memo)
        memo[id(self)] = new
        return new

    def __repr__(self):
        vartypes = {v: self.vartype(v).name for v in self.variables}
        return (f"{type(self).__name__}({self.linear}, {self.quadratic}, "
                f"{self.offset}, {vartypes}, dtype={self.dtype.name!r})")

    def __add__(self, other: typing.Union['QuadraticModel', Bias]) -> 'QuadraticModel':
        # in python 3.8+ we could do this is functools.singledispatchmethod
        if isinstance(other, QuadraticModel):
            new = self.copy()
            new.update(other)
            return new
        if isinstance(other, Number):
            new = self.copy()
            new.offset += other
            return new
        return NotImplemented

    def __iadd__(self, other: typing.Union['QuadraticModel', Bias]) -> 'QuadraticModel':
        # in python 3.8+ we could do this is functools.singledispatchmethod
        if isinstance(other, QuadraticModel):
            self.update(other)
            return self
        if isinstance(other, Number):
            self.offset += other
            return self
        return NotImplemented

    def __radd__(self, other: Bias) -> 'QuadraticModel':
        # should only miss on number
        if isinstance(other, Number):
            new = self.copy()
            new.offset += other
            return new
        return NotImplemented

    def __mul__(self, other: typing.Union['QuadraticModel', Bias]) -> 'QuadraticModel':
        if isinstance(other, QuadraticModel):
            if not (self.is_linear() and other.is_linear()):
                raise TypeError(
                    "cannot multiply QMs with interactions")

            # todo: performance

            new = type(self)(dtype=self.dtype)

            for v in self.variables:
                new.add_variable(self.vartype(v), v,
                                 lower_bound=self.lower_bound(v),
                                 upper_bound=self.upper_bound(v))
            for v in other.variables:
                new.add_variable(other.vartype(v), v,
                                 lower_bound=other.lower_bound(v),
                                 upper_bound=other.upper_bound(v))

            self_offset = self.offset
            other_offset = other.offset

            for u, ubias in self.linear.items():
                for v, vbias in other.linear.items():
                    if u == v:
                        u_vartype = self.vartype(u)
                        if u_vartype is Vartype.BINARY:
                            new.add_linear(u, ubias*vbias)
                        elif u_vartype is Vartype.SPIN:
                            new.offset += ubias * vbias
                        elif u_vartype is Vartype.INTEGER or u_vartype is Vartype.REAL:
                            new.add_quadratic(u, v, ubias*vbias)
                        else:
                            raise RuntimeError("unexpected vartype")
                    else:
                        new.add_quadratic(u, v, ubias * vbias)

                new.add_linear(u, ubias * other_offset)

            for v, bias in other.linear.items():
                new.add_linear(v, bias*self_offset)

            new.offset += self_offset*other_offset

            return new
        if isinstance(other, Number):
            new = self.copy()
            new.scale(other)
            return new
        return NotImplemented

    def __imul__(self, other: Bias) -> 'QuadraticModel':
        # in-place multiplication is only defined for numbers
        if isinstance(other, Number):
            self.scale(other)
            return self
        return NotImplemented

    def __rmul__(self, other: Bias) -> 'QuadraticModel':
        # should only miss on number
        if isinstance(other, Number):
            return self * other  # communative
        return NotImplemented

    def __neg__(self: 'QuadraticModel') -> 'QuadraticModel':
        new = self.copy()
        new.scale(-1)
        return new

    def __pow__(self, other: int) -> 'QuadraticModel':
        if isinstance(other, int):
            if other != 2:
                raise ValueError("the only supported power for quadratic models is 2")
            if not self.is_linear():
                raise ValueError("only linear models can be squared")
            return self * self
        return NotImplemented

    def __sub__(self, other: typing.Union['QuadraticModel', Bias]) -> 'QuadraticModel':
        if isinstance(other, QuadraticModel):
            new = self.copy()
            new.scale(-1)
            new.update(other)
            new.scale(-1)
            return new
        if isinstance(other, Number):
            new = self.copy()
            new.offset -= other
            return new
        return NotImplemented

    def __isub__(self, other: typing.Union['QuadraticModel', Bias]) -> 'QuadraticModel':
        if isinstance(other, QuadraticModel):
            self.scale(-1)
            self.update(other)
            self.scale(-1)
            return self
        if isinstance(other, Number):
            self.offset -= other
            return self
        return NotImplemented

    def __rsub__(self, other: Bias) -> 'QuadraticModel':
        # should only miss on a number
        if isinstance(other, Number):
            new = self.copy()
            new.scale(-1)
            new += other
            return new
        return NotImplemented

    def __truediv__(self, other: Bias) -> 'BQM':
        return self * (1 / other)

    def __itruediv__(self, other: Bias) -> 'BQM':
        self *= (1 / other)
        return self

    def __eq__(self, other: Number) -> Comparison:
        if isinstance(other, Number):
            return Eq(self, other)
        return NotImplemented

    def __ge__(self, other: Bias) -> Comparison:
        if isinstance(other, Number):
            return Ge(self, other)
        return NotImplemented

    def __le__(self, other: Bias) -> Comparison:
        if isinstance(other, Number):
            return Le(self, other)
        return NotImplemented

    @property
    def dtype(self) -> np.dtype:
        """Data-type of the model's biases."""
        return self.data.dtype

    @property
    def num_interactions(self) -> int:
        """Number of interactions in the model.

        The complexity is linear in the number of variables.
        """
        return self.data.num_interactions()

    @property
    def num_variables(self) -> int:
        """Number of variables in the model."""
        return self.data.num_variables()

    @property
    def offset(self) -> np.number:
        """Constant energy offset associated with the model."""
        return self.data.offset

    @offset.setter
    def offset(self, offset):
        self.data.offset = offset

    @property
    def shape(self) -> tuple[int, int]:
        """A 2-tuple of :attr:`num_variables` and :attr:`num_interactions`."""
        return self.num_variables, self.num_interactions

    @property
    def variables(self) -> Variables:
        """The variables of the quadratic model.

        Examples:

            >>> qm = dimod.QuadraticModel()
            >>> qm.add_variable('INTEGER', 'i')
            'i'
            >>> qm.add_variable('BINARY')
            1
            >>> qm.add_variable('BINARY', 'y')
            'y'
            >>> qm.variables
            Variables(['i', 1, 'y'])
        """
        return self.data.variables

    @forwarding_method
    def add_linear(self, v: Variable, bias: Bias, *,
                   default_vartype=None,
                   default_lower_bound=None,
                   default_upper_bound=None,
                   ):
        """Add a linear bias to an existing variable or a new variable with
        specified vartype.

        Args:
            v: Variable label.
            bias: Linear bias for the variable.
            default_vartype: The vartype of any variables not already in the
                model. If ``default_vartype`` is ``None`` then missing
                variables raise a ``ValueError``.
            default_lower_bound: The lower bound of any variables not already
                in the model. Ignored if ``default_vartype`` is ``None`` or
                when the variable is :class:`~dimod.Vartype.BINARY` or
                :class:`~dimod.Vartype.SPIN`.
            default_upper_bound: The upper bound of any variables not already
                in the model. Ignored if ``default_vartype`` is ``None`` or
                when the variable is :class:`~dimod.Vartype.BINARY` or
                :class:`~dimod.Vartype.SPIN`.

        Raises:
            ValueError: If the variable is not in the model and
            ``default_vartype`` is ``None``.

        """
        return self.data.add_linear

    def add_linear_from(self,
                        linear: typing.Union[collections.abc.Mapping[Variable, Bias], collections.abc.Iterable[tuple[Variable, Bias]]],
                        *,
                        default_vartype=None,
                        default_lower_bound=None,
                        default_upper_bound=None,
                        ):
        """Add variables and linear biases to a quadratic model.

        Args:
            linear:
                Variables and their associated linear biases, as either a dict of
                form ``{v: bias, ...}`` or an iterable of ``(v, bias)`` pairs,
                where ``v`` is a variable and ``bias`` is its associated linear
                bias.
            default_vartype: The vartype of any variables not already in the
                model. If ``default_vartype`` is ``None`` then missing
                variables raise a ``ValueError``.
            default_lower_bound: The lower bound of any variables not already
                in the model. Ignored if ``default_vartype`` is ``None`` or
                when the variable is :class:`~dimod.Vartype.BINARY` or
                :class:`~dimod.Vartype.SPIN`.
            default_upper_bound: The upper bound of any variables not already
                in the model. Ignored if ``default_vartype`` is ``None`` or
                when the variable is :class:`~dimod.Vartype.BINARY` or
                :class:`~dimod.Vartype.SPIN`.

        Raises:
            ValueError: If the variable is not in the model and
            ``default_vartype`` is ``None``.

        """
        add_linear = self.data.add_linear

        if isinstance(linear, collections.abc.Mapping):
            linear = linear.items()

        # checking whether the keyword arguments are present actually
        # results in a pretty shocking performance difference, almost x2
        # for when they are not there
        # I did try using functools.partial() as well
        if default_vartype is None:
            if default_lower_bound is None and default_upper_bound is None:
                for v, bias in linear:
                    add_linear(v, bias)
            else:
                for v, bias in linear:
                    add_linear(v, bias,
                               default_lower_bound=default_lower_bound,
                               default_upper_bound=default_upper_bound,
                               )
        else:
            default_vartype = as_vartype(default_vartype, extended=True)

            if default_lower_bound is None and default_upper_bound is None:
                for v, bias in linear:
                    add_linear(v, bias,
                               default_vartype=default_vartype,
                               )
            else:
                for v, bias in linear:
                    add_linear(v, bias,
                               default_vartype=default_vartype,
                               default_lower_bound=default_lower_bound,
                               default_upper_bound=default_upper_bound,
                               )

    @forwarding_method
    def add_quadratic(self, u: Variable, v: Variable, bias: Bias):
        """Add quadratic bias to a pair of variables.

        Args:
            u: Variable in the quadratic model.
            v: Variable in the quadratic model.
            bias: Quadratic bias for the interaction.

        Raises:
            ValueError: If a specified variable is not in the model.
            ValueError: If any self-loops are given on binary-valued variables.
                E.g. ``(u, u, bias)`` is not a valid triplet for spin variables.
        """
        return self.data.add_quadratic

    def add_quadratic_from(self, quadratic: typing.Union[collections.abc.Mapping[tuple[Variable, Variable], Bias],
                                                         collections.abc.Iterable[tuple[Variable, Variable, Bias]]]):
        """Add quadratic biases.

        Args:
            quadratic:
                Interactions and their associated quadratic biases, as either a
                dict of form ``{(u, v): bias, ...}`` or an iterable of
                ``(u, v, bias)`` triplets, where ``u`` and ``v`` are variables in
                the model and ``bias`` is the associated quadratic bias.
                If the interaction already exists, the bias is added.

        Raises:
            ValueError: If a specified variable is not in the model.
            ValueError: If any self-loops are given on binary-valued variables.
                E.g. ``(u, u, bias)`` is not a valid triplet for spin variables.
        """
        if isinstance(quadratic, collections.abc.Mapping):
            self.data.add_quadratic_from_iterable(
                (u, v, bias) for (u, v), bias in quadratic.items())
        else:
            self.data.add_quadratic_from_iterable(quadratic)

    @forwarding_method
    def add_variable(self, vartype: VartypeLike, v: typing.Optional[Variable] = None,
                     *,
                     lower_bound: float = 0,
                     upper_bound: typing.Optional[float] = None,
                     ) -> Variable:
        """Add a variable to the quadratic model.

        Args:
            vartype:
                Variable type. One of:

                * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
                * :class:`~dimod.Vartype.INTEGER`, ``'INTEGER'``
                * :class:`~dimod.Vartype.REAL`, ``'REAL'``

            v:
                Label for the variable. Defaults to the length of the
                quadratic model, if that label is available. Otherwise defaults
                to the lowest available positive integer label.

            lower_bound:
                Lower bound on the variable. Ignored when the variable is
                :class:`~dimod.Vartype.BINARY` or :class:`~dimod.Vartype.SPIN`.

            upper_bound:
                Upper bound on the variable. Ignored when the variable is
                :class:`~dimod.Vartype.BINARY` or :class:`~dimod.Vartype.SPIN`.

        Returns:
            The variable label.
        """
        return self.data.add_variable

    def add_variables_from(self, vartype: VartypeLike, variables: collections.abc.Iterable[Variable]):
        """Add multiple variables of the same type to the quadratic model.

        Args:
            vartype: Variable type. One of:

                * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
                * :class:`~dimod.Vartype.INTEGER`, ``'INTEGER'``
                * :class:`~dimod.Vartype.REAL`, ``'REAL'``

            variables: collections.abc.Iterable of variable labels.

        Examples:
            >>> from dimod import QuadraticModel, Binary
            >>> qm = QuadraticModel()
            >>> qm.add_variables_from('BINARY', ['x', 'y'])

        """
        vartype = as_vartype(vartype, extended=True)
        add_variable = self.data.add_variable
        for v in variables:
            add_variable(vartype, v)

    def add_variables_from_model(self,
                                 model: typing.Union[BinaryQuadraticModel,
                                                     ConstrainedQuadraticModel,
                                                     QuadraticModel],
                                 *,
                                 variables: typing.Optional[collections.abc.Iterable[Variable]] = None,
                                 ):
        """Add variables from another model.

        Args:
            model: A binary quadratic model, constrained quadratic model or
                quadratic model.

            variables: The variables from the model to add. If not specified
                all of the variables are added.

        Examples:
            >>> qm0 = dimod.Integer('i', lower_bound=5, upper_bound=10) + dimod.Binary('x')
            >>> qm1 = dimod.QuadraticModel()
            >>> qm1.add_variables_from_model(qm0)
            >>> qm1.variables
            Variables(['i', 'x'])
            >>> qm1.lower_bound('i')
            5.0

        """
        # avoid circular import
        from dimod.binary import BinaryQuadraticModel
        from dimod.constrained import ConstrainedQuadraticModel

        if variables is None:
            variables = model.variables

        vartype = model.vartype if callable(model.vartype) else lambda v: model.vartype

        for v in variables:
            vt = vartype(v)
            if vt is Vartype.SPIN or vt is Vartype.BINARY:
                self.add_variable(vt, v)
            else:
                self.add_variable(vt, v,
                                  lower_bound=model.lower_bound(v),
                                  upper_bound=model.upper_bound(v))

    def change_vartype(self, vartype: VartypeLike, v: Variable) -> "QuadraticModel":
        """Change the variable type of the given variable, updating the biases.

        Args:
            vartype: Variable type. One of:

                * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``
                * :class:`~dimod.Vartype.INTEGER`, ``'INTEGER'``
                * :class:`~dimod.Vartype.REAL`, ``'REAL'``

            v: Variable to change to the specified ``vartype``.

        Example:
            >>> qm = dimod.QuadraticModel()
            >>> a = qm.add_variable('SPIN', 'a')
            >>> qm.set_linear(a, 1.5)
            >>> qm.energy({a: +1})
            1.5
            >>> qm.energy({a: -1})
            -1.5
            >>> qm.change_vartype('BINARY', a)
            QuadraticModel({'a': 3.0}, {}, -1.5, {'a': 'BINARY'}, dtype='float64')
            >>> qm.energy({a: 1})
            1.5
            >>> qm.energy({a: 0})
            -1.5

        """
        self.data.change_vartype(vartype, v)
        return self

    def clear(self) -> None:
        """Remove the offset and all variables and interactions from the model."""
        self.data.clear()

    def copy(self):
        """Return a copy."""
        return deepcopy(self)

    @forwarding_method
    def degree(self, v: Variable) -> int:
        """Return the degree of specified variable.

        The degree is the number of interactions that contain a variable, ``v``.

        Args:
            v: Variable in the quadratic model.
        """
        return self.data.degree

    def energies(self, samples_like, dtype: typing.Optional[DTypeLike] = None) -> np.ndarray:
        """Determine the energies of the given samples-like.

        Args:
            samples_like (samples_like):
                Raw samples. `samples_like` is an extension of
                NumPy's `array_like`_ structure. See :func:`.as_samples`.

            dtype:
                Desired NumPy data type for the energy.
                Defaults to :class:`~numpy.float64`.

        Returns:
            Energies for the samples.

        Examples:
            >>> from dimod import QuadraticModel, Binary
            >>> qm = QuadraticModel()
            >>> qm.add_variables_from('BINARY', ['x', 'y'])
            >>> qm.add_quadratic('x', 'y', -2)
            >>> qm.energies([{'x': 1, 'y': 0}, {'x': 0, 'y': 0}, {'x': 1, 'y': 1}])
            array([ 0.,  0., -2.])

        .. _`array_like`:  https://numpy.org/doc/stable/user/basics.creation.html

        """
        return self.data.energies(samples_like, dtype=dtype)

    def energy(self, sample, dtype=None) -> Bias:
        """Determine the energy of the given sample.

        Args:
            sample (samples_like):
                Raw sample. `samples_like` is an extension of
                NumPy's `array_like`_ structure. See :func:`.as_samples`.

            dtype:
                Desired NumPy data type for the energy.
                Defaults to :class:`~numpy.float64`.

        Returns:
            Energy for the sample.

        Examples:
            >>> from dimod import QuadraticModel, Binary
            >>> qm = QuadraticModel()
            >>> qm.add_variables_from('BINARY', ['x', 'y'])
            >>> qm.add_quadratic('x', 'y', -2)
            >>> qm.energy([{'x': 1, 'y': 1}])
            -2.0

        .. _`array_like`:  https://numpy.org/doc/stable/user/basics.creation.html

        """
        energies = self.energies(sample, dtype=dtype)

        if not len(energies):  # the empty case, happens with []
            return self.dtype.type(0)

        energy, = energies
        return energy

    def flip_variable(self, v: Variable):
        r"""Flip the specified binary-valued variable.

        Args:
            v: Binary-valued (:math:`\{0, 1\}` or :math:`\{-1, 1\}`) variable in
                the quadratic model.

        Raises:
            ValueError: If ``v`` is not a variable in the model.
            ValueError: If ``v`` is not a :class:`Vartype.SPIN` or
                :class:`Vartype.BINARY` variable.

        Examples:
            In this example we flip the value of a binary variable ``x``. That
            is we substitute ``(1 - x)`` which always takes the opposite value.

            >>> x = dimod.Binary('x')
            >>> s = dimod.Spin('s')
            >>> qm = x + 2*s + 3*x*s
            >>> qm.flip_variable('x')
            >>> qm.is_equal((1-x) + 2*s + 3*(1-x)*s)
            True

            In this example we flip the value of a spin variable ``s``. That
            is we substitute ``-s`` which always takes the opposite value.

            >>> x = dimod.Binary('x')
            >>> s = dimod.Spin('s')
            >>> qm = x + 2*s + 3*x*s
            >>> qm.flip_variable('s')
            >>> qm.is_equal(x + 2*-s + 3*x*-s)
            True

        """
        vartype = self.vartype(v)
        if vartype is Vartype.SPIN:
            for u, bias in self.iter_neighborhood(v):
                self.set_quadratic(u, v, -1*bias)
            self.set_linear(v, -1*self.get_linear(v))
        elif vartype is Vartype.BINARY:
            for u, bias in self.iter_neighborhood(v):
                self.set_quadratic(u, v, -1*bias)
                self.add_linear(u, bias)
            self.offset += self.get_linear(v)
            self.set_linear(v, -1*self.get_linear(v))
        else:
            raise ValueError(f"can only flip SPIN and BINARY variables, {v} is {vartype.name}")

    @classmethod
    def from_bqm(cls, bqm: 'BinaryQuadraticModel') -> 'QuadraticModel':
        """Construct a quadratic model from a binary quadratic model.

        Args:
            bqm: Binary quadratic model from which to create the quadratic model.

        Returns:
            Quadratic model.

        Examples:
            >>> from dimod import QuadraticModel, BinaryQuadraticModel
            >>> bqm = BinaryQuadraticModel({'a': 0.1, 'b': 0.2}, {'ab': -1}, 'SPIN')
            >>> qm = QuadraticModel.from_bqm(bqm)
        """
        obj = cls.__new__(cls)

        try:
            obj.data = obj._DATA_CLASSES[np.dtype(bqm.dtype)].from_cybqm(bqm.data)
        except (TypeError, KeyError):
            # not a cybqm or unsupported dtype
            obj = cls()
        else:
            return obj

        # fallback to python
        for v in bqm.variables:
            obj.set_linear(obj.add_variable(bqm.vartype, v), bqm.get_linear(v))

        for u, v, bias in bqm.iter_quadratic():
            obj.set_quadratic(u, v, bias)

        obj.offset = bqm.offset

        return obj

    @classmethod
    def from_file(cls, fp: typing.Union[typing.BinaryIO, typing.Union[bytes, bytearray]]):
        """Construct a quadratic model from a file-like object.

        The inverse of :meth:`~QuadraticModel.to_file`.
        """
        if isinstance(fp, (bytes, bytearray, memoryview)):
            file_like: typing.BinaryIO = _BytesIO(fp)  # type: ignore[assignment]
        else:
            file_like = fp

        header_info = read_header(file_like, QM_MAGIC_PREFIX)

        num_variables, num_interactions = header_info.data['shape']
        dtype = np.dtype(header_info.data['dtype'])
        itype = np.dtype(header_info.data['itype'])

        if header_info.version > (2, 0):
            raise ValueError("cannot load a QM serialized with version "
                             f"{header_info.version!r}, "
                             "try upgrading your dimod version")

        obj = cls(dtype=dtype)

        # the vartypes
        obj.data._ivartypes_load(VartypesSection.load(file_like), num_variables)

        # offset
        obj.offset += OffsetSection.load(file_like, dtype=dtype)

        # linear
        obj.data.add_linear_from_array(
            LinearSection.load(file_like, dtype=dtype, num_variables=num_variables))

        # quadratic
        for vi in range(num_variables):
            obj.data._ilower_triangle_load(vi, *NeighborhoodSection.load(file_like))

        # labels (if applicable)
        if header_info.data['variables']:
            obj.relabel_variables(dict(enumerate(VariablesSection.load(file_like))))

        return obj

    @forwarding_method
    def get_linear(self, v: Variable) -> Bias:
        """Get the linear bias of the specified variable.

        Args:
            v: Variable in the quadratic model.
        """
        return self.data.get_linear

    @forwarding_method
    def get_quadratic(self, u: Variable, v: Variable,
                      default: typing.Optional[Bias] = None) -> Bias:
        """Get the quadratic bias of the specified pair of variables.

        Args:
            u: Variable in the quadratic model.
            v: Variable in the quadratic model.
            default: Value to return if variables ``u`` and ``v`` have no interaction.
        """
        return self.data.get_quadratic

    def is_almost_equal(self, other: typing.Union['QuadraticModel', 'BinaryQuadraticModel', Bias],
                        places: int = 7) -> bool:
        """Test for near equality to all biases of a given quadratic model.

        Args:
            other:
                Quadratic model with which to compare biases.
            places:
                Number of decimal places to which the Python :func:`round`
                function calculates approximate equality.

        Examples:
            >>> from dimod import QuadraticModel
            >>> qm1 = QuadraticModel({'x': 0.0, 'i': 0.1234}, {('i', 'x'): -1.1234},
            ...                      0.0, {'x': 'BINARY', 'i': 'INTEGER'})
            >>> qm2 = QuadraticModel({'x': 0.0, 'i': 0.1232}, {('i', 'x'): -1.1229},
            ...                      0.0, {'x': 'BINARY', 'i': 'INTEGER'})
            >>> qm1.is_almost_equal(qm2, 4)
            False
            >>> qm1.is_almost_equal(qm2, 3)
            True
        """
        if isinstance(other, Number):
            return not (self.num_variables or round(self.offset - other, places))

        def eq(a, b):
            return not round(a - b, places)

        try:
            if callable(other.vartype):
                vartype_eq = all(self.vartype(v) == other.vartype(v) for v in self.variables)
            else:
                vartype_eq = all(self.vartype(v) == other.vartype for v in self.variables)

            return (vartype_eq
                    and self.shape == other.shape
                    and eq(self.offset, other.offset)
                    and all(eq(self.get_linear(v), other.get_linear(v))
                            for v in self.variables)
                    and all(eq(bias, other.get_quadratic(u, v))
                            for u, v, bias in self.iter_quadratic())
                    )
        except (AttributeError, ValueError):
            # it's not a BQM or variables/interactions don't match
            return False

    def is_equal(self, other: typing.Union['QuadraticModel', Number]) -> bool:
        """Return True if the given model has the same variables, vartypes and biases.

        Args:
            other: Quadratic model to compare against.
        """
        if isinstance(other, Number):
            return not self.num_variables and bool(self.offset == other)
        # todo: performance
        try:
            if callable(other.vartype):
                vartype_eq = all(self.vartype(v) == other.vartype(v) for v in self.variables)
            else:
                vartype_eq = all(self.vartype(v) == other.vartype for v in self.variables)

            return (vartype_eq
                    and self.shape == other.shape  # redundant, fast to check
                    and self.offset == other.offset
                    and self.linear == other.linear
                    and self.adj == other.adj)
        except AttributeError:
            return False

    def is_linear(self) -> bool:
        """Return True if the model has no quadratic interactions."""
        return self.data.is_linear()

    @forwarding_method
    def iter_neighborhood(self, v: Variable) -> collections.abc.Iterator[tuple[Variable, Bias]]:
        """Iterate over the neighbors and quadratic biases of a variable.

        Args:
            v: Variable in the quadratic model.

        Returns:
            Neighbors of the specified variable and their quadratic biases.

        Examples:
            >>> from dimod import QuadraticModel
            >>> qm = QuadraticModel()
            >>> qm.add_variables_from('BINARY', ['x', 'y', 'z'])
            >>> qm.add_quadratic('x', 'y', -2)
            >>> qm.add_quadratic('x', 'z', 2)
            >>> list(qm.iter_neighborhood('x'))
            [('y', -2.0), ('z', 2.0)]

        """
        return self.data.iter_neighborhood

    @forwarding_method
    def iter_quadratic(self) -> collections.abc.Iterator[tuple[Variable, Variable, Bias]]:
        """Iterate over the interactions of a quadratic model.

        Returns:
            Interactions of the quadratic model and their biases.

        Examples:
            >>> from dimod import QuadraticModel
            >>> qm = QuadraticModel()
            >>> qm.add_variables_from('BINARY', ['x', 'y', 'z'])
            >>> qm.add_quadratic('x', 'y', -2)
            >>> qm.add_quadratic('x', 'z', 2)
            >>> list(qm.iter_quadratic())
            [('y', 'x', -2.0), ('z', 'x', 2.0)]

        """
        return self.data.iter_quadratic

    @forwarding_method
    def lower_bound(self, v: Variable) -> Bias:
        """Return the lower bound on the specified variable.

        Args:
            v: Variable in the quadratic model.
        """
        return self.data.lower_bound

    def nbytes(self, capacity: bool = False) -> int:
        """Get the total bytes consumed by the biases, vartype info, bounds,
        and indices.

        Does not include the memory consumed by non-element attributes of
        the quadratic model object.
        Also does not include the memory consumed by the variable labels.

        Args:
            capacity: If ``capacity`` is true, also include the ``std::vector::capacity``
            of the underlying vectors in the calculation.

        Returns:
            The number of bytes.

        """
        return self.data.nbytes(capacity)

    def set_lower_bound(self, v: Variable, lb: float):
        """Set the lower bound for a variable.

        Args:
            v: Variable in the quadratic model.
            lb: Lower bound to set for variable ``v``.

        Raises:
            ValueError: If ``v`` is a :class:`~dimod.Vartype.SPIN`
                or :class:`~dimod.Vartype.BINARY` variable.

        """
        return self.data.set_lower_bound(v, lb)

    def set_upper_bound(self, v: Variable, ub: float):
        """Set the upper bound for a variable.

        Args:
            v: Variable in the quadratic model.
            ub: Upper bound to set for variable ``v``.

        Raises:
            ValueError: If ``v`` is a :class:`~dimod.Vartype.SPIN`
                or :class:`~dimod.Vartype.BINARY` variable.

        """
        return self.data.set_upper_bound(v, ub)

    @forwarding_method
    def reduce_linear(self, function: collections.abc.Callable,
                      initializer: typing.Optional[Bias] = None) -> typing.Any:
        """Apply function of two arguments cumulatively to the linear biases.

        Args:
            function: Function of two arguments to apply to the linear biases.

            initializer: Prefixed in the calculation to the iterable containing
                the linear biases or used as the default if no linear biases are
                set in the quadratic model.

        Returns:
            Result of applying the specified function to the linear biases.

        Examples:
            >>> from operator import add
            >>> from dimod import QuadraticModel
            >>> qm = QuadraticModel({'x': 0.5, 's': 1, 'i': 2},
            ...                     {('x', 'i'): 2}, 0.0,
            ...                     {'x': 'BINARY', 's': 'SPIN', 'i': 'INTEGER'})
            >>> qm.reduce_linear(add)
            3.5

        For information on the related functional programming method
        see :func:`functools.reduce`.
        """
        return self.data.reduce_linear

    @forwarding_method
    def reduce_neighborhood(self, v: Variable, function: collections.abc.Callable,
                            initializer: typing.Optional[Bias] = None) -> typing.Any:
        """Apply function of two arguments cumulatively to the quadratic biases
        associated with a single variable.

        Args:
            v: Variable in the quadratic model.

            function: Function of two arguments to apply to the quadratic biases
                of variable ``v``.

            initializer: Prefixed in the calculation to the iterable containing
                the quadratic biases or used as the default if variable ``v`` has
                no quadratic biases.

        Returns:
            Result of applying the specified function to the specified variable's
            quadratic biases.

        Examples:
            >>> from dimod import QuadraticModel
            >>> qm = QuadraticModel({'x': 0.5, 's': 1, 'i': 2},
            ...                     {('x', 'i'): 2, ('s', 'i'): 3}, 0.0,
            ...                     {'x': 'BINARY', 's': 'SPIN', 'i': 'INTEGER'})
            >>> qm.reduce_neighborhood('i', max)
            3.0

        For information on the related functional programming method
        see :func:`functools.reduce`.
        """
        return self.data.reduce_neighborhood

    @forwarding_method
    def reduce_quadratic(self, function: collections.abc.Callable,
                         initializer: typing.Optional[Bias] = None) -> typing.Any:
        """Apply function of two arguments cumulatively to the quadratic
        biases.

        Args:
            function: Function of two arguments to apply to the quadratic biases.

            initializer: Prefixed in the calculation to the iterable containing
                the quadratic biases or used as the default if no quadratic biases
                are set in the quadratic model.

        Returns:
            Result of applying the specified function to the quadratic biases.

        Examples:
            >>> from dimod import QuadraticModel
            >>> qm = QuadraticModel({'x': 0.5, 's': 1, 'i': 2},
            ...                     {('x', 'i'): 2, ('s', 'i'): 3}, 0.0,
            ...                     {'x': 'BINARY', 's': 'SPIN', 'i': 'INTEGER'})
            >>> qm.reduce_quadratic(min)
            2.0

        For information on the related functional programming method
        see :func:`functools.reduce`.
        """
        return self.data.reduce_quadratic

    def relabel_variables(self, mapping: collections.abc.Mapping[Variable, Variable],
                          inplace: bool = True) -> 'QuadraticModel':
        """Relabel the variables according to the given mapping.

        Args:
            mapping: Mapping of current variable labels to new ones. If an
                incomplete mapping is provided, unmapped variables retain their
                current labels.

            inplace: If set to False, returns a new quadratic model
                mapped to the new labels.

        Returns:
            The original or new quadratic model with updated variable labels.

        Examples:
            >>> from dimod import QuadraticModel, BinaryQuadraticModel, generators
            >>> bqm = generators.ran_r(1, 5)
            >>> qm = QuadraticModel.from_bqm(bqm)
            >>> qm_new = qm.relabel_variables({0: 'a', 1: 'b', 2: 'c'}, inplace=False)
            >>> qm_new.variables
            Variables(['a', 'b', 'c', 3, 4])

        """
        if not inplace:
            return self.copy().relabel_variables(mapping, inplace=True)

        self.data.relabel_variables(mapping)
        return self

    def relabel_variables_as_integers(self, inplace: bool = True
                                      ) -> tuple['QuadraticModel', collections.abc.Mapping[Variable, Variable]]:
        """Relabel the variables as `[0, n)` and return the mapping.

        Args:
            inplace: If set to False, returns a new quadratic model
                mapped to the new labels.

        Returns:
            The original or new quadratic model with updated variable labels.

        """
        if not inplace:
            return self.copy().relabel_variables_as_integers(inplace=True)

        mapping = self.data.relabel_variables_as_integers()
        return self, mapping

    @forwarding_method
    def remove_interaction(self, u: Variable, v: Variable):
        """Remove the interaction between `u` and `v`.

        Args:
            u: Variable in the quadratic model.
            v: Variable in the quadratic model.
        """
        return self.data.remove_interaction

    @forwarding_method
    def remove_variable(self, v: typing.Optional[Variable] = None) -> Variable:
        """Remove the specified variable from the quadratic model.

        Args:
            v: Variable in the quadratic model.
        """
        return self.data.remove_variable

    @forwarding_method
    def scale(self, scalar: Bias):
        """Scale the biases by the given number.

        Args:
            scalar: Value by which to scale the biases of the quadratic model.

        Examples:
            >>> from dimod import QuadraticModel
            >>> qm = QuadraticModel()
            >>> qm.add_variables_from('INTEGER', ['i', 'j'])
            >>> qm.set_linear('i', 2)
            >>> qm.set_quadratic('i', 'j', -1)
            >>> qm.scale(1.5)
            >>> print(qm.get_linear('i'), qm.get_quadratic('i', 'j'))
            3.0 -1.5
        """
        return self.data.scale

    @forwarding_method
    def set_linear(self, v: Variable, bias: Bias):
        """Set the linear bias of a variable in the quadratic model.

        Args:
            v: Variable in the quadratic model.
            bias: Linear bias to set for variable ``v``.

        Raises:
            TypeError: If `v` is not hashable.
            ValueError: If the variable is not in the model.
        """
        return self.data.set_linear

    @forwarding_method
    def set_quadratic(self, u: Variable, v: Variable, bias: Bias):
        """Set the quadratic bias between a pair of variables in the quadratic model.

        Args:
            u: Variable in the quadratic model.
            v: Variable in the quadratic model.
            bias: Quadratic bias to set for interaction ``(u, v)``.

        Raises:
            TypeError: If ``u`` or ``v`` is not hashable.
            ValueError: If a variable is not in the model or if ``u == v`` for
                binary-valued variables (self-loops are not allowed for such
                variables).
        """
        return self.data.set_quadratic

    def spin_to_binary(self, inplace: bool = False) -> 'QuadraticModel':
        """Convert any spin-valued variables to binary-valued.

        Args:
            inplace: If set to False, returns a new quadratic model
                with spin-valued variables converted to binary-valued variables.

        Examples:
            >>> from dimod import QuadraticModel
            >>> qm = QuadraticModel()
            >>> qm.add_variables_from('SPIN', ['s1', 's2'])
            >>> qm.add_variable('BINARY', 'b')
            'b'
            >>> qm_b = qm.spin_to_binary(inplace=False)
            >>> qm_b.vartype('s1')
            <Vartype.BINARY: frozenset({0, 1})>
        """
        if not inplace:
            return self.copy().spin_to_binary(inplace=True)

        for s in self.variables:
            if self.vartype(s) is Vartype.SPIN:
                self.change_vartype(Vartype.BINARY, s)

        return self

    def to_file(self, *,
                spool_size: int = int(1e9),
                ) -> tempfile.SpooledTemporaryFile:
        """Serialize the QM to a file-like object.

        Args:
            spool_size: Defines the `max_size` passed to the constructor of
                :class:`tempfile.SpooledTemporaryFile`. Determines whether
                the returned file-like's contents will be kept on disk or in
                memory.

        Format Specification (Version 1.0):

            This format is inspired by the `NPY format`_

            The first 7 bytes are a magic string: exactly "DIMODQM".

            The next 1 byte is an unsigned byte: the major version of the file
            format.

            The next 1 byte is an unsigned byte: the minor version of the file
            format.

            The next 4 bytes form a little-endian unsigned int, the length of
            the header data HEADER_LEN.

            The next HEADER_LEN bytes form the header data. This is a
            json-serialized dictionary. The dictionary is exactly:

            .. code-block:: python

                data = dict(shape=qm.shape,
                            dtype=qm.dtype.name,
                            itype=qm.data.index_dtype.name,
                            type=type(qm).__name__,
                            variables=not qm.variables._is_range(),
                            )

            it is terminated by a newline character and padded with spaces to
            make the entire length of the entire header divisible by 64.

            The quadratic model data comes after the header.

        .. _NPY format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html

        """
        # todo: document the serialization format sections

        file = SpooledTemporaryFile(max_size=spool_size)

        data = dict(shape=self.shape,
                    dtype=self.dtype.name,
                    itype=self.data.index_dtype.name,
                    type=type(self).__name__,
                    variables=not self.variables._is_range(),
                    )

        write_header(file, QM_MAGIC_PREFIX, data, version=(1, 0))

        # the vartypes
        file.write(VartypesSection(self.data).dumps())

        # offset
        file.write(OffsetSection(self.data).dumps())

        # linear
        file.write(LinearSection(self.data).dumps())

        # quadraic
        neighborhood_section = NeighborhoodSection(self.data)
        for vi in range(self.num_variables):
            file.write(neighborhood_section.dumps(vi=vi))

        # the labels (if needed)
        if data['variables']:
            file.write(VariablesSection(self.variables).dumps())

        file.seek(0)
        return file

    def update(self, other: typing.typing.Union[QuadraticModel, BinaryQuadraticModel]):
        """Update the quadratic model from another quadratic model.

        Adds to the quadratic model the variables, linear biases, quadratic biases,
        and offset of the specified quadratic model. Linear and quadratic biases
        for variables that exist in both models are summed.

        Args:
            other: Another quadratic model.

        Raises:
            ValueError: If a variable exists in both quadratic models but has
                different vartypes or, for integer variables, different bounds.

        Examples:
            >>> from dimod import QuadraticModel
            >>> qm1 = QuadraticModel({'s1': -2.0, 's2': 0.0, 's3': 0.0, 's0': 0.0},
            ...         {('s2', 's1'): -1.0, ('s3', 's2'): 1.0}, 0.0,
            ...         {'s1': 'SPIN', 's2': 'SPIN', 's3': 'SPIN', 's0': 'SPIN'},
            ...         dtype='float64')
            >>> qm2 = QuadraticModel({'s1': -2.0, 's2': 0.0, 's3': 0.0, 's4': -3.0},
            ...         {('s2', 's1'): -1.0, ('s3', 's1'): 1.0}, 0.0,
            ...         {'s1': 'SPIN', 's2': 'SPIN', 's3': 'SPIN', 's4': 'SPIN'},
            ...         dtype='float64')
            >>> qm1.update(qm2)
            >>> print(qm1.get_linear('s0'), qm1.get_linear('s1'), qm1.get_linear('s4'))
            0.0 -4.0 -3.0
            >>> print(qm1.get_quadratic('s2', 's1'), qm1.get_quadratic('s3', 's1'))
            -2.0 1.0
        """
        try:
            return self.data.update(other.data)
        except (AttributeError, TypeError):
            pass

        # looks like we have a model that either has object dtype or isn't
        # a cython model we recognize, so let's fall back on python

        # need a couple methods to be generic between bqm and qm
        vartype = other.vartype if callable(other.vartype) else lambda v: other.vartype

        def lower_bound(v: Variable) -> Bias:
            try:
                return other.lower_bound(v)
            except AttributeError:
                pass

            if other.vartype is Vartype.SPIN:
                return -1
            elif other.vartype is Vartype.BINARY:
                return 0
            else:
                raise RuntimeError  # shouldn't ever happen

        def upper_bound(v: Variable) -> Bias:
            try:
                return other.upper_bound(v)
            except AttributeError:
                pass

            return 1

        for v in other.variables:
            if v not in self.variables:
                continue

            if self.vartype(v) != vartype(v):
                raise ValueError(f"conflicting vartypes: {v!r}")
            if self.lower_bound(v) != lower_bound(v):
                raise ValueError(f"conflicting lower bounds: {v!r}")
            if self.upper_bound(v) != upper_bound(v):
                raise ValueError(f"conflicting upper bounds: {v!r}")

        for v in other.variables:
            self.add_linear(self.add_variable(vartype(v), v,
                                              lower_bound=lower_bound(v),
                                              upper_bound=upper_bound(v)),
                            other.get_linear(v))

        for u, v, bias in other.iter_quadratic():
            self.add_quadratic(u, v, bias)

        self.offset += other.offset

    @forwarding_method
    def upper_bound(self, v: Variable) -> Bias:
        """Return the upper bound on the specified variable.

        Args:
            v: Variable in the quadratic model.
        """
        return self.data.upper_bound

    @forwarding_method
    def vartype(self, v: Variable) -> Vartype:
        """The variable type of the given variable.

        Args:
            v: Variable in the quadratic model.
        """
        return self.data.vartype


QM = QuadraticModel


@unique_variable_labels
def Integer(label: typing.Optional[Variable] = None, bias: Bias = 1,
            dtype: typing.Optional[DTypeLike] = None,
            *, lower_bound: float = 0, upper_bound: typing.Optional[float] = None) -> QuadraticModel:
    """Return a quadratic model with a single integer variable.

    Args:
        label: Hashable label to identify the variable. Defaults to a
            generated :class:`uuid.UUID` as a string.
        bias: The bias to apply to the variable.
        dtype: Data type for the returned quadratic model.
        lower_bound: Keyword-only argument to specify integer lower bound.
        upper_bound: Keyword-only argument to specify integer upper bound.

    Returns:
        Instance of :class:`~dimod.QuadraticModel`.

    Examples:
        This example generates a quadratic model to represent the polynomial,
        :math:`3i - 1.5`, where :math:`i` is an integer variable.

        >>> i = dimod.Integer('i')
        >>> qm = 3*i - 1.5
        >>> print(qm.to_polystring())
        -1.5 + 3*i
    """
    qm = QM(dtype=dtype)
    v = qm.add_variable(Vartype.INTEGER, label, lower_bound=lower_bound, upper_bound=upper_bound)
    qm.set_linear(v, bias)
    return qm

def Integers(labels: typing.Union[int, collections.abc.Iterable[Variable]],
             dtype: typing.Optional[DTypeLike] = None) -> collections.abc.Iterator[QuadraticModel]:
    """Yield quadratic models, each with a single integer variable.

    Args:
        labels: Either an iterable of variable labels or a number. If a number
            labels are generated using :class:`uuid.UUID`.
        dtype: Data type for the returned quadratic models.

    Yields:
        A :class:`~dimod.QuadraticModel` for each integer variable.

    Examples:
        >>> i, j = dimod.Integers(['i', 'j'])
        >>> qm = 2*(pow(i, 2) + pow(j, 2)) - 3*i*j - i - j
        >>> print(qm.to_polystring())
        -i - j + 2*i*i - 3*i*j + 2*j*j
    """
    if isinstance(labels, collections.abc.Iterable):
        yield from (Integer(v, dtype=dtype) for v in labels)
    else:
        yield from (Integer(dtype=dtype) for _ in range(labels))


def IntegerArray(labels: typing.Union[int, collections.abc.Iterable[Variable]],
                 dtype: typing.Optional[DTypeLike] = None) -> np.ndarray:
    """Return a NumPy array of quadratic models, each with a
    single integer variable.

    Args:
        labels: Either an iterable of variable labels or a number. If a number
            labels are generated using :class:`uuid.UUID`.
        dtype: Data type for the returned quadratic models.

    Returns:
        Array of quadratic models, each with a single integer variable.

    Examples:
        This example creates a quadratic model that represents a clique
        (fully-connected graph) of three nodes with integer values.

        >>> import numpy as np
        >>> i = dimod.IntegerArray(["i0", "i1", "i2"])
        >>> j = dimod.IntegerArray(["j0", "j1", "j2"])
        >>> qm = dimod.quicksum(dimod.quicksum(np.outer(i, j)))
        >>> print(qm.to_polystring())
        i0*j0 + j0*i1 + j0*i2 + i0*j1 + i1*j1 + i2*j1 + i0*j2 + i1*j2 + i2*j2
    """
    return _VariableArray(Integers, labels, dtype)


def _VariableArray(variable_generator: collections.abc.Callable,
                   labels: typing.Union[int, collections.abc.Iterable[Variable]],
                   dtype: typing.Optional[DTypeLike] = None) -> np.ndarray:
    """Builds NumPy array from a variable generator method."""
    if isinstance(labels, int):
        number_of_elements = labels
    elif isinstance(labels, collections.abc.Sized):
        number_of_elements = len(labels)
    else:
        labels = list(labels)
        number_of_elements = len(labels)

    variable_array = np.empty(number_of_elements, dtype=object)
    for index, element in enumerate(variable_generator(labels, dtype)):
        variable_array[index] = element

    return variable_array


@unique_variable_labels
def Real(label: typing.Optional[Variable] = None, bias: Bias = 1,
         dtype: typing.Optional[DTypeLike] = None,
         *, lower_bound: float = 0, upper_bound: typing.Optional[float] = None) -> QuadraticModel:
    """Return a quadratic model with a single real-valued variable.

    Args:
        label: Hashable label to identify the variable. Defaults to a
            generated :class:`uuid.UUID` as a string.
        bias: The bias to apply to the variable.
        dtype: Data type for the returned quadratic model.
        lower_bound: Keyword-only argument to specify the lower bound.
        upper_bound: Keyword-only argument to specify the upper bound.

    Returns:
        Instance of :class:`.QuadraticModel`.

    """
    qm = QM(dtype=dtype)
    v = qm.add_variable(Vartype.REAL, label, lower_bound=lower_bound, upper_bound=upper_bound)
    qm.set_linear(v, bias)
    return qm


def Reals(labels: typing.Union[int, collections.abc.Iterable[Variable]],
          dtype: typing.Optional[DTypeLike] = None) -> collections.abc.Iterator[QuadraticModel]:
    """Yield quadratic models, each with a single real-valued variable.

    Args:
        labels: Either an iterable of variable labels or a number. If a number
            labels are generated using :class:`uuid.UUID`.
        dtype: Data type for the returned quadratic models.

    Yields:
        Quadratic models, each with a single real-valued variable.

    """
    if isinstance(labels, collections.abc.Iterable):
        yield from (Real(v, dtype=dtype) for v in labels)
    else:
        yield from (Real(dtype=dtype) for _ in range(labels))

# register fileview loader
load.register(QM_MAGIC_PREFIX, QuadraticModel.from_file)
