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

r"""Binary quadratic models (BQMs) are problems of the form:

.. math::

    E(\bf{v})
    = \sum_{i} a_i v_i
    + \sum_{i<j} b_{i,j} v_i v_j
    + c
    \qquad\qquad v_i \in\{-1,+1\} \text{  or } \{0,1\}

where :math:`a_{i}, b_{ij}, c` are real values.
"""

from __future__ import annotations

import collections.abc as abc
import copy
import itertools
import io
import json
import operator
import tempfile
import typing
import warnings

from numbers import Integral, Number
from typing import (Any, BinaryIO, Callable, Dict,
                    Hashable, Iterable, Iterator,
                    Mapping, MutableMapping, Optional, Sequence,
                    Tuple, Union, Literal
                    )

import numpy as np

try:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    ArrayLike = Any
    DTypeLike = Any

from dimod.binary.cybqm import cyBQM_float32, cyBQM_float64
from dimod.binary.pybqm import pyBQM
from dimod.binary.vartypeview import VartypeView
from dimod.decorators import forwarding_method, unique_variable_labels
from dimod.quadratic import QuadraticModel, QM
from dimod.quadratic.quadratic_model import _VariableArray
from dimod.serialization.fileview import SpooledTemporaryFile, _BytesIO, VariablesSection
from dimod.serialization.fileview import load, read_header, write_header
from dimod.sym import Eq, Ge, Le
from dimod.typing import (Bias, BQMVectors, LabelledBQMVectors, QuadraticVectors,
                          Variable, VartypeLike)
from dimod.variables import Variables, iter_deserialize_variables
from dimod.vartypes import as_vartype, Vartype
from dimod.views.quadratic import QuadraticViewsMixin

__all__ = ['BinaryQuadraticModel',
           'BQM',
           'DictBQM',
           'Float32BQM',
           'Float64BQM',
           'as_bqm',
           'Spin', 'Binary', 'Spins', 'Binaries', 'SpinArray', 'BinaryArray',
           'quicksum',
           'AdjDictBQM', 'AdjVectorBQM',
           ]

BQM_MAGIC_PREFIX = b'DIMODBQM'


class BinaryQuadraticModel(QuadraticViewsMixin):
    r"""Binary quadratic model.

    This class encodes Ising and quadratic unconstrained binary optimization
    (QUBO) models used by samplers such as the D-Wave system.

    With one or more of the following parameters,

            * ``vartype``: The valid variable types for binary quadratic models, is
              one of:

                * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, +1}``
                * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            * ``bqm``: An existing BQM.

            * ``n``: Required number of variables.

            * ``quadratic``: Quadratic biases, as a dictionary of form
              ``{(u, v): b, ...}`` or a square array_like_.

            * ``linear``: Linear biases, as a dictionary of the form ``{v: b, ...}``
              or a one-dimensional array_like_.

            * ``offset``: Offset as a number.

    you can create BQMs in several ways:

    * ``BinaryQuadraticModel(vartype)`` with no variables or interactions.

    * ``BinaryQuadraticModel(bqm)`` from an existing BQM. The resulting
      BQM has the same variables, linear biases, quadratic biases and offset as
      ``bqm``.

    * ``BinaryQuadraticModel(bqm, vartype)`` from an existing BQM, changing to
      the specified ``vartype`` if necessary.

    * ``BinaryQuadraticModel(n, vartype)`` with ``n`` variables, indexed linearly
      from zero, setting all biases to zero.

    * ``BinaryQuadraticModel(quadratic, vartype)`` from quadratic biases. When
      formed with SPIN-variables, biases on the diagonal are added to the offset.

    * ``BinaryQuadraticModel(linear, quadratic, vartype)`` from linear and
      quadratic biases.

    * ``BinaryQuadraticModel(linear, quadratic, offset, vartype)`` from linear
      and quadratic biases and an offset.

    Args:
        *args: See above.

        offset: Offset (see above) may be supplied as a keyword argument.

        vartype: Variable type (see above) may be supplied as a keyword
            argument.

        dtype: Data type.
            :class:`numpy.float32` and :class:`numpy.float64` are supported.
            Defaults to :class:`numpy.float64`.

    .. _array_like: https://numpy.org/doc/stable/user/basics.creation.html

    """

    _DATA_CLASSES = {
        np.dtype(np.float32): cyBQM_float32,
        np.dtype(np.float64): cyBQM_float64,
        np.dtype(object): pyBQM,
    }

    DEFAULT_DTYPE = np.float64
    """The default dtype used to construct the class."""

    def __init__(self, *args,
                 offset: Optional[Bias] = None,
                 vartype: Optional[VartypeLike] = None,
                 dtype: Optional[DTypeLike] = None):

        if vartype is not None:
            args = [*args, vartype]

        # developer note: I regret ever setting up this construction system
        # but it's gotten out there so we're stuck with it
        # I would like to reestablish kwarg construction at some point
        if len(args) == 0:
            raise TypeError("A valid vartype or another bqm must be provided")
        if len(args) == 1:
            # BQM(bqm) or BQM(vartype)
            if hasattr(args[0], 'vartype'):
                # bqm case
                if offset is not None:
                    # see note for (linear, quadratic, offset, vartype) below
                    raise TypeError("cannot provide 'offset' when input is a binary quadratic model")
                self._init_bqm(args[0], vartype=args[0].vartype, dtype=dtype)
            else:
                self._init_empty(vartype=args[0], dtype=dtype)
        elif len(args) == 2:
            # BQM(bqm, vartype), BQM(n, vartype) or BQM(M, vartype)
            if isinstance(args[0], Integral):
                self._init_empty(vartype=args[1], dtype=dtype)
                self.resize(args[0])
            elif hasattr(args[0], 'vartype'):
                if offset is not None:
                    # see note for (linear, quadratic, offset, vartype) below
                    raise TypeError("cannot provide 'offset' when input is a binary quadratic model")
                self._init_bqm(args[0], vartype=args[1], dtype=dtype)
            else:
                self._init_components([], args[0], 0.0, args[1], dtype=dtype)
        elif len(args) == 3:
            # BQM(linear, quadratic, vartype)
            self._init_components(args[0], args[1], 0.0, args[2], dtype=dtype)
        elif len(args) == 4:
            # BQM(linear, quadratic, offset, vartype)

            if offset is not None:
                # we don't strictly need to fail in this case, we could instead
                # add it, but I think this is closer to the normal python behavior
                # of failing if an argument is provided twice
                raise TypeError("BinaryQuadraticModel() got multiple values for 'offset'")

            self._init_components(*args, dtype=dtype)
        else:
            msg = "__init__() takes 4 positional arguments but {} were given."
            raise TypeError(msg.format(len(args)))

        # we already checked the one case that doesn't support offset
        if offset is not None:
            self.offset += offset

    def _init_bqm(self, bqm, vartype, dtype):
        if dtype is None:
            dtype = bqm.dtype
        if vartype is None:
            vartype = bqm.vartype
        self.data = type(self)._DATA_CLASSES[np.dtype(dtype)](vartype)
        self.update(bqm)

    def _init_components(self, linear, quadratic, offset, vartype, dtype):
        self._init_empty(vartype, dtype)

        self.offset = offset

        vartype = self.vartype

        if isinstance(quadratic, (abc.Mapping, abc.Iterator)):
            if isinstance(quadratic, abc.Mapping):
                quadratic = ((u, v, b) for (u, v), b in quadratic.items())

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

        if isinstance(linear, (abc.Iterator, abc.Mapping)):
            self.add_linear_from(linear)
        else:
            self.add_linear_from_array(linear)

    def _init_empty(self, vartype, dtype):
        dtype = self.DEFAULT_DTYPE if dtype is None else dtype
        self.data = type(self)._DATA_CLASSES[np.dtype(dtype)](vartype)

    def __init_subclass__(cls, default_dtype=np.float64, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.DEFAULT_DTYPE = np.dtype(default_dtype)

    def __copy__(self):
        new = type(self).__new__(type(self))
        new.data = copy.copy(self.data)
        return new

    def __deepcopy__(self, memo):
        new = type(self).__new__(type(self))
        new.data = copy.deepcopy(self.data, memo)
        memo[id(self)] = new
        return new

    def __len__(self):
        return self.num_variables

    def __repr__(self):
        return "{!s}({!s}, {!s}, {!s}, {!r})".format(type(self).__name__,
                                                     self.linear,
                                                     self.quadratic,
                                                     self.offset,
                                                     self.vartype.name)

    # todo: singledisptach in 3.8
    def __add__(self, other: Union[BQM, QM, Bias]) -> Union[BQM, QM]:
        if isinstance(other, BinaryQuadraticModel):
            if other.num_variables and other.vartype != self.vartype:
                # promote to QM
                qm = QuadraticModel.from_bqm(self)
                qm += QuadraticModel.from_bqm(other)
                return qm
            bqm = self.copy()
            bqm.update(other)
            return bqm
        if isinstance(other, QuadraticModel):
            return QuadraticModel.from_bqm(self) + other
        if isinstance(other, Number):
            bqm = self.copy()
            bqm.offset += other
            return bqm

        return NotImplemented

    def __iadd__(self, other: Union[BQM, Bias]) -> BinaryQuadraticModel:
        if isinstance(other, BinaryQuadraticModel):
            if other.num_variables and other.vartype != self.vartype:
                return NotImplemented  # fallback on __add__
            self.update(other)
            return self
        if isinstance(other, Number):
            self.offset += other
            return self
        return NotImplemented

    def __radd__(self, other: Union[QM, Bias]) -> Union[BQM, QM]:
        if isinstance(other, Number):
            return self + other  # communative
        if isinstance(other, QuadraticModel):
            # promote to QM
            qm = other.copy()
            qm += QuadraticModel.from_bqm(self)
            return qm
        return NotImplemented

    def __mul__(self, other: Union[BQM, QM, Bias]) -> Union[BQM, QM]:
        if isinstance(other, BinaryQuadraticModel):
            if not (self.is_linear() and other.is_linear()):
                raise TypeError(
                    "cannot multiply BQMs with interactions")
            elif other.num_variables and other.vartype != self.vartype:
                # promote self
                return QuadraticModel.from_bqm(self) * other

            bqm = self.empty(self.vartype)

            self_offset = self.offset
            other_offset = other.offset

            for u, ubias in self.linear.items():
                for v, vbias in other.linear.items():
                    if u == v:
                        if self.vartype is Vartype.BINARY:
                            bqm.add_linear(u, ubias*vbias)
                        else:
                            bqm.offset += ubias * vbias
                    else:
                        bqm.add_quadratic(u, v, ubias * vbias)

                bqm.add_linear(u, ubias * other_offset)

            for v, bias in other.linear.items():
                bqm.add_linear(v, bias*self_offset)

            bqm.offset += self_offset*other_offset

            return bqm
        if isinstance(other, QuadraticModel):
            # promote to QM
            qm = QuadraticModel.from_bqm(self)
            qm *= other
            return qm
        if isinstance(other, Number):
            bqm = self.copy()
            bqm.scale(other)
            return bqm
        return NotImplemented

    def __imul__(self, other: Bias) -> BinaryQuadraticModel:
        # in-place multiplication is only defined for numbers
        if isinstance(other, Number):
            self.scale(other)
            return self
        return NotImplemented

    def __rmul__(self, other: Union[QM, Bias]) -> Union[BQM, QM]:
        if isinstance(other, Number):
            return self * other  # communative
        if isinstance(other, QuadraticModel):
            # promote self to QM
            qm = QuadraticModel.from_bqm(self)
            qm *= other
            return qm
        return NotImplemented

    def __neg__(self: 'BinaryQuadraticModel') -> 'BinaryQuadraticModel':
        new = self.copy()
        new.scale(-1)
        return new

    def __pos__(self: 'BinaryQuadraticModel') -> 'BinaryQuadraticModel':
        return self

    def __pow__(self, other: int) -> 'BinaryQuadraticModel':
        if isinstance(other, int):
            if other != 2:
                raise ValueError("the only supported power for binary quadratic models is 2")
            if not self.is_linear():
                raise ValueError("only linear models can be squared")
            return self * self
        return NotImplemented

    def __sub__(self, other: Union[BQM, QM, Bias]) -> Union[BQM, QM]:
        if isinstance(other, BinaryQuadraticModel):
            if other.num_variables and other.vartype != self.vartype:
                qm = QuadraticModel.from_bqm(self)
                qm -= QuadraticModel.from_bqm(other)
                return qm
            bqm = self.copy()
            bqm.scale(-1)
            bqm.update(other)
            bqm.scale(-1)
            return bqm
        if isinstance(other, QuadraticModel):
            # promote self to QM
            return QuadraticModel.from_bqm(self) - other
        if isinstance(other, Number):
            bqm = self.copy()
            bqm.offset -= other
            return bqm
        return NotImplemented

    def __isub__(self, other: Union[BQM, Bias]) -> BinaryQuadraticModel:
        if isinstance(other, BinaryQuadraticModel):
            if other.num_variables and other.vartype != self.vartype:
                return NotImplemented  # fallback on __sub__
            self.scale(-1)
            self.update(other)
            self.scale(-1)
            return self
        if isinstance(other, Number):
            self.offset -= other
            return self
        return NotImplemented

    def __rsub__(self, other: Union[QM, Bias]) -> Union[BQM, QM]:
        if isinstance(other, Number):
            bqm = -self  # makes a new one
            bqm.offset += other
            return bqm
        if isinstance(other, QuadraticModel):
            # promote to QM
            return other - QuadraticModel.from_bqm(self)
        return NotImplemented

    def __truediv__(self, other: Bias) -> BinaryQuadraticModel:
        return self * (1 / other)

    def __itruediv__(self, other: Bias) -> BinaryQuadraticModel:
        self *= (1 / other)
        return self

    def __eq__(self, other):
        if isinstance(other, Number):
            return Eq(self, other)
        # support equality for backwards compatibility
        return self.is_equal(other)

    def __ge__(self, other: Bias):
        if isinstance(other, Number):
            return Ge(self, other)
        return NotImplemented

    def __le__(self, other: Bias):
        if isinstance(other, Number):
            return Le(self, other)
        return NotImplemented

    def __ne__(self, other):
        return not self.is_equal(other)

    @property
    def binary(self) -> 'BinaryQuadraticModel':
        r"""Binary-valued version of the binary quadratic model.

        If the binary quadratic model is binary-valued, this references itself,
        otherwise it references a view.

        Examples:
            This example uses a binary-valued Boolean XOR gate,
            :func:`dimod.generators.xor_gate`, to ensure that an unbiased
            eleven-spin ferromagnetic chain, which has two best solutions of
            all-up and all-down spins, selects the all-down solution by requiring
            that :math:`s_5 = s_0 \oplus s_10` (the only assignment that produces
            a ferromagnetic chain is :math:`-1 = -1 \oplus -1`).

            >>> import dimod
            >>> import numpy as np
            >>> s = dimod.SpinArray(range(11))
            >>> bqm = - dimod.quicksum(s[1:11] * s[0:10])   # ferromagnetic chain
            >>> bqm_xor = dimod.generators.xor_gate(0, 10, 5, "aux")
            >>> bqm.spin.update(bqm_xor)   # xor left & right spins to middle spin
            >>> bqm.vartype is dimod.Vartype.SPIN
            True
            >>> sampleset = dimod.drop_variables(dimod.ExactSolver().sample(bqm), "aux")
            >>> set(sampleset.first.sample.values()) == {-1}
            True
            """
        if self.vartype is Vartype.BINARY:
            return self

        try:
            bqm = self._binary
        except AttributeError:
            pass
        else:
            if bqm.vartype is Vartype.BINARY:
                return bqm

        bqm = type(self).__new__(type(self))
        bqm.data = VartypeView(self.data, Vartype.BINARY)

        bqm._spin = self

        self._binary: BinaryQuadraticModel = bqm
        return bqm

    @property
    def dtype(self) -> np.dtype:
        """Data-type of the model's biases."""
        return self.data.dtype

    @property
    def offset(self) -> np.number:
        """Constant energy offset associated with the model."""
        return self.data.offset

    @offset.setter
    def offset(self, offset):
        if isinstance(offset, np.ndarray):
            offset = offset.reshape(tuple())  # try to just flatten into a scalar
        self.data.offset = offset

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
    def shape(self) -> Tuple[int, int]:
        """A 2-tuple of :attr:`num_variables` and :attr:`num_interactions`."""
        return self.num_variables, self.num_interactions

    @property
    def spin(self) -> 'BinaryQuadraticModel':
        """Spin-valued version of the binary quadratic model.

        If the binary quadratic model is spin-valued, this references itself,
        otherwise it references a view.

        Examples:
            This example exploits the simplicity of the Ising representation of
            a Boolean NOT gate, :math:`s_1s_2`, relative to its QUBO counterpart,
            :math:`2x_1x_2−x_1−x_2+1`, to change a QUBO representation of an AND
            gate, derived in the :ref:`AND example <qpu_example_and>`, to
            represent a NAND gate.

            >>> Q = [[0, 1, -2], [0, 0, -2], [0, 0, 3]]
            >>> x = dimod.BinaryArray(["in1", "in2", "out_and"])
            >>> bqm = x.dot(Q).dot(x)               # bqm represents an AND gate
            >>> bqm.spin.add_quadratic("out_and", "out_nand", 1)    # adds a NOT
            >>> bqm.vartype is dimod.Vartype.BINARY
            True
            >>> print(dimod.ExactSolver().sample(bqm).lowest())
              in1 in2 out_and out_nand energy num_oc.
            0   1   1       1        0   -1.0       1
            1   0   1       0        1   -1.0       1
            2   1   0       0        1   -1.0       1
            3   0   0       0        1   -1.0       1
            ['BINARY', 4 rows, 4 samples, 4 variables]
        """
        if self.vartype is Vartype.SPIN:
            return self

        try:
            bqm = self._spin
        except AttributeError:
            pass
        else:
            if bqm.vartype is Vartype.SPIN:
                return bqm

        bqm = type(self).__new__(type(self))
        bqm.data = VartypeView(self.data, Vartype.SPIN)

        bqm._binary = self

        self._spin: BinaryQuadraticModel = bqm
        return bqm

    @property
    def variables(self) -> Variables:
        """The variables of the binary quadratic model."""
        return self.data.variables

    @property
    def vartype(self) -> Vartype:
        """The model's variable type.

        One of :class:`~dimod.Vartype.SPIN` or :class:`~dimod.Vartype.BINARY`.
        """
        return self.data.vartype()

    @classmethod
    def shapeable(cls) -> bool:
        """Returns True if the binary quadratic model is shapeable.

        .. deprecated:: 0.10.0

            All BQMs are shapeable. This method will be removed in dimod
            0.12.0.

        """
        name = cls.__name__
        warnings.warn(
            f"{name}.shapeable() is deprecated since dimod 0.10.0 and will be "
            "removed in 0.12.0. All BQMs are shapeable.",
            DeprecationWarning,
            stacklevel=2)
        return True

    @forwarding_method
    def add_linear(self, v: Variable, bias: Bias):
        """Add a linear term.

        Args:
            v: Variable label.
            bias: Linear bias for the variable.
        """
        return self.data.add_linear

    def add_linear_equality_constraint(
            self, terms: Iterable[Tuple[Variable, Bias]],
            lagrange_multiplier: Bias, constant: Bias):
        r"""Add a linear constraint as a quadratic objective.

        Adds a linear constraint of the form
        :math:`\\sum_{i} a_{i} x_{i} + C = 0`
        to the binary quadratic model as a quadratic objective.

        Args:
            terms:
                Values of the :math:`\sum_{i} a_{i} x_{i}` term as an
                :math:`i`--length iterable of 2-tuples, ``(variable, bias)``, with
                each tuple constituting a term in the summation.
            lagrange_multiplier:
                Weight or penalty strength. The linear constraint is multiplied
                by this value (which does not appear explicitly in the above
                equation) when added to the binary quadratic model.
            constant:
                Value of the constant term, :math:`C`, of the linear constraint.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel("BINARY")
            >>> bqm.add_linear_equality_constraint([("x1", 5), ("x2", -2)], 10, -3)
            >>> print(bqm)
            BinaryQuadraticModel({'x1': -50.0, 'x2': 160.0}, {('x2', 'x1'): -200.0}, 90.0, 'BINARY')
        """
        try:
            self.data.add_linear_equality_constraint(
                terms, lagrange_multiplier, constant)
            return
        except NotImplementedError:
            pass

        for pair in itertools.combinations_with_replacement(terms, 2):
            (u, ubias), (v, vbias) = pair

            if u == v:
                if self.vartype is Vartype.SPIN:
                    self.add_linear(
                        u, 2 * lagrange_multiplier * ubias * constant)
                    self.offset += lagrange_multiplier * ubias * vbias
                else:
                    self.add_linear(
                        u, lagrange_multiplier * ubias * (2*constant + vbias))
            else:
                self.add_quadratic(
                    u, v, 2 * lagrange_multiplier * ubias * vbias)
        self.offset += lagrange_multiplier * constant * constant

    def add_linear_inequality_constraint(
                self, terms: Iterable[Tuple[Variable, int]],
                lagrange_multiplier: Bias,
                label: str,
                constant: int = 0,
                lb: int = np.iinfo(np.int64).min,
                ub: int = 0,
                cross_zero: bool = False,
                penalization_method: Literal["slack", "unbalanced"] = "slack",
        ) -> Iterable[Tuple[Variable, int]]:
        r"""Add a linear inequality constraint as a quadratic objective.

        The linear inequality constraint is of the form:
        :math:`lb <= \sum_{i,k} a_{i,k} x_{i,k} + constant <= ub`.

        For constraints with fractional coefficients, multiply both sides of the
        inequality by an appropriate factor of ten to attain or approximate
        integer coefficients.

        Args:
            terms:
                Values of the :math:`\sum_{i} a_{i} x_{i}` term as an
                :math:`i`--length iterable of 2-tuples, ``(variable, bias)``, with
                each tuple constituting a term in the summation.
            lagrange_multiplier:
                Weight or penalty strength. The linear constraint is multiplied
                by this value (which does not appear explicitly in the above
                equation) when added to the binary quadratic model.
            label:
                Prefix for labels of any slack variables used in the added
                objective.
            constant:
                Value of the constant term of the linear constraint.
            lb:
                Lower bound for the constraint.
            ub:
                Upper bound for the constraint.
            cross_zero:
                When True, adds zero to the domain of constraint.
            penalization_method:
                Whether to use slack variables or the unbalanced penalization method [1].
                ("slack", "unbalanced")
                [1] https://arxiv.org/abs/2211.13914

        Returns:
            slack_terms:  Values of :math:`\sum_{i} b_{i} slack_{i}` as an
            :math:`i`--length iterable of 2-tuples, ``(slack variable, bias)``,
            with each tuple constituting a term in the summation.
        """

        if isinstance(terms, Iterator):
            terms = list(terms)

        if int(constant) != constant or int(lb) != lb or int(ub) != ub or any(
                int(bias) != bias for _, bias in terms):
            warnings.warn("For constraints with fractional coefficients, "
                          "multiply both sides of the inequality by an "
                          "appropriate factor of ten to attain or "
                          "approximate integer coefficients. ")

        terms_upper_bound = sum(v for _, v in terms if v > 0)
        terms_lower_bound = sum(v for _, v in terms if v < 0)
        ub_c = min(terms_upper_bound, ub - constant)
        lb_c = max(terms_lower_bound, lb - constant)

        if terms_upper_bound <= ub_c and terms_lower_bound >= lb_c:
            warnings.warn(
                f'Did not add constraint {label}.'
                ' This constraint is feasible'
                ' with any value for state variables.')
            return []

        if ub_c < lb_c:
            raise ValueError(
                f'The given constraint ({label}) is infeasible with any value'
                ' for state variables.')
        if penalization_method == "slack":
            slack_upper_bound = int(ub_c - lb_c)
            if slack_upper_bound == 0:
                self.add_linear_equality_constraint(terms, lagrange_multiplier, -ub_c)
                return []
            else:
                slack_terms = []
                zero_constraint = False
                if cross_zero:
                    if lb_c > 0 or ub_c < 0:
                        if ub_c-slack_upper_bound > 0:
                            zero_constraint = True
    
                num_slack = int(np.floor(np.log2(slack_upper_bound)))
                slack_coefficients = [2 ** j for j in range(num_slack)]
                if slack_upper_bound - 2 ** num_slack >= 0:
                    slack_coefficients.append(slack_upper_bound - 2 ** num_slack + 1)
    
                for j, s in enumerate(slack_coefficients):
                    sv = self.add_variable(f'slack_{label}_{j}')
                    slack_terms.append((sv, s))
    
                if zero_constraint:
                    sv = self.add_variable(f'slack_{label}_{num_slack + 1}')
                    slack_terms.append((sv, ub_c - slack_upper_bound))
    
            self.add_linear_equality_constraint(terms + slack_terms,
                                                lagrange_multiplier, -ub_c)
            return slack_terms

        elif penalization_method == "unbalanced":
            if not isinstance(lagrange_multiplier, Iterable):
                raise TypeError('A list with two lagrange_multiplier are needed'
                                ' for the unbalanced penalization method.')

            for v, bias in terms:
                self.add_linear(v, lagrange_multiplier[0] * bias)
            self.offset += -ub_c
            self.add_linear_equality_constraint(terms, lagrange_multiplier[1], -ub_c)
            
            return []
        else:
            raise ValueError(f"The method {penalization_method} is not a valid method."
                             ' Choose between ["slack", "unbalanced"]')

    def add_linear_from_array(self, linear: Sequence):
        """Add linear biases from an array-like to a binary quadratic model.

        Args:
            linear:
                Linear biases as a one-dimensional `array_like`_.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel("BINARY")
            >>> bqm.add_linear_from_array([0.5, 0.2, -0.3, 0.25])
            >>> print(bqm.linear)
            {0: 0.5, 1: 0.2, 2: -0.3, 3: 0.25}

    .. _array_like: https://numpy.org/doc/stable/user/basics.creation.html

        """
        ldata = np.asarray(linear)

        # cython has trouble with readonly buffers as of 0.29.22, in the
        # future we can remove this
        if not ldata.flags.writeable:
            ldata = np.array(ldata, copy=True)

        self.data.add_linear_from_array(np.asarray(ldata))

    def add_offset(self, bias):
        """Add offset to to the model.

        .. deprecated:: 0.10.0

            ``BinaryQuadraticModel.add_offset(bias)`` will be removed in dimod 0.12.0.
            Use ``bqm.offset += bias`` instead.

        """
        name = type(self).__name__
        warnings.warn(
            f"{name}.add_offset(bias) is deprecated since dimod 0.10.0 and will be "
            "removed in 0.12.0. Please use bqm.offset += bias.",
            DeprecationWarning,
            stacklevel=2)
        self.offset += bias

    @forwarding_method
    def add_quadratic(self, u: Variable, v: Variable, bias: Bias):
        """Add a quadratic bias between two variables.

        Args:
            u: Variable in the binary quadratic model.
            v: Variable in the binary quadratic model.
            bias: Quadratic bias for the interaction.

        Raises:
            ValueError: If any self-loops are given. E.g. ``(u, u, bias)`` is not
                a valid triplet.
        """
        return self.data.add_quadratic

    def add_interaction(self, *args, **kwargs):
        """Alias for :meth:`.add_quadratic`."""
        return self.add_quadratic(*args, **kwargs)

    def add_quadratic_from(self, quadratic: Union[Mapping, Iterable]):
        """Add quadratic biases to the binary quadratic model.

        Args:
            quadratic:
                Interactions and their associated quadratic biases, as either a
                dict of form ``{(u, v): bias, ...}`` or an iterable of
                ``(u, v, bias)`` triplets, where ``u`` and ``v`` are variables in
                the model and ``bias`` is the associated quadratic bias.
                If a variable is not present in the model, it is added.
                If the interaction already exists, the bias is added.

        Raises:
            ValueError:
                If any self-loops are given. E.g. ``(u, u, bias)`` is not a valid
                triplet.
        """
        add_quadratic = self.data.add_quadratic

        if isinstance(quadratic, abc.Mapping):
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
            quadratic: Quadratic biases as a square 2d `array_like`_.

        Raises:
            ValueError:
                If any self-loops are given; i.e., the array contains a non-zero
                value on its diagonal, which would set a bias for interaction
                ``(u, u)``.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel("BINARY")
            >>> bqm.add_quadratic_from_dense([[0, -0.4, 0.2],[0, 0, 0], [0, 0, 0]])
            >>> print(bqm)
            BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0}, {(1, 0): -0.4, (2, 0): 0.2}, 0.0, 'BINARY')

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
        """Add a variable to a binary quadratic model.

        Args:
            v: Variable label. If not provided, the next integer label
                is used.
            bias: Linear bias for the added variable.
        """
        return self.data.add_variable

    def change_vartype(self, vartype: Vartype,
                       inplace: bool = True) -> 'BinaryQuadraticModel':
        """Return a binary quadratic model with the specified vartype.

        Args:
            vartype:
                Variable type for the changed model. Accepted input values:

                * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            inplace:
                If True, the binary quadratic model is updated in-place;
                otherwise, a new binary quadratic model is returned.

        Returns:
            A binary quadratic model with the specified vartype.
        """
        if not inplace:
            return self.copy().change_vartype(vartype, inplace=True)
        self.data.change_vartype(vartype)
        return self

    def clear(self) -> None:
        """Remove the offset and all variables and interactions from the model."""
        self.data.clear()

    def contract_variables(self, u: Variable, v: Variable):
        """Enforce u, v being the same variable in a binary quadratic model.

        The resulting variable is labeled ``u``. Values of interactions between
        ``v`` and variables that ``u`` interacts with are added to the
        corresponding interactions of ``u``.

        Args:
            u: Variable in the binary quadratic model.
            v: Variable in the binary quadratic model.

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

        try:
            self.remove_interaction(u, v)
        except ValueError:
            pass

        # add all of v's interactions to u's
        for w, b in self.iter_neighborhood(v):
            self.add_quadratic(u, w, b)

        # finally remove v
        self.remove_variable(v)

    def copy(self, deep=False):
        """Return a copy.

        Args:
            deep: If True, returns a deep copy.
        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    @forwarding_method
    def degree(self, v: Variable) -> int:
        """Return the degree of a variable.

        The degree is the number of interactions that contain ``v``.

        Args:
            v: Variable in the binary quadratic model.
        """
        return self.data.degree

    def degrees(self, array: bool = False, dtype: DTypeLike = int
                ) -> Union[np.ndarray, Mapping[Variable, int]]:
        """Return the degrees of a binary quadratic model's variables.

        Args:
            array:
                If True, returns a :obj:`numpy.ndarray`; otherwise returns a dict.
            dtype:
                The data type of the returned degrees. Applies only if
                ``array==True``.

        Returns:
            Degrees of all variables.

        """
        if array:
            return np.fromiter(map(self.degree, self.variables),
                               count=len(self), dtype=dtype)
        return {v: self.degree(v) for v in self.variables}

    @classmethod
    def empty(cls, vartype):
        """Create a new binary quadratic model with no variables and no offset.
        """
        return cls(vartype)

    def energies(self, samples_like, dtype: Optional[DTypeLike] = None) -> np.ndarray:
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
            >>> bqm = dimod.BinaryQuadraticModel({}, {("x0", "x1"): -1}, "BINARY")
            >>> bqm.energies([{"x0": 1, "x1": 1},{"x0": 0, "x1": 1}])
            array([-1.,  0.])

            See also the example under :meth:`energy`.

        .. _`array_like`:  https://numpy.org/doc/stable/user/basics.creation.html

        """
        return self.data.energies(samples_like, dtype=dtype)

    def energy(self, sample, dtype: Optional[DTypeLike] = None) -> Bias:
        """Determine the energy of the given sample.

        Args:
            sample (samples_like):
                Raw sample. `samples_like` is an extension of
                NumPy's `array_like`_ structure. See :func:`.as_samples`.

            dtype:
                Desired NumPy data type for the energy.
                Defaults to :class:`~numpy.float64`.

        Returns:
            The energy.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({}, {("x0", "x1"): -1}, "BINARY")
            >>> bqm.energy({"x0": 1, "x1": 1})
            -1.0

            See also the example under :meth:`energies`.

        .. _`array_like`:  https://numpy.org/doc/stable/user/basics.creation.html

        """
        energies = self.energies(sample, dtype=dtype)

        if not len(energies):  # the empty case, happens with []
            return self.dtype.type(0)

        energy, = energies
        return energy

    def maximum_energy_delta(self) -> Bias:
        """Compute a conservative bound on the maximum change in energy that can
        result from flipping a single variable in a binary quadratic model.

        The bound is useful as a starting point for determining the values of
        `penalty parameters`_ in a :ref:`penalty model <concept_penalty>`.

        Returns:
            Bound on change in energy.

        Examples:
            >>> Q = {(0, 0): -1, (0, 1): 1, (1, 2): -4.5}
            >>> bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
            >>> bqm.maximum_energy_delta()
            5.5

        .. _`penalty parameters`: https://en.wikipedia.org/wiki/Penalty_method
        """
        if not self.num_variables:
            return 0

        if self.vartype is Vartype.SPIN:
            scale = 2
        elif self.vartype is Vartype.BINARY:
            scale = 1
        else:
            raise RuntimeError("unexpected vartype")

        return max(abs(self.get_linear(v))
                   + sum(abs(bias) for u, bias in self.iter_neighborhood(v))
                   for v in self.variables) * scale

    def flip_variable(self, v: Variable):
        """Flip the specified variable in a binary quadratic model."""
        if self.vartype is Vartype.SPIN:
            for u, bias in self.iter_neighborhood(v):
                self.set_quadratic(u, v, -1*bias)
            self.set_linear(v, -1*self.get_linear(v))
        elif self.vartype is Vartype.BINARY:
            for u, bias in self.iter_neighborhood(v):
                self.set_quadratic(u, v, -1*bias)
                self.add_linear(u, bias)
            self.offset += self.get_linear(v)
            self.set_linear(v, -1*self.get_linear(v))
        else:
            raise RuntimeError("unexpected vartype")

    @classmethod
    def from_coo(cls, obj, vartype=None):
        """Deserialize a BQM from a Coordinate format encoding.

        Args:
            obj: (str/file):
                Either a string or a `.read()`-supporting file object that
                represents linear and quadratic biases for a binary quadratic
                model.

        Note:
            This method is deprecated. Use
            :func:`~dimod.serialization.coo.load` or
            :func:`~dimod.serialization.coo.loads` instead.

        .. deprecated:: 0.10.0

            ``BinaryQuadraticModel.from_coo()`` will be removed in dimod
            0.12.0. Use :func:`~dimod.serialization.coo.load()` or
            :func:`~dimod.serialization.coo.loads()` instead.

        """
        warnings.warn('BinaryQuadraticModel.from_coo() is deprecated since '
                      'dimod 0.10.0 and will be removed in 0.12.0. '
                      'Use dimod.serialization.coo.load() or '
                      'dimod.serialization.coo.loads() instead.',
                      DeprecationWarning, stacklevel=2)

        import dimod.serialization.coo as coo

        if isinstance(obj, str):
            return coo.loads(obj, cls=cls, vartype=vartype)

        return coo.load(obj, cls=cls, vartype=vartype)

    @classmethod
    def from_file(cls, fp: Union[BinaryIO, typing.Union[bytes, bytearray]]):
        """Construct a binary quadratic model from a file-like object.

        The inverse of :meth:`~BinaryQuadraticModel.to_file`.

        """
        if isinstance(fp, (bytes, bytearray, memoryview)):
            file_like: BinaryIO = _BytesIO(fp)  # type: ignore[assignment]
        else:
            file_like = fp

        header_info = read_header(file_like, BQM_MAGIC_PREFIX)
        version = header_info.version
        data = header_info.data

        if version >= (3, 0):
            raise ValueError("cannot load a BQM serialized with version "
                             f"{version!r}, try upgrading your dimod version")

        num_variables, num_interactions = data['shape']

        dtype = np.dtype(data['dtype'])
        itype = np.dtype(data['itype'])  # index of the variable
        ntype = np.dtype(data['ntype'])  # index of the neighborhood

        bqm = cls(data['vartype'], dtype=data['dtype'])

        # offset
        offset_bytes = file_like.read(dtype.itemsize)
        if len(offset_bytes) < dtype.itemsize:
            raise ValueError("given file is missing offset biases")
        offset_array = np.frombuffer(offset_bytes, dtype=dtype)
        bqm.data.add_offset_from_array(offset_array)

        if num_variables:
            linear_dtype = np.dtype([('nidx', ntype), ('bias', dtype)],
                                    align=False)
            quadratic_dtype = np.dtype([('outvar', itype), ('bias', dtype)],
                                       align=False)
            # linear
            ldata = np.frombuffer(
                file_like.read(num_variables*linear_dtype.itemsize),
                dtype=linear_dtype)
            if ldata.shape[0] != num_variables:
                raise ValueError("given file is missing linear data")

            bqm.data.add_linear_from_array(ldata['bias'])

            # quadratic
            for v in range(num_variables):
                if v < num_variables - 1:
                    degree = int(ldata['nidx'][v + 1] - ldata['nidx'][v])
                else:
                    degree = int(2*num_interactions - ldata['nidx'][v])

                if not degree:
                    # not needed, but helps with performance
                    continue

                qdata = np.frombuffer(
                    file_like.read(degree*quadratic_dtype.itemsize),
                    dtype=quadratic_dtype)
                if qdata.shape[0] != degree:
                    raise ValueError("given file is missing quadratic data")

                # we only want the lower triangle, so that we're always
                # appending variables - speeds up construction
                vi = np.searchsorted(qdata['outvar'], v, side='right')

                # need these to be C-ordered so we can pass them into function
                irow = np.ascontiguousarray(qdata['outvar'][:vi])
                icol = np.full(vi, v, dtype=itype)
                biases = np.ascontiguousarray(qdata['bias'][:vi])

                bqm.data.add_quadratic_from_arrays(irow, icol, biases)

        # labels
        if data['variables']:
            if version < (2, 0):
                # the variables are in the header
                bqm.relabel_variables(dict(enumerate(
                    iter_deserialize_variables(data['variables']))))
            else:
                bqm.relabel_variables(dict(enumerate(
                    VariablesSection.load(file_like))))

        return bqm

    @classmethod
    def from_ising(cls, h: Union[Mapping, Sequence],
                   J: Mapping,
                   offset: float = 0):
        """Create a binary quadratic model from an Ising problem.

        Args:
            h: Linear biases of the Ising problem, as either a dict of the form
                ``{v: bias, ...}``, where ``v`` is a spin-valued variable and
                ``bias`` is its associated bias, or as a list of biases, where
                the indices are the variable labels.

            J: Quadratic biases of the Ising problem.

            offset: Constant offset.

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

                * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

                If not provided, ``G`` must have a vartype attribute. If
                provided, ``vartype`` overrides an existing ``G.vartype``
                attribute.

            node_attribute_name (hashable, optional, default='bias'):
                Attribute name for linear biases. If the node does not have a
                matching attribute, the bias defaults to 0.

            edge_attribute_name (hashable, optional, default='bias'):
                Attribute name for quadratic biases. If the edge does not have a
                matching attribute, the bias defaults to 0.

        Returns:
            Binary quadratic model.

        .. deprecated:: 0.10.0

            ``BinaryQuadraticModel.from_networkx_graph()`` will be removed
            in dimod 0.12.0. Use :func:`~dimod.converters.from_networkx_graph()`
            instead.

        """
        warnings.warn('BinaryQuadraticModel.from_networkx_graph() is deprecated since '
                      'dimod 0.10.0 and will be removed in 0.12.0. '
                      'Use dimod.from_networkx_graph(bqm) instead.',
                      DeprecationWarning, stacklevel=2)
        from dimod.converters import from_networkx_graph  # avoid circular import
        return from_networkx_graph(G, vartype, node_attribute_name,
                                   edge_attribute_name, cls=cls)

    @classmethod
    def from_numpy_matrix(cls, mat, variable_order=None, offset=0,
                          interactions=None):
        """Deprecated.

        .. deprecated:: 0.10.0

            ``BinaryQuadraticModel.from_numpy_matrix()`` will be removed
            in dimod 0.12.0. Use ``BinaryQuadraticModel(mat, 'BINARY')``
            instead.

        """
        warnings.warn('BQM.from_numpy_matrix(M) is deprecated since dimod '
                      '0.10.0 and will be removed in 0.12.0. Use BQM(M, "BINARY") instead.',
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
    def from_numpy_vectors(cls, linear: ArrayLike, quadratic: ArrayLike,
                           offset: float, vartype: Vartype, *,
                           variable_order: Iterable = None,
                           dtype: DTypeLike = np.float64) -> 'BinaryQuadraticModel':
        """Create a binary quadratic model from NumPy vectors.

        Args:
            linear: Linear biases.

            quadratic: Quadratic biases.

            offset: Offset of the binary quadratic model.

            vartype:
                Variable type for the binary quadratic model. Accepted input
                values:

                * :class:`~dimod.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`~dimod.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            variable_order:
                Variable order for the binary quadratic model's labels.

            dtype: Data type for the returned binary quadratic model.

        Returns:
            A binary quadratic model.

        Examples:
            >>> import numpy as np
            >>> linear = np.ones(10)
            >>> quadratic = (np.arange(0, 10), np.arange(1, 11), -np.ones(10))
            >>> bqm = dimod.BQM.from_numpy_vectors(linear, quadratic, 0, "BINARY")
        """
        obj = super().__new__(cls)
        data_cls = cls._DATA_CLASSES[np.dtype(dtype)]
        obj.data = data_cls.from_numpy_vectors(
            linear, quadratic, offset, vartype,
            variable_order=variable_order)
        return obj

    @classmethod
    def from_qubo(cls, Q: Mapping, offset: float = 0):
        """Create a binary quadratic model from a QUBO problem.

        Args:
            Q: Coefficients of a quadratic unconstrained binary optimization
                (QUBO) problem as a dict of form ``{(u, v): bias, ...}``, where
                ``u``, ``v``, are binary-valued variables and ``bias`` is their
                associated coefficient.

            offset (optional, default=0.0):
                Constant offset applied to the model.

        Returns:
            A binary-valued binary quadratic model.

        """
        return cls({}, Q, offset, Vartype.BINARY)

    @classmethod
    def from_serializable(cls, obj: Mapping) -> 'BinaryQuadraticModel':
        """Deserialize a binary quadratic model.

        Args:
            obj: A binary quadratic model serialized by
                :meth:`~.BinaryQuadraticModel.to_serializable`.

        Returns:
            A binary quadratic model.

        Examples:

            Encode and decode using JSON

            >>> import json
            ...
            >>> bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0},
            ...                                  {('a', 'b'): -1.0},
            ...                                  0.0,
            ...                                  dimod.SPIN)
            >>> s = json.dumps(bqm.to_serializable())
            >>> new_bqm = dimod.BinaryQuadraticModel.from_serializable(json.loads(s))

        See also:
            :meth:`~.BinaryQuadraticModel.to_serializable`

            :func:`json.loads`, :func:`json.load` JSON deserialization functions

        """
        version = obj.get("version", {"bqm_schema": "1.0.0"})["bqm_schema"]
        if version < "2.0.0":
            raise ValueError("No longer supported serialization format")
        elif version < "3.0.0" and obj.get("use_bytes", False):
            # from 2.0.0 to 3.0.0 the formatting of the bytes changed
            raise ValueError("No longer supported serialization format")

        variables = [tuple(v) if isinstance(v, list) else v for v in obj["variable_labels"]]

        if obj["use_bytes"]:
            bias_dtype = np.dtype(obj['bias_type'])
            index_dtype = np.dtype(obj['index_type'])

            ldata = np.frombuffer(obj['linear_biases'], dtype=bias_dtype)
            qdata = np.frombuffer(obj['quadratic_biases'], dtype=bias_dtype)
            irow = np.frombuffer(obj['quadratic_head'], dtype=index_dtype)
            icol = np.frombuffer(obj['quadratic_tail'], dtype=index_dtype)
        else:
            bias_dtype = None

            ldata = obj["linear_biases"]
            qdata = obj["quadratic_biases"]
            irow = obj["quadratic_head"]
            icol = obj["quadratic_tail"]

        offset = obj["offset"]
        vartype = obj["variable_type"]

        bqm = cls.from_numpy_vectors(ldata,
                                     (irow, icol, qdata),
                                     offset,
                                     vartype,
                                     variable_order=variables,
                                     dtype=bias_dtype)
        return bqm

    def has_variable(self, v):
        """Deprecated.

        .. deprecated:: 0.10.0

            ``BinaryQuadraticModel.has_variable()`` will be removed in dimod 0.12.0.
            Use ``v in bqm.variables`` instead.

        """
        warnings.warn('bqm.has_variable(v) is deprecated since dimod 0.10.0 and will be removed in 0.12.0. '
                      'Use v in bqm.variables instead.',
                      DeprecationWarning, stacklevel=2)
        return v in self.data.variables

    def is_almost_equal(self, other: Union['BinaryQuadraticModel', QuadraticModel, Bias],
                        places: int = 7) -> bool:
        """Test for near equality to all biases of a given binary quadratic model.

        Args:
            other:
                Binary quadratic model with which to compare biases.
            places:
                Number of decimal places to which the Python :func:`round`
                function calculates approximate equality.

        Examples:
            >>> bqm1 = dimod.BinaryQuadraticModel({0: 0.1234},
            ...                                   {(0, 1): -1.1234},0, "BINARY")
            >>> bqm2 = dimod.BinaryQuadraticModel({0: 0.1232},
            ...                                   {(0, 1): -1.1229},0, "BINARY")
            >>> bqm1.is_almost_equal(bqm2, 4)
            False
            >>> bqm1.is_almost_equal(bqm2, 3)
            True
        """
        if isinstance(other, Number):
            return not (self.num_variables or round(self.offset - other, places))

        def eq(a, b):
            return not round(a - b, places)

        try:
            if isinstance(other, QuadraticModel):
                vartype_eq = all(other.vartype(v) is self.vartype for v in other.variables)
            else:
                vartype_eq = self.vartype == other.vartype

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

    def is_equal(self, other: Union['BinaryQuadraticModel', QuadraticModel, Bias]) -> bool:
        """Return True if the given model has the same variables, vartypes and biases.

        Args:
            other: Binary quadratic model to compare against.
        """
        if isinstance(other, Number):
            return not self.num_variables and bool(self.offset == other)
        # todo: performance
        try:
            if callable(other.vartype):
                vartype_eq = all(other.vartype(v) is self.vartype for v in other.variables)
            else:
                vartype_eq = self.vartype == other.vartype

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
    def iter_neighborhood(self, v: Variable) -> Iterator[Tuple[Variable, Bias]]:
        """Iterate over the neighbors and quadratic biases of a variable.

        Args:
            v: Variable in the binary quadratic model.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({},
            ...                 {(0, 1): -1, (1, 2): -2, (2, 3): -3},
            ...                 "BINARY")
            >>> list(bqm.iter_neighborhood(1))
            [(0, -1.0), (2, -2.0)]
        """
        return self.data.iter_neighborhood

    def iter_neighbors(self, u):
        """Deprecated

        .. deprecated:: 0.10.0

            ``BinaryQuadraticModel.iter_neighbors(u)`` will be removed in dimod 0.12.0.
            Use ``(v for v, _ in bqm.iter_neighborhood(u))`` instead.

        """
        warnings.warn('bqm.iter_neighbors(u) is deprecated since dimod 0.10.0 '
                      'and will be removed in 0.12.0.'
                      ' Use (v for v, _ in bqm.iter_neighborhood(u)) instead.',
                      DeprecationWarning, stacklevel=2)
        for v, _ in self.iter_neighborhood(u):
            yield v

    def iter_quadratic(self, variables=None) -> Iterator[Tuple[Variable, Variable, Bias]]:
        """Iterate over the quadratic biases.

        .. deprecated:: 0.10.0

            Argument ``variables`` will be removed in dimod 0.12.0.
            Use ``bqm.iter_neighborhood(v)`` instead.

        """
        if variables is None:
            yield from self.data.iter_quadratic()
        else:
            warnings.warn('passing variables to bqm.iter_quadratic() is '
                          'deprecated since dimod 0.10.0 and will be removed in 0.12.0. Use '
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

        .. deprecated:: 0.10.0

            ``BinaryQuadraticModel.iter_variables()`` will be removed in dimod 0.12.0.
            Use ``iter(bqm.variables)`` instead.

        """
        warnings.warn('bqm.iter_variables() is deprecated since dimod 0.10.0 '
                      'and will be removed in 0.12.0. '
                      'Use iter(bqm.variables) instead.', stacklevel=2)
        return iter(self.variables)

    @forwarding_method
    def get_linear(self, v: Variable) -> Bias:
        """Get the linear bias of a variable.

        Args:
            v: Variable in the binary quadratic model.
        """
        return self.data.get_linear

    @forwarding_method
    def get_quadratic(self, u: Variable, v: Variable,
                      default: Optional[Bias] = None):
        """Get the quadratic bias of a pair of variables.

        Args:
            u:  Variable in the binary quadratic model.
            v:  Variable in the binary quadratic model.
        """
        return self.data.get_quadratic

    def nbytes(self, capacity: bool = False) -> int:
        """Get the total bytes consumed by the biases and indices.

        Does not include the memory consumed by non-element attributes of
        the binary quadratic model object.
        Also does not include the memory consumed by the variable labels.

        Args:
            capacity: If ``capacity`` is true, also include the ``std::vector::capacity``
                of the underlying vectors in the calculation.

        Returns:
            The number of bytes.

        Raises:
            TypeError: If :attr:`.dtype` is :class:`object`.

        """
        return self.data.nbytes(capacity)

    def normalize(self, bias_range: Union[float, Tuple[float, float]] = 1,
                  quadratic_range: Union[float, Tuple[float, float], None] = None,
                  ignored_variables: Optional[Iterable[Variable]] = None,
                  ignored_interactions: Optional[Iterable[Tuple[Variable, Variable]]] = None,
                  ignore_offset: bool = False):
        """Normalize the biases of a binary quadratic model.

        Normalizes the biases to fall in the provided range(s), and adjusts the
        offset appropriately.

        Args:
            bias_range:
                Value/range that the biases of the BQM are scaled to fit
                within. If ``quadratic_range`` is provided, this range is
                used to fit the linear biases.

            quadratic_range:
                Value/range that quadratic biases of the BQM are scaled to fit
                within.

            ignored_variables:
                Biases associated with these variables are not scaled.

            ignored_interactions:
                Biases associated with these interactions, formatted as an
                iterable of 2-tuples, are not scaled.

            ignore_offset:
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
        elif not isinstance(ignored_variables, abc.Container):
            ignored_variables = set(ignored_variables)

        if ignored_interactions is None:
            ignored_interactions = set()
        elif not isinstance(ignored_interactions, abc.Container):
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

    @forwarding_method
    def reduce_linear(self, function: Callable,
                      initializer: Optional[Bias] = None) -> Any:
        """Apply function of two arguments cumulatively to the linear biases.

        See :func:`functools.reduce` for information on reducing an iterable to
        a single value.

        Args:
            function: Function of two arguments to apply cumulatively to linear
                biases.
            initializer: A value to precede the linear biases.

        Examples:
            >>> from operator import add
            >>> bqm = dimod.BinaryQuadraticModel({0: 1, 1: 2, 2: 3},
            ...                                  {(0, 1): 5},
            ...                                  0, "BINARY")
            >>> bqm.reduce_linear(add)
            6.0
        """
        return self.data.reduce_linear

    @forwarding_method
    def reduce_neighborhood(self, v: Variable, function: Callable,
                            initializer: Optional[Bias] = None) -> Any:
        """Apply function of two arguments cumulatively to the quadratic biases
        associated with a single variable.

        See :func:`functools.reduce` for information on reducing an iterable to
        a single value.

        Args:
            v:
                Variable in the binary quadratic model.
            function: Function of two arguments to apply cumulatively to quadratic
                biases.
            initializer:
                A value to precede the linear biases.

        Examples:
            >>> from operator import add
            >>> bqm = dimod.BinaryQuadraticModel({0: 10},
            ...                                  {(0, 1): 1, (0, 2): 2, (1, 2): 5},
            ...                                  0, "BINARY")
            >>> bqm.reduce_neighborhood(0, add, 0.25)
            3.25
        """
        return self.data.reduce_neighborhood

    @forwarding_method
    def reduce_quadratic(self, function: Callable,
                         initializer: Optional[Bias] = None) -> Any:
        """Apply function of two arguments cumulatively to the quadratic
        biases.

        See :func:`functools.reduce` for information on reducing an iterable to
        a single value.

        Args:
            function: Function of two arguments to apply cumulatively to quadratic
                biases.
            initializer:
                A value to precede the linear biases.

        Examples:
            >>> from operator import add
            >>> bqm = dimod.BinaryQuadraticModel({0: 10},
            ...                                  {(0, 1): 1, (0, 2): 2, (1, 2): 5},
            ...                                  0, "BINARY")
            >>> bqm.reduce_quadratic(add, 0.25)
            8.25
        """
        return self.data.reduce_quadratic

    def relabel_variables(self, mapping, inplace=True):
        """Relabel the variables of a binary quadratic model.

        Args:
            mapping: Mapping between current and new labels.
            inplace: If True, the binary quadratic model is updated in-place;
                otherwise, a new binary quadratic model is returned.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({0: 1, 1: 2, 2: 3},
            ...                                  {(0, 1): 5},
            ...                                  0, "BINARY")
            >>> bqm_new = bqm.relabel_variables({0: "a", 1: "b", 2: "c"},
            ...                                 inplace=False)
            >>> bqm_new.variables
            Variables(['a', 'b', 'c'])
        """
        if not inplace:
            return self.copy().relabel_variables(mapping, inplace=True)

        self.data.relabel_variables(mapping)
        return self

    def relabel_variables_as_integers(self, inplace=True):
        """Relabel to consecutive integers the variables of a binary quadratic
        model.

        Args:
            inplace: If True, the binary quadratic model is updated in-place;
                otherwise, a new binary quadratic model is returned.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({"a": 1, "b": 2},{"ab": -1}, 0, "BINARY")
            >>> _, mapping = bqm.relabel_variables_as_integers()
            >>> print(bqm.quadratic)
            {(1, 0): -1.0}
        """
        if not inplace:
            return self.copy().relabel_variables_as_integers(inplace=True)

        mapping = self.data.relabel_variables_as_integers()
        return self, mapping

    @forwarding_method
    def remove_interaction(self, u: Variable, v: Variable):
        """Remove the interaction between a pair of variables.

        Args:
            u: Variable in the binary quadratic model.
            v: Variable in the binary quadratic model.
        """
        return self.data.remove_interaction

    def remove_interactions_from(self, interactions: Iterable):
        """Remove the given interactions from the binary quadratic model.

        Args:
            interactions: Interactions of the binary quadratic model as an
                iterable of 2-tuples.
        """
        for u, v in interactions:
            self.remove_interaction(u, v)

    @forwarding_method
    def remove_variable(self, v: Optional[Variable] = None) -> Variable:
        """Remove the specified variable from a binary quadratic model.

        Args:
            v: Variable of the binary quadratic model.
        """
        return self.data.remove_variable

    def remove_variables_from(self, variables: Iterable[Variable]):
        """Remove the given variables from the binary quadratic model.

        Args:
            variables: Variables of the binary quadratic model.
        """
        for v in variables:
            self.remove_variable(v)

    @forwarding_method
    def resize(self, n: int):
        """Resize a binary quadratic model to the specified number of variables.

        Args:
            n: Number of variables in the resized model. If ``n`` is smaller than
                the current number of variables, variables are removed; if larger,
                variables are added.

        Returns:
            int: Difference between the number of variables in the resized model
                and previous model.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({"a": 1, "b": 2}, {}, 0, "BINARY")
            >>> bqm.resize(4)
            2
            >>> bqm.variables
            Variables(['a', 'b', 2, 3])
            >>> bqm.resize(3)
            -1
            >>> bqm.variables
            Variables(['a', 'b', 2])

        """
        return self.data.resize

    def scale(self, scalar, ignored_variables=None, ignored_interactions=None,
              ignore_offset=False):
        """Multiply all biases by the specified scalar.

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
        if ignored_variables is None and ignored_interactions is None \
            and ignore_offset is False:
            try:
                self.data.scale(scalar)
                return
            except AttributeError:
                pass

        if ignored_variables is None:
            ignored_variables = set()
        elif not isinstance(ignored_variables, abc.Container):
            ignored_variables = set(ignored_variables)

        if ignored_interactions is None:
            ignored_interactions = set()
        elif not isinstance(ignored_interactions, abc.Container):
            ignored_interactions = set(ignored_interactions)

        for v in self.variables:
            if v in ignored_variables:
                continue
            self.set_linear(v, scalar*self.get_linear(v))

        for u, v, bias in self.iter_quadratic():
            if (u, v) in ignored_interactions or (v, u) in ignored_interactions:
                continue
            self.set_quadratic(u, v, scalar*self.get_quadratic(u, v))

        if not ignore_offset:
            self.offset *= scalar

    @forwarding_method
    def set_linear(self, v: Variable, bias: Bias):
        """Set the linear bias of of a variable.

        Args:
            v: Variable in the binary quadratic model.

        Raises:
            TypeError: If ``v`` is not hashable.

        """
        return self.data.set_linear

    @forwarding_method
    def set_quadratic(self, u: Variable, v: Variable, bias: Bias):
        """Set the quadratic bias of interaction ``(u, v)``.

        Args:
            u: Variable in the binary quadratic model.
            v: Variable in the binary quadratic model.

        Raises:
            TypeError: If ``u`` or ``v`` is not hashable.

        """
        return self.data.set_quadratic

    def to_coo(self, fp=None, vartype_header: bool = False):
        """Serialize the binary quadratic model to a COOrdinate format encoding.

        If ``fp`` is provided, the serialized BQM is written to a file;
        otherwise it is returned as a string.

        Note:
            This method is deprecated, use
            :func:`~dimod.serialization.coo.dump` or
            :func:`~dimod.serialization.coo.dumps` instead.
        """
        warnings.warn('bqm.to_coo() is deprecated since dimod 0.10.0. '
                      'Use dimod.serialization.coo.dump(bqm) or '
                      'dimod.serialization.coo.dumps(bqm, fp) instead.',
                      DeprecationWarning, stacklevel=2)

        import dimod.serialization.coo as coo

        if fp is None:
            return coo.dumps(self, vartype_header)
        else:
            coo.dump(self, fp, vartype_header)

    def to_file(self, *,
                ignore_labels: bool = False,
                spool_size: int = int(1e9),
                version: Union[int, Tuple[int, int]] = 2,
                ) -> tempfile.SpooledTemporaryFile:
        """Serialize the binary quadratic model to a file-like object.

        Note that BQMs with the ``object`` data type are serialized as ``float64``.

        Args:
            ignore_labels: Treat the BQM as unlabeled. This is useful for
                large BQMs to save on space.

            spool_size: Defines the ``max_size`` passed to the constructor of
                :class:`tempfile.SpooledTemporaryFile`. Determines whether
                the returned file-like's contents is kept on disk or in
                memory.

            version: The serialization version to use. Either as an integer
                defining the major version, or as a tuple, ``(major, minor)``.

        Format Specification (Version 1.0):

            This format is inspired by the `NPY format`_

            The first 8 bytes are a magic string: exactly "DIMODBQM".

            The next 1 byte is an unsigned byte: the major version of the file
            format.

            The next 1 byte is an unsigned byte: the minor version of the file
            format.

            The next 4 bytes form a little-endian unsigned int, the length of
            the header data HEADER_LEN.

            The next HEADER_LEN bytes form the header data. This is a
            json-serialized dictionary. The dictionary is exactly:

            .. code-block:: python

                dict(shape=bqm.shape,
                     dtype=bqm.dtype.name,
                     itype=bqm.data.index_dtype.name,
                     ntype=bqm.data.index_dtype.name,
                     vartype=bqm.vartype.name,
                     type=type(bqm).__name__,
                     variables=list(bqm.variables),
                     )

            It is terminated by a newline character and padded with spaces to
            make the entire length of the entire header divisible by 64.

            The binary quadratic model data comes after the header. The number
            of bytes can be determined by the data types and the number of
            variables and number of interactions (described in the ``shape``).

            The first ``dtype.itemsize`` bytes are the offset.
            The next ``num_variables`` * (``ntype.itemsize`` + ``dtype.itemsize``)
            bytes are the linear data. The linear data includes the neighborhood
            starts and the biases. The final
            2 * ``num_interactions`` * (``itype.itemsize`` + ``dtype.itemsize``)
            bytes are the quadratic data. Stored as ``(outvar, bias)`` pairs.

        Format Specification (Version 2.0):

            In order to make the header a more reasonable length, variable
            labels have been moved to the body. The ``variables`` field of the
            header dictionary now has a boolean value, making the dictionary:

            .. code-block:: python

                dict(shape=bqm.shape,
                     dtype=bqm.dtype.name,
                     itype=bqm.data.index_dtype.name,
                     ntype=bqm.data.index_dtype.name,
                     vartype=bqm.vartype.name,
                     type=type(bqm).__name__,
                     variables=any(v != i for i,v in enumerate(bqm.variables)),
                     )

            If the BQM is index-labeled, then no additional data is added.
            Otherwise, a new section is appended after the bias data.

            The first 4 bytes are exactly "VARS".

            The next 4 bytes form a little-endian unsigned int, the length of
            the variables array ``VARIABLES_LENGTH``.

            The next VARIABLES_LENGTH bytes are a json-serialized array. As
            constructed by `json.dumps(list(bqm.variables))`.

        .. _NPY format: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html


        """
        if self.dtype == np.dtype('O'):
            # todo: allow_pickle (defaulting False) that bypasses this
            return BinaryQuadraticModel(self, dtype=np.float64).to_file(
                ignore_labels=ignore_labels, spool_size=spool_size,
                version=version)

        version_tpl: Tuple[int, int] = (version, 0) if isinstance(version, Integral) else version

        if version_tpl not in [(1, 0), (2, 0)]:
            raise ValueError(f"Unsupported version: {version!r}")

        # the file we'll be writing to
        file = SpooledTemporaryFile(max_size=spool_size)

        # the data in the header
        data = dict(shape=self.shape,
                    dtype=self.dtype.name,
                    itype=self.data.index_dtype.name,
                    ntype=self.data.index_dtype.name,
                    vartype=self.vartype.name,
                    type=type(self).__name__,
                    )

        if version_tpl < (2, 0):
            # the variable labels go in the header
            if ignore_labels:
                data.update(variables=list(range(self.num_variables)))
            else:
                data.update(variables=self.variables.to_serializable())
        elif ignore_labels:
            data.update(variables=False)
        else:
            data.update(variables=not self.variables._is_range())

        write_header(file, BQM_MAGIC_PREFIX, data, version=version_tpl)

        # write the offset
        file.write(memoryview(self.data.offset).cast('B'))

        # write the linear biases and the neighborhood lengths
        file.write(memoryview(self.data._ilinear_and_degree()).cast('B'))

        # now the neighborhoods
        for vi in range(self.data.num_variables()):
            file.write(memoryview(self.data._ineighborhood(vi)).cast('B'))

        # finally the labels (if needed)
        if version_tpl >= (2, 0) and data['variables']:  # type: ignore[operator]
            file.write(VariablesSection(self.variables).dumps())

        file.seek(0)  # go back to the start
        return file

    def to_ising(self):
        """Convert a binary quadratic model to Ising format.

        If the binary quadratic model's vartype is not :class:`.Vartype.SPIN`,
        values are converted.

        Returns:
            tuple: 3-tuple of form ``(linear, quadratic, offset)``, where
            ``linear`` is a dict of linear biases, ``quadratic`` is a dict of
            quadratic biases, and ``offset`` is a number that represents the
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


        .. deprecated:: 0.10.0

            ``BinaryQuadraticModel.to_networkx_graph()`` will be removed in dimod 0.12.0.
            Use :func:`dimod.converters.to_networkx_graph()` instead.

        """
        warnings.warn('BinaryQuadraticModel.to_networkx_graph() is deprecated since '
                      'dimod 0.10.0 and will be removed in 0.12.0. '
                      'Use dimod.to_networkx_graph() instead.',
                      DeprecationWarning, stacklevel=2)
        from dimod.converters import to_networkx_graph  # avoid circular import
        return to_networkx_graph(self, node_attribute_name, edge_attribute_name)

    def to_numpy_matrix(self, variable_order=None):
        """Deprecated

        .. deprecated:: 0.10.0

            ``BinaryQuadraticModel.to_numpy_matrix()`` will be removed in dimod 0.12.0.

        """
        warnings.warn('bqm.to_numpy_matrix() is deprecated since dimod 0.10.0 '
                      'and will be removed in 0.12.0',
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

    def to_numpy_vectors(self,
                         variable_order: Optional[Sequence[Variable]] = None,
                         *,
                         dtype: Optional[DTypeLike] = None,
                         index_dtype: Optional[DTypeLike] = None,
                         sort_indices: bool = False,
                         sort_labels: bool = True,
                         return_labels: bool = False,
                         ) -> Union[BQMVectors, LabelledBQMVectors]:
        """Convert binary quadratic model to 1-dimensional NumPy arrays.

        Args:
            variable_order:
                Variable order for the vector output. By default uses
                the order of the binary quadratic model.

            sort_indices:
                Sort the indices of the interactions such that row is always
                less than column and then lexicographical.

            sort_labels:
                Equivalent to setting ``variable_order=sorted(bqm.variables)``.
                Ignored if ``variable_order`` is provided.

            return_labels:
                If True, returns a list of variable labels in the order used.

        Returns:
            A named tuple with fields ``linear_biases``, ``quadratic``, and
            ``offset``. If ``return_labels == True``, it also includes a ``labels``
            field.

            ``linear_biases`` is a length :attr:`BinaryQuadraticModel.num_variables`
            array containing the linear biases.

            ``quadratic`` is a named tuple with fields ``row_indices``,
            ``col_indices``, and ``biases``. ``row_indices`` and ``col_indices``
            are length :attr:`BinaryQuadraticModel.num_interactions`` arrays
            containing the interaction indices. ``biases`` contains the biases.

            ``offset`` is the offset.

            ``labels`` are the variable labels used.

        .. deprecated:: 0.10.0

            The ``dtype`` and ``index_dtype`` keyword arguments will be removed
            in 0.12.0. They currently do nothing.

        """
        if dtype is not None:
            warnings.warn(
                "The 'dtype' keyword argument is deprecated since dimod 0.10.0"
                " and will be removed in 0.12.0. Does nothing",
                DeprecationWarning, stacklevel=2)
        if index_dtype is not None:
            warnings.warn(
                "The 'index_dtype' keyword argument is deprecated since dimod "
                "0.10.0 and will be removed in 0.12.0. Does nothing",
                DeprecationWarning, stacklevel=2)

        try:
            return self.data.to_numpy_vectors(
                variable_order=variable_order,
                sort_indices=sort_indices,
                sort_labels=sort_labels,
                return_labels=return_labels,
                )
        except NotImplementedError:
            # methods can defer this
            pass

        num_variables = self.num_variables
        num_interactions = self.num_interactions

        if variable_order is None:
            variable_order = list(self.variables)

            if sort_labels:
                try:
                    variable_order.sort()
                except TypeError:
                    # can't sort unlike types in py3
                    pass

        if len(variable_order) != num_variables:
            raise ValueError("variable_order does not match the number of "
                             "variables")

        ldata = np.asarray([self.get_linear(v) for v in variable_order])

        label_to_idx = {v: idx for idx, v in enumerate(variable_order)}
        irow = []
        icol = []
        qdata = []
        for u, v, bias in self.iter_quadratic():
            irow.append(label_to_idx[u])
            icol.append(label_to_idx[v])
            qdata.append(bias)

        quadratic = QuadraticVectors(
            np.asarray(irow, dtype=np.int64),
            np.asarray(icol, dtype=np.int64),
            np.asarray(qdata),
            )

        if sort_indices:
            # row index should be less than col index, this handles
            # upper-triangular vs lower-triangular
            swaps = quadratic.row_indices > quadratic.col_indices
            if swaps.any():
                # in-place
                quadratic.row_indices[swaps],  quadratic.col_indices[swaps] = \
                    quadratic.col_indices[swaps], quadratic.row_indices[swaps]

            # sort lexigraphically
            order = np.lexsort((quadratic.row_indices, quadratic.col_indices))
            if not (order == range(len(order))).all():
                quadratic = QuadraticVectors(
                    quadratic.row_indices[order],
                    quadratic.col_indices[order],
                    quadratic.biases[order],
                    )

        if return_labels:
            return LabelledBQMVectors(ldata, quadratic, ldata.dtype.type(self.offset), variable_order)
        else:
            return BQMVectors(ldata, quadratic, ldata.dtype.type(self.offset))

    def to_qubo(self) -> Tuple[Mapping[Tuple[Variable, Variable], Bias], Bias]:
        """Convert a binary quadratic model to QUBO format.

        If the binary quadratic model's vartype is not :class:`.Vartype.BINARY`,
        values are converted.

        Returns:
            tuple: 2-tuple of form ``({(u, v): bias, ...}, offset)``, where
            ``u``, ``v``, are binary-valued variables and ``bias`` is their
            associated coefficient, and ``offset`` is a number that represents the
            constant offset of the binary quadratic model.
        """
        qubo = dict(self.binary.quadratic)
        qubo.update(((v, v), bias) for v, bias in self.binary.linear.items())
        return qubo, self.binary.offset

    def to_serializable(self,
                        *,
                        use_bytes: bool = False,
                        bias_dtype: None = None,  # does nothing
                        bytes_type: Callable = bytes,
                        ) -> Dict:
        """Convert the binary quadratic model to a serializable object.

        Args:
            use_bytes:
                If True, a compact representation representing the biases as
                bytes is used. Uses :meth:`~numpy.ndarray.tobytes`. This will
                prevent the returned dict from being JSON serialized, but
                it can be serialized with `BSON <http://bsonspec.org/>`_.

            bytes_type:
                This class will be used to wrap the bytes objects in the
                serialization if ``use_bytes`` is true.

        Returns:
            An object that can be serialized.

        Examples:

            Encode using JSON

            >>> import json
            ...
            >>> bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0},
            ...                                  {('a', 'b'): -1.0},
            ...                                  0.0,
            ...                                  dimod.SPIN)
            >>> s = json.dumps(bqm.to_serializable())

        See also:
            :meth:`~.BinaryQuadraticModel.from_serializable`

            :func:`json.dumps`, :func:`json.dump` JSON encoding functions

        .. deprecated:: 0.10.0

            The ``bias_dtype`` keyword argument will be removed in 0.12.0.
            Does nothing.

        """
        if bias_dtype is not None:
            warnings.warn(
                "The 'bias_dtype' keyword argument is deprecated since dimod "
                "0.10.0 and will be removed in 0.12.0. Does nothing",
                DeprecationWarning, stacklevel=2)

        from dimod import __version__

        schema_version = "3.0.0"

        ldata, (irow, icol, qdata), offset, variables = self.to_numpy_vectors(
            sort_indices=True,
            sort_labels=True,
            return_labels=True,
            )

        doc = {
            # metadata
            "type": type(self).__name__,
            "version": {"bqm_schema": schema_version},
            "use_bytes": bool(use_bytes),
            "index_type": irow.dtype.name,
            "bias_type": ldata.dtype.name,

            # bqm
            "num_variables": len(ldata),
            "num_interactions": len(irow),
            "variable_labels": Variables(variables).to_serializable(),
            "variable_type": self.vartype.name,
            "offset": float(offset),
            "info": dict(),  # for backwards compatibility
            }

        if use_bytes:
            # these are vectors so don't need to specify byte-order
            doc.update({'linear_biases': bytes_type(ldata.tobytes()),
                        'quadratic_biases': bytes_type(qdata.tobytes()),
                        'quadratic_head': bytes_type(irow.tobytes()),
                        'quadratic_tail': bytes_type(icol.tobytes())})
        else:
            doc.update({'linear_biases': ldata.tolist(),
                        'quadratic_biases': qdata.tolist(),
                        'quadratic_head': irow.tolist(),
                        'quadratic_tail': icol.tolist()})

        return doc

    def update(self, other: 'BinaryQuadraticModel'):
        """Add the variables, interactions, offset and biases from another
        binary quadratic model.

        Args:
            other:
                Binary quadratic model from which to add values.
        """
        try:
            self.data.update(other.data)
            return
        except (NotImplementedError, AttributeError):
            # methods can defer this
            pass

        if self.vartype is Vartype.SPIN:
            other = other.spin
        elif self.vartype is Vartype.BINARY:
            other = other.binary
        else:
            raise RuntimeError("unexpected vartype")

        self.add_linear_from((v, other.get_linear(v)) for v in other.variables)
        self.add_quadratic_from(other.iter_quadratic())
        self.offset += other.offset


BQM = BinaryQuadraticModel


class DictBQM(BQM, default_dtype=object):
    pass


class Float32BQM(BQM, default_dtype=np.float32):
    pass


class Float64BQM(BQM, default_dtype=np.float64):
    pass


AdjDictBQM = DictBQM
AdjVectorBQM = Float64BQM


@unique_variable_labels
def Binary(label: Optional[Variable] = None, bias: Bias = 1,
           dtype: Optional[DTypeLike] = None) -> BinaryQuadraticModel:
    r"""Return a binary quadratic model with a single binary variable.

    Args:
        label: Hashable label to identify the variable. Defaults to a
            generated :class:`uuid.UUID` as a string.
        bias: Bias to apply to the variable.
        dtype: Data type for the returned binary quadratic model.

    Returns:
        A :class:`.BinaryQuadraticModel` representing a binary variable.

    Examples:
        This example generates a BQM to represent the polynomial,
        :math:3x -1.5`, where :math:`x` is a binary variable with values
        :math:`\{0, 1\}`.

        >>> x = dimod.Binary('x')
        >>> bqm = 3*x - 1.5
        >>> print(bqm.to_polystring())
        -1.5 + 3*x
    """
    return BQM({label: bias}, {}, 0, Vartype.BINARY, dtype=dtype)


def Binaries(labels: Union[int, Iterable[Variable]],
             dtype: Optional[DTypeLike] = None) -> Iterator[BinaryQuadraticModel]:
    r"""Yield binary quadratic models, each with a single binary variable.

    Args:
        labels: Either an iterable of variable labels or a number. If a number
            labels are generated using :class:`uuid.UUID`.
        dtype: Data type for the returned binary quadratic models.

    Yields:
        A :class:`.BinaryQuadraticModel` for each binary variable.

    Examples:
        This example generates a BQM that represents a Boolean NOT gate as a
        penalty model, :math:`2xy - x - y`, described in the
        :ref:`NOT example <qpu_example_not>` section. The output of the
        brute-force solver, :class:`~dimod.reference.samplers.ExactSolver`,
        shows that best solutions are for assignments of the variables where
        :math:`z = \overline{x}`.

        >>> x, y = dimod.Binaries(["x", "y"])
        >>> bqm = 2*x*y - x - y
        >>> print(bqm.to_polystring())
        -x - y + 2*x*y
        >>> print(dimod.ExactSolver().sample(bqm).lowest())
           x  y energy num_oc.
        0  1  0   -1.0       1
        1  0  1   -1.0       1
        ['BINARY', 2 rows, 2 samples, 2 variables]
    """
    if isinstance(labels, Iterable):
        yield from (Binary(v, dtype=dtype) for v in labels)
    else:
        yield from (Binary(dtype=dtype) for _ in range(labels))


def BinaryArray(labels: Union[int, Iterable[Variable]],
                dtype: Optional[DTypeLike] = None) -> np.ndarray:
    r"""Return a NumPy array of binary quadratic models, each with a
    single binary variable.

    Args:
        labels: Either an iterable of variable labels or the number of required
            models. If a number, labels are generated using :class:`uuid.UUID`.
        dtype: Data type for the returned binary quadratic models.

    Returns:
        Array of :class:`.BinaryQuadraticModel`\ s, each representing a single
        binary variable.

    Examples:
        This example generates the QUBO, :math:`{x}^{T} {Q}{x}`, for a Boolean
        AND gate from the QUBO coefficients matrix,

        .. math::

             Q = \\begin{bmatrix}
                    0 & 1 & -2 \\\\
                    0 & 0 & -2 \\\\
                    0 & 0 & 3
                 \\end{bmatrix},

        derived in the :ref:`AND example <qpu_example_and>` section.

        >>> Q = [[0, 1, -2], [0, 0, -2], [0, 0, 3]]
        >>> x = dimod.BinaryArray(["in1", "in2", "out"])
        >>> bqm = x.dot(Q).dot(x)
        >>> print(bqm.to_polystring())
        3*out + in1*in2 - 2*in1*out - 2*in2*out
    """
    return _VariableArray(Binaries, labels, dtype)


@unique_variable_labels
def Spin(label: Optional[Variable] = None, bias: Bias = 1,
         dtype: Optional[DTypeLike] = None) -> BinaryQuadraticModel:
    r"""Return a binary quadratic model with a single spin variable.

    Args:
        label: Hashable label to identify the variable. Defaults to a
            generated :class:`uuid.UUID` as a string.
        bias: The bias to apply to the variable.
        dtype: Data type for the returned binary quadratic model.

    Returns:
        A :class:`.BinaryQuadraticModel` representing a spin-valued binary variable.

    Examples:
        This example generates a BQM to represent the polynomial,
        :math:`3s - 1.5`, where :math:`s` is a binary variable with spin values
        :math:`\{-1, 1\}`.

        >>> s = dimod.Spin('s')
        >>> bqm = 3*s - 1.5
        >>> print(bqm.to_polystring())
        -1.5 + 3*s
    """
    return BQM({label: bias}, {}, 0, Vartype.SPIN, dtype=dtype)


def Spins(labels: Union[int, Iterable[Variable]],
          dtype: Optional[DTypeLike] = None) -> Iterator[BinaryQuadraticModel]:
    """Yield binary quadratic models, each with a single spin variable.

    Args:
        labels: Either an iterable of variable labels or a number. If a number
            labels are generated using :class:`uuid.UUID`.
        dtype: Data type for the returned binary quadratic models.

    Yields:
        A :class:`.BinaryQuadraticModel` for each spin-valued binary variable.

    Examples:
        This example generates a BQM that represents the constraint :math:`s_1 = s_2`
        as a penalty model, :math:`-\\frac{1}{2}s_1s_2`, described in the
        :ref:`concept_penalty` section.
        The output of the brute-force solver,
        :class:`~dimod.reference.samplers.ExactSolver`, shows that best
        solutions are for eqaul assignments of the variables' values.

        >>> s1, s2 = dimod.Spins(["s1", "s2"])
        >>> bqm = -0.5*s1*s2
        >>> print(bqm.to_polystring())
        -0.5*s1*s2
        >>> print(dimod.ExactSolver().sample(bqm).lowest())
          s1 s2 energy num_oc.
        0 -1 -1   -0.5       1
        1 +1 +1   -0.5       1
        ['SPIN', 2 rows, 2 samples, 2 variables]
    """
    if isinstance(labels, Iterable):
        yield from (Spin(v, dtype=dtype) for v in labels)
    else:
        yield from (Spin(dtype=dtype) for _ in range(labels))


def SpinArray(labels: Union[int, Iterable[Variable]],
              dtype: Optional[DTypeLike] = None) -> np.ndarray:
    r"""Return a NumPy array of binary quadratic models, each with a
    single spin variable.

    Args:
        labels: Either an iterable of variable labels or the number of required
            models. If a number, labels are generated using :class:`uuid.UUID`.
        dtype: Data type for the returned binary quadratic models.

    Returns:
        Array of :class:`.BinaryQuadraticModel`\ s, each representing a single
        spin-valued binary variable.

    Examples:
        This example creates a BQM for a ferromagnetic chain of ten spin variables
        (representing, for instance, ten coupled qubits) with increasing biases.

        >>> import numpy as np
        >>> x = dimod.SpinArray(range(10))
        >>> lin_biases = np.linspace(1, 10, 10)
        >>> bqm = x.dot(lin_biases) - dimod.quicksum(x[1:10] * x[0:9])
        >>> print(bqm.to_polystring())           # doctest:+ELLIPSIS
        v0 + 2*v1 + 3*v2 + ... + 9*v8 + 10*v9 - v0*v1 - v1*v2 - v2*v3 - ... - v8*v9
    """
    return _VariableArray(Spins, labels, dtype)


def as_bqm(*args, cls: None = None, copy: bool = False,
           dtype: Optional[DTypeLike] = None) -> BinaryQuadraticModel:
    """Convert the input to a binary quadratic model.

    Converts the following input formats to a binary quadratic model (BQM):

        as_bqm(vartype)
            Creates an empty binary quadratic model.

        as_bqm(bqm)
            Creates a BQM from another BQM. See ``copy`` and ``cls`` kwargs below.

        as_bqm(bqm, vartype)
            Creates a BQM from another BQM, changing to the appropriate
            ``vartype`` if necessary. See ``copy`` and ``cls`` kwargs below.

        as_bqm(n, vartype)
            Creates a BQM with ``n`` variables, indexed linearly from zero,
            setting all biases to zero.

        as_bqm(quadratic, vartype)
            Creates a BQM from quadratic biases given as a square array_like_
            or a dictionary of the form ``{(u, v): b, ...}``. Note that when
            formed with SPIN-variables, biases on the diagonal are added to the
            offset.

        as_bqm(linear, quadratic, vartype)
            Creates a BQM from linear and quadratic biases, where ``linear`` is a
            one-dimensional array_like_ or a dictionary of the form
            ``{v: b, ...}``, and ``quadratic`` is a square array_like_ or a
            dictionary of the form ``{(u, v): b, ...}``. Note that when formed
            with SPIN-variables, biases on the diagonal are added to the offset.

        as_bqm(linear, quadratic, offset, vartype)
            Creates a BQM from linear and quadratic biases, where ``linear`` is a
            one-dimensional array_like_ or a dictionary of the form
            ``{v: b, ...}``, and ``quadratic`` is a square array_like_ or a
            dictionary of the form ``{(u, v): b, ...}``, and ``offset`` is a
            numerical offset. Note that when formed with SPIN-variables, biases
            on the diagonal are added to the offset.

    Args:
        *args: See above.

        cls: Deprecated. This function always returns a
            :class:`.BinaryQuadraticModel`.

        copy: If False, a new BQM is only constructed when
            necessary.

        dtype: Data type of the returned BQM.

    Returns:
        A binary quadratic model.

    .. _array_like: https://numpy.org/doc/stable/user/basics.creation.html

    """
    if cls is not None:
        warnings.warn("the 'cls' keyword argument of 'as_bqm' is deprecated "
                      " as of dimod 0.10.0 and does nothing.",
                      DeprecationWarning, stacklevel=2)

    if not copy:
        # the only cases we don't copy in are when the dtype and vartype match
        # a given bqm
        if isinstance(args[0], BinaryQuadraticModel):
            bqm = args[0]
            if dtype is None or np.dtype(dtype) == bqm.dtype:
                if len(args) == 1:
                    return bqm
                elif len(args) == 2:
                    vartype = args[1]
                    if bqm.vartype is as_vartype(vartype):
                        return bqm

    return BinaryQuadraticModel(*args, dtype=dtype)


def quicksum(iterable: Iterable[Union[BinaryQuadraticModel, QuadraticModel, Bias]]
             ) -> Union[BinaryQuadraticModel, QuadraticModel]:
    r"""Sum `iterable`'s items.

    This function is an alternative to the built-in :func:`sum`. It is
    generally faster when adding many :class:`BinaryQuadraticModel`\s and
    :class:`QuadraticModel`\s because it creates fewer intermediate objects.

    """
    iterable = iter(iterable)

    try:
        model = next(iterable)
    except StopIteration:
        return QuadraticModel()

    model = copy.deepcopy(model)

    for obj in iterable:
        model += obj

    return model


# register fileview loader
load.register(BQM_MAGIC_PREFIX, BinaryQuadraticModel.from_file)
