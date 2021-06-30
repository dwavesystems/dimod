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

from collections.abc import Callable
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Iterator, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

try:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    ArrayLike = Any
    DTypeLike = Any

from dimod.decorators import forwarding_method
from dimod.quadratic.cyqm import cyQM_float32, cyQM_float64
from dimod.typing import Variable, Bias, VartypeLike
from dimod.variables import Variables
from dimod.vartypes import Vartype
from dimod.views.quadratic import QuadraticViewsMixin

if TYPE_CHECKING:
    # avoid circular imports
    from dimod import BinaryQuadraticModel


__all__ = ['QuadraticModel', 'QM', 'Integer']


class QuadraticModel(QuadraticViewsMixin):
    _DATA_CLASSES = {
        np.dtype(np.float32): cyQM_float32,
        np.dtype(np.float64): cyQM_float64,
    }

    DEFAULT_DTYPE = np.float64
    """The default dtype used to construct the class."""

    def __init__(self, *, dtype: Optional[DTypeLike] = None):
        dtype = np.dtype(self.DEFAULT_DTYPE) if dtype is None else np.dtype(dtype)
        self.data = self._DATA_CLASSES[np.dtype(dtype)]()

    def __deepcopy__(self, memo: Dict[int, Any]) -> 'QuadraticModel':
        new = type(self).__new__(type(self))
        new.data = deepcopy(self.data, memo)
        memo[id(self)] = new
        return new

    def __add__(self, other: Union['QuadraticModel', Bias]) -> 'QuadraticModel':
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

    def __iadd__(self, other: Union['QuadraticModel', Bias]) -> 'QuadraticModel':
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

    def __mul__(self, other: Union['QuadraticModel', Bias]) -> 'QuadraticModel':
        if isinstance(other, QuadraticModel):
            if not (self.is_linear() and other.is_linear()):
                raise TypeError(
                    "cannot multiply QMs with interactions")

            new = type(self)(dtype=self.dtype)

            for v in self.variables:
                new.add_variable(self.vartype(v), v)
            for v in other.variables:
                new.add_variable(other.vartype(v), v)

            self_offset = self.offset
            other_offset = other.offset

            for u, ubias in self.linear.items():
                for v, vbias in other.linear.items():
                    if u == v:
                        if self.vartype is Vartype.BINARY:
                            new.add_linear(u, ubias*vbias)
                        else:
                            new.offset += ubias * vbias
                    else:
                        new.add_quadratic(u, v, ubias * vbias)

                new.add_linear(u, ubias * other_offset)

            for v, bias in other.linear.items():
                new.add_linear(v, bias*self_offset)

            return new
        if isinstance(other, Number):
            new = self.copy()
            new.scale(other)
            return new
        return NotImplemented

    def __imul__(self, other: Bias) -> 'QuadraticModel':
        # in-place multiplication is only defined for numbers
        if isinstance(other, Number):
            raise NotImplementedError
        return NotImplemented

    def __rmul__(self, other: Bias) -> 'QuadraticModel':
        # should only miss on number
        if isinstance(other, Number):
            return self * other  # communative
        return NotImplemented

    def __neg__(self: 'QuadraticModel') -> 'QuadraticModel':
        raise NotImplementedError

    def __sub__(self, other: Union['QuadraticModel', Bias]) -> 'QuadraticModel':
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

    def __isub__(self, other: Union['QuadraticModel', Bias]) -> 'QuadraticModel':
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
    def shape(self) -> Tuple[int, int]:
        """A 2-tuple of :attr:`num_variables` and :attr:`num_interactions`."""
        return self.num_variables, self.num_interactions

    @property
    def variables(self) -> Variables:
        """The variables of the quadratic model"""
        return self.data.variables

    @forwarding_method
    def add_linear(self, v: Variable, bias: Bias):
        """Add a quadratic term."""
        return self.data.add_linear

    @forwarding_method
    def add_quadratic(self, u: Variable, v: Variable, bias: Bias):
        return self.data.add_quadratic

    @forwarding_method
    def add_variable(self, vartype: VartypeLike,
                     v: Optional[Variable] = None, bias: Bias = 0) -> Variable:
        """Add a quadratic term."""
        return self.data.add_variable

    def copy(self):
        """Return a copy."""
        return deepcopy(self)

    @forwarding_method
    def degree(self, v: Variable) -> int:
        """Return the degree of variable `v`.

        The degree is the number of interactions that contain `v`.
        """
        return self.data.degree

    @classmethod
    def from_bqm(cls, bqm: 'BinaryQuadraticModel') -> 'QuadraticModel':
        obj = cls(dtype=bqm.dtype)

        # this can be improved a great deal with c++, but for now let's use
        # the python fallback for everything

        for v in bqm.variables:
            obj.set_linear(obj.add_variable(bqm.vartype, v), bqm.get_linear(v))

        for u, v, bias in bqm.iter_quadratic():
            obj.set_quadratic(u, v, bias)

        obj.offset = bqm.offset

        return obj

    @forwarding_method
    def get_linear(self, v: Variable) -> Bias:
        """Get the linear bias of `v`."""
        return self.data.get_linear

    @forwarding_method
    def get_quadratic(self, u: Variable, v: Variable,
                      default: Optional[Bias] = None) -> Bias:
        return self.data.get_quadratic

    def is_linear(self) -> bool:
        """Return True if the model has no quadratic interactions."""
        return self.data.is_linear()

    @forwarding_method
    def iter_neighborhood(self, v: Variable) -> Iterator[Tuple[Variable, Bias]]:
        """Iterate over the neighbors and quadratic biases of a variable."""
        return self.data.iter_neighborhood

    @forwarding_method
    def iter_quadratic(self) -> Iterator[Tuple[Variable, Variable, Bias]]:
        return self.data.iter_quadratic

    @forwarding_method
    def reduce_linear(self, function: Callable,
                      initializer: Optional[Bias] = None) -> Any:
        """Apply function of two arguments cumulatively to the linear biases.
        """
        return self.data.reduce_linear

    @forwarding_method
    def reduce_neighborhood(self, v: Variable, function: Callable,
                            initializer: Optional[Bias] = None) -> Any:
        """Apply function of two arguments cumulatively to the quadratic biases
        associated with a single variable.
        """
        return self.data.reduce_neighborhood

    @forwarding_method
    def reduce_quadratic(self, function: Callable,
                         initializer: Optional[Bias] = None) -> Any:
        """Apply function of two arguments cumulatively to the quadratic
        biases.
        """
        return self.data.reduce_quadratic

    def remove_interaction(self, u: Variable, v: Variable):
        # This is needed for the views, but I am not sure how often users are
        # removing variables/interactions. For now let's leave it here so
        # we satisfy the ABC and see if it comes up. If not, in the future we
        # can consider removing __delitem__ from the various views.
        raise NotImplementedError("not yet implemented - please open a feature request")

    def remove_variable(self, v: Optional[Variable] = None) -> Variable:
        # see note in remove_interaction
        raise NotImplementedError("not yet implemented - please open a feature request")

    @forwarding_method
    def scale(self, scalar: Bias):
        return self.data.scale

    @forwarding_method
    def set_linear(self, v: Variable, bias: Bias):
        """Set the linear bias of `v`.

        Raises:
            TypeError: If `v` is not hashable.

        """
        return self.data.set_linear

    @forwarding_method
    def set_quadratic(self, u: Variable, v: Variable, bias: Bias):
        """Set the quadratic bias of `(u, v)`.

        Raises:
            TypeError: If `u` or `v` is not hashable.

        """
        return self.data.set_quadratic

    def update(self, other: 'QuadraticModel'):
        # this can be improved a great deal with c++, but for now let's use
        # python for simplicity

        if any(v in self.variables and self.vartype(v) != other.vartype(v)
               for v in other.variables):
            raise ValueError("given quadratic model has variables with conflicting vartypes")

        for v in other.variables:
            self.add_linear(self.add_variable(other.vartype(v), v), other.get_linear(v))

        for u, v, bias in other.iter_quadratic():
            self.add_quadratic(u, v, bias)

        self.offset += other.offset

    @forwarding_method
    def vartype(self, v: Variable) -> Vartype:
        """The variable type of the given variable."""
        return self.data.vartype


QM = QuadraticModel


def Integer(label: Variable, bias: Bias = 1,
            dtype: Optional[DTypeLike] = None) -> QuadraticModel:
    if label is None:
        raise TypeError("label cannot be None")
    qm = QM(dtype=dtype)
    v = qm.add_variable(Vartype.INTEGER, label)
    qm.set_linear(v, bias)
    return qm
