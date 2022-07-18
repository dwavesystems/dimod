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
import functools

from collections.abc import Collection, Iterator, Callable, Sequence
from operator import add
from typing import Tuple, Iterator, Optional, Mapping, Any

import numpy as np

try:
    from numpy.typing import ArrayLike, DTypeLike
except ImportError:
    DTypeLike = Any

from dimod.binary.pybqm import pyBQM
from dimod.sampleset import as_samples
from dimod.typing import Bias, Variable
from dimod.vartypes import BINARY, SPIN, Vartype, as_vartype, VartypeLike

__all__ = ['VartypeView']


def view_method(f):
    @functools.wraps(f)
    def wrapper(obj, *args, **kwargs):
        # explicitly check that we're getting the expected vartype combinations
        if obj._vartype == obj.data.vartype():
            # if the vartype matches, just use the underlying data method
            return getattr(obj.data, f.__name__)(*args, **kwargs)

        elif obj.data.vartype() is SPIN and obj._vartype is BINARY:
            return f(obj, *args, **kwargs)

        elif obj.data.vartype() is BINARY and obj._vartype is SPIN:
            return f(obj, *args, **kwargs)

        else:
            raise RuntimeError("unexpected vartype combination")

    return wrapper


class VartypeView:
    def __init__(self, data, vartype: Vartype):
        self.data = data
        self._vartype = vartype

    def __copy__(self):
        # since we'd need to copy the underlying data anyway, let's just return
        # that instead. It doesn't really make sense to maintain the view
        # of a detached copy.
        new = copy.copy(self.data)
        new.change_vartype(self._vartype)
        return new

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def variables(self) -> Sequence:
        return self.data.variables

    @property
    def offset(self) -> Bias:
        if self._vartype == self.data.vartype():
            return self.data.offset
        elif self._vartype is BINARY and self.data.vartype() is SPIN:
            # binary <- spin
            return (self.data.offset
                    - self.data.reduce_linear(add, 0)
                    + self.data.reduce_quadratic(add, 0))
        elif self._vartype is SPIN and self.data.vartype() is BINARY:
            # spin <- binary
            return (self.data.offset
                    + self.data.reduce_linear(add, 0) / 2
                    + self.data.reduce_quadratic(add, 0) / 4)
        else:
            raise RuntimeError("unexpected vartype combination")

    @offset.setter
    def offset(self, bias: Bias):
        if self._vartype == self.data.vartype:
            self.data.offset = bias
        elif self._vartype is BINARY and self.data.vartype() is SPIN:
            # binary <- spin
            self.data.offset += bias - self.offset  # use the difference
        elif self._vartype is SPIN and self.data.vartype() is BINARY:
            # spin <- binary
            self.data.offset += bias - self.offset  # use the difference
        else:
            raise RuntimeError("unexpected vartype combination")

    @view_method
    def add_linear(self, v: Variable, bias: Bias):
        if self._vartype is BINARY:  # binary -> spin
            self.data.add_linear(v, bias / 2)
            self.data.offset += bias / 2
        else:  # spin -> binary
            self.data.add_linear(v, 2*bias)
            self.data.offset -= bias

    def add_linear_equality_constraint(self, *args, **kwargs):
        raise NotImplementedError  # defer to caller

    @view_method
    def add_quadratic(self, u: Variable, v: Variable, bias: Bias):
        if self._vartype is BINARY:  # binary -> spin
            self.data.add_quadratic(u, v, bias / 4)
            self.data.add_linear(u, bias / 4)
            self.data.add_linear(v, bias / 4)
            self.data.offset += bias / 4
        else:  # spin -> binary
            self.data.add_quadratic(u, v, 4*bias)
            self.data.add_linear(u, -2*bias)
            self.data.add_linear(v, -2*bias)
            self.data.offset += bias

    def add_variable(self, v: Optional[Variable] = None,
                     bias: Bias = 0) -> Variable:
        v = self.data.add_variable(v)
        self.add_linear(v, bias)
        return v

    def change_vartype(self, vartype: VartypeLike):
        self._vartype = as_vartype(vartype)

    def clear(self) -> None:
        return self.data.clear()

    def degree(self, v: Variable):
        return self.data.degree(v)

    @view_method
    def energies(self, samples_like, dtype: DTypeLike = None):
        samples, labels = as_samples(samples_like, copy=True)

        if self._vartype is BINARY:  # binary -> spin
            samples *= 2
            samples -= 1
        else:  # spin -> binary
            samples += 1
            samples //= 2

        return self.data.energies((samples, labels), dtype=dtype)

    @view_method
    def get_linear(self, v: Variable) -> Bias:
        if self._vartype is BINARY:  # binary <- spin
            return (2 * self.data.get_linear(v)
                    - 2 * self.data.reduce_neighborhood(v, add, 0))
        else:  # spin <- binary
            return (self.data.get_linear(v) / 2
                    + self.data.reduce_neighborhood(v, add, 0) / 4)

    @view_method
    def get_quadratic(self, u: Variable, v: Variable,
                      default: Optional[Bias] = None) -> Bias:
        if u == v:
            raise ValueError(f"{u!r} cannot have an interaction with itself")

        try:
            if self._vartype is BINARY:  # binary <- spin
                return 4 * self.data.get_quadratic(u, v)
            else:  # spin <- binary
                return self.data.get_quadratic(u, v) / 4
        except ValueError as err:
            if default is None:
                raise ValueError(
                    f"{u!r} and {v!r} have no interaction") from None
            return default

    def is_linear(self) -> bool:
        return self.data.is_linear()

    @view_method
    def iter_neighborhood(self, v: Variable) -> Iterator[Tuple[Variable, Bias]]:
        if self._vartype is BINARY:  # binary <- spin
            for u, bias in self.data.iter_neighborhood(v):
                yield u, 4 * bias
        else:  # spin <- binary
            for u, bias in self.data.iter_neighborhood(v):
                yield u, bias / 4

    @view_method
    def iter_quadratic(self) -> Iterator[Tuple[Variable, Variable, Bias]]:
        if self._vartype is BINARY:  # binary <- spin
            for u, v, bias in self.data.iter_quadratic():
                yield u, v, 4 * bias
        else:  # spin <- binary
            for u, v, bias in self.data.iter_quadratic():
                yield u, v, bias / 4

    def nbytes(self, capacity: bool = False) -> int:
        return self.data.nbytes(capacity=capacity)

    def num_interactions(self):
        return self.data.num_interactions()

    def num_variables(self):
        return self.data.num_variables()

    def reduce_linear(self, function: Callable,
                      initializer: Optional[Bias] = None) -> Bias:
        gen = (self.get_linear(v) for v in self.variables)
        if initializer is None:
            return functools.reduce(function, gen)
        else:
            return functools.reduce(function, gen, initializer)

    def reduce_neighborhood(self, v: Variable, function: Callable,
                            initializer: Optional[Bias] = None) -> Bias:
        gen = (b for _, b in self.iter_neighborhood(v))
        if initializer is None:
            return functools.reduce(function, gen)
        else:
            return functools.reduce(function, gen, initializer)

    def reduce_quadratic(self, function: Callable,
                         initializer: Optional[Bias] = None) -> Bias:
        gen = (b for _, _, b in self.iter_quadratic())
        if initializer is None:
            return functools.reduce(function, gen)
        else:
            return functools.reduce(function, gen, initializer)

    @view_method
    def remove_interaction(self, u: Variable, v: Variable):
        self.get_quadratic(u, v)  # raise an error if it doesn't exist
        self.set_quadratic(u, v, 0)  # zero it out in the appropriate vartype
        self.data.remove_interaction(u, v)

    @view_method
    def remove_variable(self, v: Optional[Variable] = None) -> Variable:
        if v is None:
            try:
                v = self.variables[-1]
            except IndexError:
                raise ValueError("cannot pop from an empty model")
        # set everything associated with `v` to 0
        for u, _ in self.iter_neighborhood(v):
            self.set_quadratic(u, v, 0)
        self.set_linear(v, 0)
        return self.data.remove_variable(v)

    def relabel_variables(self, mapping: Mapping[Variable, Variable]):
        self.data.relabel_variables(mapping)

    def relabel_variables_as_integers(self) -> Mapping[int, Variable]:
        return self.data.relabel_variables_as_integers()

    @view_method
    def set_linear(self, v: Variable, bias: Bias):
        self.add_linear(v, 0)  # make sure it exists
        self.add_linear(v, bias - self.get_linear(v))  # just add the delta

    def set_quadratic(self, u: Variable, v: Variable, bias: Bias):
        self.add_variable(u)
        self.add_variable(v)
        # just add the delta
        self.add_quadratic(u, v, 0)  # make sure it exists
        self.add_quadratic(u, v, bias - self.get_quadratic(u, v))

    def to_numpy_vectors(self, *args, **kwargs):
        raise NotImplementedError  # defer to the caller

    def update(self, *args, **kwargs):
        raise NotImplementedError  # defer to the caller

    def vartype(self, v: Optional[Variable] = None) -> Vartype:
        return self._vartype
