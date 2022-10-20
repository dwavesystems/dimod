# Copyright 2022 D-Wave Systems Inc.
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

from __future__ import annotations

import abc
import numbers
import tempfile
import typing

import dimod

from dimod.constrained.cyexpression import cyObjectiveView, cyConstraintView
from dimod.views.quadratic import QuadraticViewsMixin


class _ExpressionMixin(QuadraticViewsMixin):
    # dev note: I think most of these can actually be promoted to the QuadraticViewsMixin
    # one, but let's leave it alone for now

    @property
    @abc.abstractmethod
    def num_variables(self) -> int:
        pass

    @property
    def shape(self) -> typing.Tuple[int, int]:
        return self.num_variables, self.num_interactions

    def energy(self, sample, dtype=None) -> dimod.typing.Bias:
        energies = self.energies(sample)

        if not len(energies):  # the empty case, happens with []
            return self.dtype.type(0)

        energy, = energies
        return energy

    @abc.abstractmethod
    def energies(self, samples_like):
        pass

    def __add__(self, other):
        if isinstance(other, QuadraticViewsMixin):
            qm = dimod.QuadraticModel()
            qm.update(self)
            qm.update(other)
            return qm

        if isinstance(other, numbers.Number):
            raise NotImplementedError

        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, QuadraticViewsMixin):
            qm = dimod.QuadraticModel()
            qm.update(other)
            qm.update(self)
            return qm

        if isinstance(other, numbers.Number):
            return self + other

        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, numbers.Number):
            return Eq(self, other)
        # support equality for backwards compatibility
        return self.is_equal(other)

    def is_almost_equal(self, other, places: int = 7) -> bool:
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
        if isinstance(other, numbers.Number):
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

    def is_equal(self, other) -> bool:
        """Return True if the given model has the same variables, vartypes and biases.

        Args:
            other: Quadratic model to compare against.
        """
        if isinstance(other, numbers.Number):
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

    def to_file(self, *, spool_size: int = int(1e9)) -> tempfile.SpooledTemporaryFile:
        # serialize as a QM.
        # todo: turn this into a function rather than a method
        from dimod.quadratic.quadratic_model import QuadraticModel
        from dimod.variables import Variables

        class Model:
            data = self
            dtype = self.dtype
            index_dtype = self.index_dtype
            num_variables = self.num_variables
            shape = (self.num_variables, self.num_interactions)
            variables = Variables(self.variables)

        return QuadraticModel.to_file(Model(), spool_size=spool_size)

    @abc.abstractmethod
    def vartype(self, v: dimod.typing.Variable) -> dimod.Vartype:
        pass


class ObjectiveView(cyObjectiveView, _ExpressionMixin):
    pass


class ConstraintView(cyConstraintView, _ExpressionMixin):
    pass
