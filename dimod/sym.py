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

import abc
import enum
import typing

from dimod.typing import Variable

__all__ = ['Sense', 'Eq', 'Ge', 'Le']


class Sense(enum.Enum):
    """Sense of a constraint.

    Supported values are:

    * ``Le``: less or equal (:math:`\le`)
    * ``Ge``: greater or equal (:math:`\ge`)
    * ``Eq``: equal (:math:`=`)

    Example:

        >>> from dimod import ConstrainedQuadraticModel, Integer
        >>> i = Integer("i")
        >>> cqm = ConstrainedQuadraticModel()
        >>> cqm.add_constraint(i <= 3, "i le 3")
        'i le 3'

    """

    Le = '<='
    Ge = '>='
    Eq = '=='


class Comparison(abc.ABC):
    def __init__(self, lhs, rhs):
        # todo: type checking
        self.lhs = lhs
        self.rhs = rhs

    def __init_subclass__(cls, sense=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if sense is not None:
            cls.sense = sense if isinstance(sense, Sense) else Sense(sense)

    @property
    @abc.abstractmethod
    def sense(self):
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.lhs!r}, {self.rhs!r})"

    def __str__(self) -> str:
        return self.to_polystring()

    def to_polystring(self, encoder: typing.Optional[typing.Callable[[Variable], str]] = None) -> str:
        """Return a string representing the model as an equation.

        Args:
            encoder: A function mapping variables to a string. By default
                string variables are mapped directly whereas all other types
                are mapped to a string :code:`f"v{variable!r}"`.

        Returns:
            A string representing the comparison.

        """
        return f"{self.lhs.to_polystring(encoder)} {self.sense.value} {self.rhs!r}"


class Eq(Comparison, sense='=='):
    def __bool__(self):
        try:
            return self.lhs.is_equal(self.rhs)
        except AttributeError:
            return False


class Ge(Comparison, sense='>='):
    pass


class Le(Comparison, sense='<='):
    pass
