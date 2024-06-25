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
import collections.abc
import io
import operator

from collections.abc import ItemsView, MutableMapping
from typing import Any, Callable, Collection, Iterable, Iterator, Mapping, Optional, Tuple, Union

from dimod.typing import Bias, Variable

__all__ = ['Adjacency', 'Linear', 'Quadratic', 'QuadraticViewsMixin']


class TermsView:
    __slots__ = ['_model']

    def __init__(self, model: 'QuadraticViewsMixin'):
        self._model = model

    def __repr__(self):
        # let's just print the whole (potentially massive) thing for now, in
        # the future we'd like to do something a bit more clever (like hook
        # into dimod's Formatter)
        stream = io.StringIO()
        stream.write('{')
        last = len(self) - 1
        for i, (key, value) in enumerate(self.items()):
            stream.write(f'{key!r}: {value}')
            if i != last:
                stream.write(', ')
        stream.write('}')
        return stream.getvalue()


class Neighborhood(Mapping, TermsView):
    __slots__ = ['_var']

    def __init__(self, model: 'QuadraticViewsMixin', v: Variable):
        super().__init__(model)
        self._var = v

    def __getitem__(self, v: Variable) -> Bias:
        try:
            return self._model.get_quadratic(self._var, v)
        except ValueError as e:
            raise KeyError(*e.args)

    def __iter__(self) -> Iterator[Variable]:
        for v, _ in self._model.iter_neighborhood(self._var):
            yield v

    def __len__(self) -> int:
        return self._model.degree(self._var)

    def __setitem__(self, v: Variable, bias: Bias):
        self._model.set_quadratic(self._var, v, bias)

    def max(self, *, default: Optional[Bias] = None) -> Bias:
        """Return the maximum quadratic bias of the neighborhood"""
        try:
            return self._model.reduce_neighborhood(self._var, max)
        except TypeError as err:
            pass

        if default is None:
            raise ValueError("cannot find max of an empty sequence")

        return default

    def min(self, *, default: Optional[Bias] = None) -> Bias:
        """Return the minimum quadratic bias of the neighborhood"""
        try:
            return self._model.reduce_neighborhood(self._var, min)
        except TypeError as err:
            pass

        if default is None:
            raise ValueError("cannot find min of an empty sequence")

        return default

    def sum(self, start=0) -> Bias:
        """Return the sum of the quadratic biases of the neighborhood"""
        return self._model.reduce_neighborhood(self._var, operator.add, start)


class Adjacency(Mapping, TermsView):
    """Quadratic biases as a nested dict of dicts.

    Accessed like a dict of dicts, where the keys of the outer dict are all
    of the model's variables (e.g. `v`) and the values are the neighborhood of
    `v`. Each neighborhood is a dict where the keys are the neighbors of `v`
    and the values are their associated quadratic biases.
    """
    def __getitem__(self, v: Variable) -> Neighborhood:
        return Neighborhood(self._model, v)

    def __iter__(self) -> Iterator[Variable]:
        yield from self._model.variables

    def __len__(self) -> int:
        return len(self._model.variables)


class Linear(MutableMapping, TermsView):
    """Linear biases as a mapping.

    Accessed like a dict, where keys are the variables of the model and values
    are the linear biases.
    """
    def __delitem__(self, v: Variable):
        try:
            self._model.remove_variable(v)
        except ValueError:
            raise KeyError(repr(v))

    def __getitem__(self, v: Variable) -> Bias:
        try:
            return self._model.get_linear(v)
        except ValueError as e:
            raise KeyError(*e.args)

    def __iter__(self) -> Iterator[Variable]:
        yield from self._model.variables

    def __len__(self) -> int:
        return len(self._model.variables)

    def __setitem__(self, v: Variable, bias: Bias):
        self._model.set_linear(v, bias)

    def max(self, *, default: Optional[Bias] = None) -> Bias:
        """Return the maximum linear bias."""
        try:
            return self._model.reduce_linear(max)
        except TypeError:
            pass

        if default is None:
            raise ValueError("cannot find max of an empty sequence")

        return default

    def min(self, *, default: Optional[Bias] = None) -> Bias:
        """Return the minimum linear bias."""
        try:
            return self._model.reduce_linear(min)
        except TypeError:
            pass

        if default is None:
            raise ValueError("cannot find min of an empty sequence")

        return default

    def sum(self, start=0):
        """Return the sum of the linear biases."""
        return self._model.reduce_linear(operator.add, start)


class QuadraticItemsView(ItemsView):
    # speed up iteration
    def __iter__(self) -> Iterator[Tuple[Tuple[Variable, Variable], Bias]]:
        for u, v, bias in self._mapping._model.iter_quadratic():
            yield (u, v), bias


class Quadratic(MutableMapping, TermsView):
    """Quadratic biases as a flat mapping.

    Accessed like a dict, where keys are 2-tuples of variables, which represent
    an interaction and values are the quadratic biases.
    """
    def __delitem__(self, uv: Tuple[Variable, Variable]):
        try:
            self._model.remove_interaction(*uv)
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

    def __getitem__(self, uv: Tuple[Variable, Variable]) -> Bias:
        try:
            return self._model.get_quadratic(*uv)
        except ValueError as e:
            raise KeyError(*e.args)

    def __iter__(self) -> Iterator[Tuple[Variable, Variable]]:
        for u, v, _ in self._model.iter_quadratic():
            yield u, v

    def __len__(self) -> int:
        return self._model.num_interactions

    def __setitem__(self, uv: Tuple[Variable, Variable], bias: Bias):
        self._model.set_quadratic(*uv, bias)

    def items(self) -> ItemsView:
        return QuadraticItemsView(self)

    def max(self, *, default: Optional[Bias] = None) -> Bias:
        """Return the maximum quadratic bias."""
        try:
            return self._model.reduce_quadratic(max)
        except TypeError:
            pass

        if default is None:
            raise ValueError("cannot find max of an empty sequence")

        return default

    def min(self, *, default: Optional[Bias] = None) -> Bias:
        """Return the minimum quadratic bias."""
        try:
            return self._model.reduce_quadratic(min)
        except TypeError:
            pass

        if default is None:
            raise ValueError("cannot find min of an empty sequence")

        return default

    def sum(self, start=0):
        """Return the sum of the quadratic biases."""
        return self._model.reduce_quadratic(operator.add, start)


class QuadraticViewsMixin(abc.ABC):

    @property
    def adj(self) -> Adjacency:
        """Adjacency structure as a nested mapping of mappings.

        Accessed like a dict of dicts, where the keys of the outer dict are all
        of the model's variables (e.g. ``v``) and the values are the neighborhood
        of ``v``. Each neighborhood is a dict where the keys are the neighbors of
        ``v`` and the values are their associated quadratic biases.

        Examples:
            >>> from dimod import QuadraticModel, Binary, Integer
            >>> qm = QuadraticModel()
            >>> qm.add_variables_from('BINARY', ['x', 'y'])
            >>> qm.add_variables_from('INTEGER', ['i', 'j'])
            >>> qm.add_quadratic('i', 'j', 2)
            >>> qm.add_quadratic('x', 'i', -1)
            >>> qm.adj
            {'x': {'i': -1.0}, 'y': {}, 'i': {'x': -1.0, 'j': 2.0}, 'j': {'i': 2.0}}

        """
        # we could use cached property but this is way simpler and doesn't
        # break the docs
        try:
            return self._adj  # type: ignore[has-type]
        except AttributeError:
            pass
        self._adj = adj = Adjacency(self)
        return adj

    @property
    def linear(self) -> Linear:
        """Linear biases as a mapping.

        Accessed like a dict, where keys are the variables of the model and
        values are the linear biases.

        Examples:
            >>> from dimod import QuadraticModel, Binary, Integer
            >>> qm = QuadraticModel()
            >>> qm.add_variables_from('BINARY', ['x', 'y'])
            >>> qm.add_variables_from('INTEGER', ['i', 'j'])
            >>> qm.add_linear('x', 0.5)
            >>> qm.add_linear('i', -2)
            >>> qm.linear
            {'x': 0.5, 'y': 0.0, 'i': -2.0, 'j': 0.0}

        """
        # we could use cached property but this is way simpler and doesn't
        # break the docs
        try:
            return self._linear  # type: ignore[has-type]
        except AttributeError:
            pass
        self._linear = linear = Linear(self)
        return linear

    @property
    def quadratic(self) -> Quadratic:
        """Quadratic biases as a flat mapping.

        Accessed like a dict, where keys are 2-tuples of variables, which
        represent an interaction and values are the quadratic biases.

        Examples:
            >>> from dimod import QuadraticModel, Binary, Integer
            >>> qm = QuadraticModel()
            >>> qm.add_variables_from('BINARY', ['x', 'y'])
            >>> qm.add_variables_from('INTEGER', ['i', 'j'])
            >>> qm.add_quadratic('i', 'j', 2)
            >>> qm.add_quadratic('x', 'i', -1)
            >>> qm.quadratic
            {('i', 'x'): -1.0, ('j', 'i'): 2.0}
        """
        # we could use cached property but this is way simpler and doesn't
        # break the docs
        try:
            return self._quadratic  # type: ignore[has-type]
        except AttributeError:
            pass
        self._quadratic = quadratic = Quadratic(self)
        return quadratic

    @property
    @abc.abstractmethod
    def num_interactions(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def offset(self) -> Bias:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def variables(self) -> Collection[Variable]:
        raise NotImplementedError

    @abc.abstractmethod
    def add_linear(self, v: Variable, bias: Bias):
        raise NotImplementedError

    @abc.abstractmethod
    def add_quadratic(self, u: Variable, v: Variable, bias: Bias):
        raise NotImplementedError

    @abc.abstractmethod
    def degree(self, v: Variable) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def get_linear(self, v: Variable) -> Bias:
        raise NotImplementedError

    @abc.abstractmethod
    def get_quadratic(self, u: Variable, v: Variable) -> Bias:
        raise NotImplementedError

    @abc.abstractmethod
    def iter_neighborhood(self, v: Variable) -> Iterator[Tuple[Variable, Bias]]:
        raise NotImplementedError

    @abc.abstractmethod
    def iter_quadratic(self) -> Iterator[Tuple[Variable, Variable, Bias]]:
        raise NotImplementedError

    @abc.abstractmethod
    def reduce_linear(self, function: Callable, initializer: Optional[Bias] = None) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def reduce_neighborhood(self, v: Variable, function: Callable,
                            initializer: Optional[Bias] = None) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def reduce_quadratic(self, function: Callable, initializer: Optional[Bias] = None) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def remove_interaction(self, u: Variable, v: Variable):
        raise NotImplementedError

    @abc.abstractmethod
    def remove_variable(self, v: Variable) -> Variable:
        raise NotImplementedError

    @abc.abstractmethod
    def set_linear(self, v: Variable, bias: Bias):
        raise NotImplementedError

    @abc.abstractmethod
    def set_quadratic(self, u: Variable, v: Variable, bias: Bias):
        raise NotImplementedError

    def add_linear_from(self, linear: Union[Mapping[Variable, Bias],
                                            Iterable[Tuple[Variable, Bias]]]):
        """Add variables and linear biases to a binary quadratic model.

        Args:
            linear:
                Variables and their associated linear biases, as either a dict of
                form ``{v: bias, ...}`` or an iterable of ``(v, bias)`` pairs,
                where ``v`` is a variable and ``bias`` is its associated linear
                bias.

        """
        if isinstance(linear, collections.abc.Mapping):
            iterator = linear.items()
        elif isinstance(linear, collections.abc.Iterable):
            iterator = linear
        else:
            raise TypeError(
                "expected 'linear' to be a dict or an iterable of 2-tuples.")

        for v, bias in iterator:
            self.add_linear(v, bias)

    add_variables_from = add_linear_from
    """Alias for :meth:`add_linear_from`."""

    def add_quadratic_from(self, quadratic: Union[Mapping[Tuple[Variable, Variable], Bias],
                                                  Iterable[Tuple[Variable, Variable, Bias]]]):
        """Add variables and quadratic biases to a binary quadratic model.

        Args:
            quadratic:
                Variables and their associated quadratic biases, as either a dict of
                form ``{u, v: bias, ...}`` or an iterable of ``(u, v, bias)`` triple,
                where ``u``, ``v`` are variables and ``bias`` is its associated
                quadratic bias.

        """
        if isinstance(quadratic, collections.abc.Mapping):            
            for (u, v), bias in quadratic.items():
                self.add_quadratic(u, v, bias)
        elif isinstance(quadratic, collections.abc.Iterable):
            for u, v, bias in quadratic:
                self.add_quadratic(u, v, bias)
        else:
            raise TypeError(
                "expected 'quadratic' to be a dict or an iterable of 3-tuples.")


    def fix_variable(self, v: Variable, value: float):
        """Remove a variable by fixing its value.

        Args:
            v: Variable to be fixed.

            value: Value assigned to the variable. Values should generally
                match the :class:`.Vartype` of the variable, but do not have
                to.

        Raises:
            ValueError: If ``v`` is not a variable in the model.

        """
        add_linear = self.add_linear
        for u, bias in self.iter_neighborhood(v):
            add_linear(u, value*bias)

        self.offset += value*self.get_linear(v)
        self.remove_variable(v)

    def fix_variables(self,
                      fixed: Union[Mapping[Variable, float], Iterable[Tuple[Variable, float]]]):
        """Fix the value of the variables and remove them.

        Args:
            fixed: A dictionary or an iterable of 2-tuples of variable
                assignments.

        """
        if isinstance(fixed, Mapping):
            fixed = fixed.items()

        fix_variable = self.fix_variable
        for v, val in fixed:
            fix_variable(v, val)

    def iter_linear(self) -> Iterator[Tuple[Variable, Bias]]:
        """Iterate over the variables and their biases."""
        get = self.get_linear
        for v in self.variables:
            yield v, get(v)

    def to_polystring(self, encoder: Optional[Callable[[Variable], str]] = None) -> str:
        """Return a string representing the model as a polynomial.

        Args:
            encoder: A function mapping variables to a string. By default
                string variables are mapped directly whereas all other types
                are mapped to a string :code:`f"v{variable!r}"`.

        Returns:
            A string representing the binary quadratic model.

        Examples:

            >>> x, y, z = dimod.Binaries(['x', 'y', 'z'])
            >>> (2*x + 3*y*z + 6).to_polystring()
            '6 + 2*x + 3*y*z'

        """

        if encoder is None:
            def encoder(v: Variable) -> str:
                return v if isinstance(v, str) else f"v{v!r}"

        # developer note: we use floats everywhere because they have some nice
        # methods and because for this method we're not too worried about
        # performance

        def neg(bias: float) -> bool: return bias < 0

        def string(bias: float) -> str:
            return repr(abs(int(bias))) if bias.is_integer() else repr(abs(float(bias)))

        def coefficient(bias: float) -> str:
            return '' if abs(bias) == 1 else f"{string(bias)}*"

        # linear variables that have a positive bias or are not going to be
        # included implicitly in quadratic
        linear = ((v, float(bias)) for v, bias in self.iter_linear()
                  if bias or not any(bias for _, bias in self.iter_neighborhood(v)))

        # non-zero quadratic biases
        quadratic = ((u, v, float(bias)) for u, v, bias in self.iter_quadratic() if bias)

        # offset
        offset = float(self.offset)

        sio = io.StringIO()

        # the first element is special, since we put the sign adjacent to it so
        # let's handle that case
        if offset or not len(self.variables):
            if neg(offset):
                sio.write('-')
            sio.write(string(offset))

        else:
            # the first element can come from quadratic or linear

            try:
                v, bias = next(linear)
            except StopIteration:
                # we are guaranteed that this exists
                u, v, bias = next(quadratic)
                if neg(bias):
                    sio.write('-')
                sio.write(f"{coefficient(bias)}{encoder(v)}*{encoder(u)}")
            else:
                # there is a linear bias
                if neg(bias):
                    sio.write('-')
                sio.write(f'{coefficient(bias)}{encoder(v)}')

        for v, bias in linear:
            sio.write(f" {'-' if neg(bias) else '+'} {coefficient(bias)}{encoder(v)}")

        for u, v, bias in quadratic:
            sio.write(f" {'-' if neg(bias) else '+'} {coefficient(bias)}{encoder(v)}*{encoder(u)}")

        sio.seek(0)
        return sio.read()
