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
import io
import operator

from collections.abc import Callable, ItemsView, Iterable, Mapping, MutableMapping
from typing import Any, Collection, Iterator, Optional, Tuple

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
            stream.write(f'{key!r}: {value!r}')
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
    def variables(self) -> Collection[Variable]:
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
