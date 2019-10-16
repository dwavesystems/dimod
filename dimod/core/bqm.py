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

try:
    from collections.abc import KeysView, Mapping, MutableMapping
except ImportError:
    from collections import KeysView, Mapping, MutableMapping

from six import add_metaclass


@add_metaclass(abc.ABCMeta)
class BQM:
    @abc.abstractmethod
    def __init__(self, obj):
        pass

    @abc.abstractproperty
    def num_interactions(self):
        """int: The number of interactions in the model."""
        pass

    @abc.abstractproperty
    def num_variables(self):
        """int: The number of variables in the model."""
        pass

    @abc.abstractmethod
    def get_linear(self, u, v):
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
    def set_linear(self, u, v):
        pass

    @abc.abstractmethod
    def set_quadratic(self, u, v, bias):
        pass

    # mixins

    def __len__(self):
        """The number of variables in the binary quadratic model."""
        return self.num_variables

    @property
    def shape(self):
        """2-tuple: (num_variables, num_interactions)."""
        return self.num_variables, self.num_interactions

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


class ShapeableBQM(BQM):
    @abc.abstractmethod
    def add_variable(self, v=None):
        """Add a variable to the binary quadratic model.

        Args:
            label (hashable, optional):
                A label for the variable. Defaults to the length of the binary
                quadratic model, if that label is available. Otherwise defaults
                to the lowest available positive integer label.

        Returns:
            hashable: The label of the added variable.

        Raises:
            TypeError: If the label is not hashable.

        """
        pass

    @abc.abstractmethod
    def pop_variable(self):
        pass

    @abc.abstractmethod
    def remove_interaction(self, u, v):
        pass
