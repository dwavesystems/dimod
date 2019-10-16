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
try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from collections import OrderedDict
from numbers import Integral

import numpy as np

from dimod.core.bqm import ShapeableBQM
from dimod.vartypes import as_vartype

__all__ = ['AdjDictBQM']


class AdjDictBQM(ShapeableBQM):
    """
    """
    def __init__(self, obj=0, vartype=None):
        # we could actually have these variable but to keep this consistent
        # with the other BQMs we fix them for now
        self.dtype = dtype = np.dtype(np.double)
        self.index_dtype = np.dtype(np.uintc)  # this is unused

        # we use ordered dict because the other BQM types are ordered. However
        # collection's OrderedDict is optimized for both FILO and FIFO so has
        # a larger memory footprint than we need. We could instead store a list
        # with the dict which would be more performant but complicate the
        # implementation
        self._adj = adj = OrderedDict()

        # handle the case where only vartype is given
        if vartype is None:
            try:
                vartype = obj.vartype
            except AttributeError:
                vartype = obj
                obj = 0
        self._vartype = as_vartype(vartype)

        if isinstance(obj, Integral):
            adj.update((v, {v: dtype.type(0)}) for v in range(obj))
        elif isinstance(obj, tuple):
            if len(obj) == 2:
                linear, quadratic = obj
            elif len(obj) == 3:
                linear, quadratic, self.offset = obj
            else:
                raise ValueError()

            if isinstance(linear, abc.Mapping):
                for var, b in linear.items():
                    self.set_linear(var, b)
            else:
                raise NotImplementedError

            if isinstance(quadratic, abc.Mapping):
                for (uvar, var), b in quadratic.items():
                    self.set_quadratic(uvar, var, b)
            else:
                raise NotImplementedError
        else:
            # assume it's dense

            D = np.atleast_2d(np.asarray(obj, dtype=self.dtype))

            num_variables = D.shape[0]

            if D.ndim != 2 or num_variables != D.shape[1]:
                raise ValueError("expected dense to be a 2 dim square array")

            adj.update((v, {v: dtype.type(0)}) for v in range(num_variables))

            it = np.nditer(D, flags=['multi_index'])
            while not it.finished:
                u, v = it.multi_index
                bias = dtype.type(it.value)

                if bias and not np.isnan(bias):
                    if u in adj[v]:
                        adj[u][v] = adj[v][u] = adj[u][v] + bias
                    else:
                        adj[u][v] = adj[v][u] = bias

                it.iternext()

    @property
    def num_variables(self):
        """int: The number of variables in the model."""
        return len(self._adj)

    @property
    def num_interactions(self):
        """int: The number of interactions in the model."""
        return (sum(map(len, self._adj.values())) - len(self._adj)) // 2

    @property
    def offset(self):
        try:
            return self._offset
        except AttributeError:
            pass
        self.offset = 0  # type coersion etc
        return self.offset

    @offset.setter
    def offset(self, offset):
        self._offset = self.dtype.type(offset)

    @property
    def vartype(self):
        """:class:`.Vartype`: The vartype of the binary quadratic model. One of
        :class:`.Vartype.SPIN` or :class:`.Vartype.BINARY`.
        """
        # so it's readonly
        return self._vartype

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

        Examples:

            >>> bqm = dimod.AdjDictBQM('SPIN')
            >>> bqm.add_variable()
            0
            >>> bqm.add_variable('a')
            'a'
            >>> bqm.add_variable()
            2

            >>> bqm = dimod.AdjDictBQM('SPIN')
            >>> bqm.add_variable(1)
            1
            >>> bqm.add_variable()  # 1 is taken
            0
            >>> bqm.add_variable()
            2

        """
        if v is None:

            v = self.num_variables

            if self.has_variable(v):
                # if v already exists there must be a smaller one that's free
                for v in range(self.num_variables):
                    if not self.has_variable(v):
                        break

        self._adj.setdefault(v, {v: self.dtype.type(0)})
        return v

    def get_linear(self, v):
        """Get the linear bias of v.

        Args:
            v (hashable):
                A variable in the binary quadratic model.

        Returns:
            float: The linear bias of v.

        Raises:
            ValueError: If v is not a variable in the binary quadratic model.

        """
        try:
            return self._adj[v][v]
        except KeyError:
            pass
        raise ValueError("{} is not a variable".format(v))

    def get_quadratic(self, u, v):
        """Get the quadratic bias of (u, v).

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

        Returns:
            float: The quadratic bias of (u, v).

        Raises:
            ValueError: If either u or v is not a variable in the binary
            quadratic model, if u == v or if (u, v) is not an interaction in
            the binary quadratic model.

        """
        if u == v:
            raise ValueError("No interaction between {} and itself".format(u))
        try:
            return self._adj[u][v]
        except KeyError:
            pass
        raise ValueError('No interaction between {} and {}'.format(u, v))

    def iter_linear(self):
        for u, neighborhood in self._adj.items():
            yield u, neighborhood[u]  # u and it's bias

    def iter_quadratic(self, variables=None):
        if variables is None:
            variables = self._adj
        elif self.has_variable(variables):
            variables = [variables]

        seen = set()
        for u in variables:
            neighborhood = self._adj[u]
            seen.add(u)  # also avoids self-loops
            for v, bias in neighborhood.items():
                if v not in seen:
                    yield (u, v, bias)

    def pop_variable(self):
        """Remove a variable from the binary quadratic model.

        Returns:
            hashable: The last variable added to the binary quadratic model.

        Raises:
            ValueError: If the binary quadratic model is empty.

        """
        if len(self._adj) == 0:
            raise ValueError("pop from empty binary quadratic model")

        v, neighborhood = self._adj.popitem(last=True)

        for u in neighborhood:
            if u != v:
                self._adj[u].pop(v)

        return v

    def remove_interaction(self, u, v):
        """Remove the interaction between variables u and v.

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

        Returns:
            bool: If there was an interaction to remove.

        Raises:
            ValueError: If either u or v is not a variable in the binary
            quadratic model.

        """
        try:
            self._adj[u].pop(v)
        except KeyError:
            # nothing pop
            return False

        # want to raise an exception in the case that they got out of sync
        # as a sanity check
        self._adj[v].pop(u)

        return True

    def set_linear(self, v, bias):
        """Set the linear biase of a variable v.

        Args:
            v (hashable):
                A variable in the binary quadratic model. It is added if not
                already in the model.

            b (numeric):
                The linear bias of v.

        Raises:
            TypeError: If v is not hashable

        """
        bias = self.dtype.type(bias)  # convert to the appropriate dtype

        if v in self._adj:
            self._adj[v][v] = bias
        else:
            self._adj[v] = {v: bias}

    def set_quadratic(self, u, v, bias):
        """Set the quadratic bias of (u, v).

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

            b (numeric):
                The linear bias of v.

        Raises:
            TypeError: If u or v is not hashable.

        """
        bias = self.dtype.type(bias)  # convert to the appropriate dtype

        # make sure the variables exist
        self.add_variable(u)
        self.add_variable(v)

        adj = self._adj
        adj[u][v] = adj[v][u] = bias
