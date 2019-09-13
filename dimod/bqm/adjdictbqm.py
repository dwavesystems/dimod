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

__all__ = ['AdjDictBQM']


class AdjDictBQM(ShapeableBQM):
    def __init__(self, obj=0):
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

        if isinstance(obj, Integral):
            adj.update((v, {v: dtype.type(0)}) for v in range(obj))
        elif isinstance(obj, tuple):
            if len(obj) == 2:
                linear, quadratic = obj
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
                bias = it.value

                if bias and not np.isnan(bias):
                    if u in adj[v]:
                        adj[u][v] = adj[v][u] = adj[u][v] + bias
                    else:
                        adj[u][v] = adj[v][u] = bias

                it.iternext()

    @property
    def num_variables(self):
        return len(self._adj)

    @property
    def num_interactions(self):
        return (sum(map(len, self._adj.values())) - len(self._adj)) // 2

    def add_variable(self, v=None):
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
        try:
            return self._adj[v][v]
        except KeyError:
            pass
        raise ValueError("{} is not a variable".format(v))

    def get_quadratic(self, u, v):
        if u != v:
            try:
                return self._adj[u][v]
            except KeyError:
                pass
        raise ValueError('No interaction between {} and {}'.format(u, v))

    def iter_variables(self):
        return iter(self._adj)

    def pop_variable(self):
        if len(self._adj) == 0:
            raise ValueError("pop from empty binary quadratic model")

        v, neighbourhood = self._adj.popitem(last=True)

        for u in neighbourhood:
            if u != v:
                self._adj[u].pop(v)

        return v

    def remove_interaction(self, u, v):
        try:
            self._adj[u].pop(v)
        except KeyError:
            # nothing pop
            return False
        else:
            # want to raise an exception in the case that they got out of sync
            # as a sanity check
            self._adj[v].pop(u)

        return True

    def set_linear(self, v, bias):
        bias = self.dtype.type(bias)  # convert to the appropriate dtype

        if v in self._adj:
            self._adj[v][v] = bias
        else:
            self._adj[v] = {v: bias}

    def set_quadratic(self, u, v, bias):
        bias = self.dtype.type(bias)  # convert to the appropriate dtype

        # make sure the variables exist
        self.add_variable(u)
        self.add_variable(v)

        adj = self._adj
        adj[u][v] = adj[v][u] = bias
