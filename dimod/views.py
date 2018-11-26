# Copyright 2018 D-Wave Systems Inc.
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
# ================================================================================================
try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import numpy as np

from six.moves import zip

from dimod.variables import Variables


class VariableArrayView(abc.Mapping):
    """Create a mapping out of :class:`dimod.views.Variables' and :obj:`numpy.ndarray`."""
    __slots__ = '_variables', '_data'

    def __init__(self, variables, data):

        if not isinstance(variables, Variables):
            raise TypeError("variables should be a Variables object")
        if not isinstance(data, np.ndarray):
            raise TypeError("data should be a numpy 1 dimensional array")
        if data.ndim != 1:
            raise ValueError("data should be a numpy 1 dimensional array")
        if len(variables) != len(data):
            raise ValueError("variables and data should match length")

        self._variables = variables
        self._data = data

    def __getitem__(self, v):
        return self._data[self._variables.index(v)]

    def __iter__(self):
        return iter(self._variables)

    def __len__(self):
        return len(self._variables)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self._variables, self._data)

    def __str__(self):
        return str(dict(self))

    def values(self):
        return ArrayValuesView(self)

    def items(self):
        return ArrayItemsView(self)


class ArrayValuesView(abc.ValuesView):
    """Faster read access to the numpy array"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return iter(self._mapping._data.flat)


class ArrayItemsView(abc.ItemsView):
    """Faster read access to the numpy array"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return zip(self._mapping._variables, self._mapping._data.flat)


class SampleView(VariableArrayView):
    """View each row of the samples record as if it was a dict."""
    __slots__ = ()

    def __repr__(self):
        return str(self)
