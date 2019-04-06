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

import numpy as np

from six.moves import zip

from dimod.variables import Variables


class SampleView(abc.Mapping):
    __slots__ = '_variables', '_data'

    def __init__(self, data, variables):
        self._variables = variables
        self._data = data

    def __getitem__(self, v):
        return self._data[self._variables.index(v)]

    def __iter__(self):
        return iter(self._variables)

    def __len__(self):
        return len(self._variables)

    def __repr__(self):
        return str(dict(self))

    def values(self):
        return IndexValuesView(self)

    def items(self):
        return IndexItemsView(self)


class IndexItemsView(abc.ItemsView):
    """Faster read access to the numpy array"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return zip(self._mapping._variables, self._mapping._data.flat)


class IndexValuesView(abc.ValuesView):
    """Faster read access to the numpy array"""
    __slots__ = ()

    def __iter__(self):
        # Inherited __init__ puts the Mapping into self._mapping
        return iter(self._mapping._data.flat)


# developer note: Iterator functionality is deprecated
class SamplesArray(abc.Sequence, abc.Iterator):
    __slots__ = ('_samples', '_variables',
                 '_itercount')  # used for deprecated iteration feature

    def __init__(self, samples, variables):
        self._samples = samples

        if isinstance(variables, Variables):
            # we will be treating this as immutable so we don't need to
            # recreate it
            self._variables = variables
        else:
            self._variables = Variables(variables)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            # multiindex
            try:
                row, col = index
            except ValueError:
                raise IndexError("too many indices")

            return self._getmultiindex(row, col)

        elif isinstance(index, int):
            # single row
            return SampleView(self._samples[index, :], self._variables)

        else:
            # multiple rows
            return type(self)(self._samples[index, :], self._variables)

    def _getmultiindex(self, row, col):

        variables = self._variables
        samples = self._samples

        if col in variables:
            # single variable

            if isinstance(row, int):
                # return a single value
                return self[row][col]

            # return a vector
            return samples[row, variables.index(col)]

        # multiple variables
        try:
            index = (row, [variables.index[v] for v in col])
        except (TypeError):
            raise KeyError('{!r} is not a variable in samples'.format(col))

        if isinstance(row, (abc.Sequence, np.ndarray)):
            # we know that column is a sequence (because we just constructed it)
            # so we have triggered advanced indexing which we don't want. Use
            # ix_ to get back to basic indexing
            index = np.ix_(row, [variables.index[v] for v in col])

        return samples[index]

    def __iter__(self):
        # __iter__ is a mixin for Sequence but we can speed it up by
        # implementing it ourselves
        variables = self._variables
        for row in self._samples:
            yield SampleView(row, variables)

    def __next__(self):
        import warnings
        msg = ("SampleSet.samples() will return an iterable not an iterator in "
               "the future")
        warnings.warn(msg, DeprecationWarning)

        itercount = getattr(self, '_itercount', 0)
        if itercount < len(self):
            self._itercount = itercount + 1
            return self[itercount]
        raise StopIteration

    next = __next__  # for python2


    def __len__(self):
        return self._samples.shape[0]
