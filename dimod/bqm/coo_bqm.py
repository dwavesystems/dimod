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
from itertools import chain

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import numpy as np

from dimod.bqm.cyutils import coo_sort
from dimod.decorators import vartype_argument
from dimod.variables import Variables
from dimod.vartypes import SPIN

__all__ = ['CooBinaryQuadraticModel',
           'CooBQM',
           ]


class Flags(object):
    # todo: look into enum.Flag
    def __init__(self, bqm):
        self.bqm = bqm

        self.sorted = None  # unknown, falsy


class ArrayLinearView(abc.Mapping):
    def __init__(self, bqm):
        self.bqm = bqm

    def __getitem__(self, v):
        return self.bqm.ldata[self.bqm.variables.index[v]]

    def __iter__(self):
        return iter(self.bqm.variables)

    def __len__(self):
        return len(self.bqm.variables)


class CooQuadraticView(abc.Mapping):
    def __init__(self, bqm):
        self.bqm = bqm

    def __getitem__(self, interaction):
        bqm = self.bqm
        irow = bqm.irow
        icol = bqm.icol
        qdata = bqm.qdata

        iu, iv = map(bqm.variables.index, interaction)

        if bqm.flags.sorted:
            # O(log(|E|))
            if iu > iv:
                iu, iv = iv, iu

            left = np.searchsorted(irow, iu, side='left')

            # we could search a smaller window for the right bound by using the
            # number of variables in the BQM but I am not sure if we want to
            # assume that length of bqm.linear exists
            off = np.searchsorted(irow[left:], iu, side='right')

            idx = left + np.searchsorted(icol[left:left+off], iv, side='left')

            if irow[idx] == iu and icol[idx] == iv:
                return qdata[idx]
        else:
            # O(|E|)
            # if we cythonized it we could avoid making the mask which would
            # save on memory. Creating the mask is faster than iterating in
            # python with a list
            mask = ((irow == iu) & (icol == iv)) ^ ((irow == iv) & (icol == iu))
            if np.any(mask):
                return np.sum(qdata[mask])

        raise KeyError

    def __iter__(self):
        # note: duplicates will appear if not de-duplicated
        bqm = self.bqm
        variables = bqm.variables
        for iu, iv in zip(bqm.irow, bqm.icol):
            yield variables[iu], variables[iv]

    def __len__(self):
        return len(self.bqm.qdata)

    def items(self):
        return CooQuadraticItemsView(self)


class CooQuadraticItemsView(abc.ItemsView):
    def __iter__(self):
        # much faster than doing the O(|E|) or O(log|E|) bias lookups each time
        bqm = self._mapping.bqm
        variables = bqm.variables
        for iu, iv, bias in zip(bqm.irow, bqm.icol, bqm.qdata):
            yield (variables[iu], variables[iv]), bias


class CooBinaryQuadraticModel(object):

    @vartype_argument('vartype')
    def __init__(self, linear, quadratic, offset, vartype,
                 variables=None,
                 dtype=np.float, index_dtype=np.int,
                 copy=True, sort_and_reduce=False):

        self.flags = Flags(self)

        self.dtype = dtype = np.dtype(dtype)
        self.index_dtype = index_dtype = np.dtype(index_dtype)

        # linear
        self.ldata = ldata = np.array(linear, dtype=dtype, copy=copy)

        # quadratic
        irow, icol, qdata = quadratic
        irow = np.array(irow, dtype=index_dtype, copy=copy)
        icol = np.array(icol, dtype=index_dtype, copy=copy)
        qdata = np.array(qdata, dtype=dtype, copy=copy)

        # todo: check that they are all 1dim, same length etc

        if sort_and_reduce:
            coo_sort(irow, icol, qdata)  # in-place

            # reduce unique
            unique = ((irow[1:] != icol[:-1]) | (icol[1:] != icol[:-1]))
            if not unique.all():
                # copy
                unique = np.append(True, unique)

                irow = irow[unique]
                icol = icol[unique]

                unique_idxs, = np.nonzero(unique)
                qdata = np.add.reduceat(qdata, unique_idxs, dtype=dtype)

            self.flags.sorted = True

        self.irow = irow
        self.icol = icol
        self.qdata = qdata

        # offset
        self.offset = dtype.type(offset)

        # vartype
        self.vartype = vartype

        # variables
        if not copy and isinstance(variables, Variables):
            self.variables = variables
        else:
            self.variables = Variables(variables)

        # views
        self.linear = ArrayLinearView(self)
        self.quadratic = CooQuadraticView(self)

    @classmethod
    def from_ising(cls, h, J, offset=0,
                   dtype=np.float, index_dtype=np.int):

        # get all of the labels
        variables = Variables(chain(h, chain(*J)))

        # get the quadratic
        num_interactions = len(J)

        irow = np.empty(num_interactions, dtype=index_dtype)
        icol = np.empty(num_interactions, dtype=index_dtype)
        qdata = np.empty(num_interactions, dtype=dtype)

        # we could probably speed this up with cython
        for idx, ((u, v), bias) in enumerate(J.items()):
            irow[idx] = variables.index[u]
            icol[idx] = variables.index[v]
            qdata[idx] = bias

        # next linear
        ldata = np.fromiter((h.get(v, 0) for v in variables),
                            count=len(variables), dtype=dtype)

        return cls(ldata, (irow, icol, qdata), offset, SPIN,
                   variables=variables,
                   dtype=dtype, index_dtype=index_dtype,
                   sort_and_reduce=True,
                   copy=False)  # we just made these objects so no need to copy


CooBQM = CooBinaryQuadraticModel
