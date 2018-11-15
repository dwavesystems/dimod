from itertools import chain

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import numpy as np

from dimod.bqm.vectorbqm import VectorBQM
from dimod.sampleset import as_samples
from dimod.views import Variables, LinearView, QuadraticView, AdjacencyView


__all__ = 'FastBinaryQuadraticModel', 'FastBQM'


class FastBinaryQuadraticModel(VectorBQM, abc.Iterable, abc.Container):
    __slots__ = 'variables', 'linear', 'quadratic', 'adj'

    def __init__(self, linear, quadratic, offset, vartype,
                 dtype=np.float, index_dtype=np.int64,
                 labels=None):

        # get the labels

        if labels is None:
            # if labels are none, we derive from linear/quadratic and set the labels last

            if isinstance(linear, abc.Mapping):
                linear_labels = linear.keys()
            else:
                linear_labels = range(len(linear))

            if isinstance(quadratic, abc.Mapping):
                quadratic_labels = (v for interaction in quadratic.keys() for v in interaction)
            elif isinstance(quadratic, tuple) and len(quadratic) == 3:
                row, col, _ = quadratic
                try:
                    quadratic_labels = range(max(max(row), max(col)) + 1)
                except ValueError:
                    # if row/col are empty
                    quadratic_labels = []
            else:
                # assume dense
                quadratic = np.atleast_2d(np.asarray(quadratic, dtype=dtype))

                quadratic_labels = range(quadratic.shape[1])

            labels = chain(linear_labels, quadratic_labels)

        self.variables = variables = Variables(labels)

        # index-label if appropriate

        if isinstance(linear, abc.Mapping):
            # dev note: it would be nice if we didn't need to make this intermediate copy
            linear = [linear.get(v, 0) for v in variables]

        if isinstance(quadratic, abc.Mapping):
            if quadratic:
                quadratic = tuple(zip(*((variables.index(u), variables.index(v), bias)
                                      for (u, v), bias in quadratic.items())))
            else:
                quadratic = [], [], []

        super(FastBQM, self).__init__(linear, quadratic, offset, vartype,
                                      dtype=dtype, index_dtype=index_dtype)

        self.linear = LinearView(variables, self.ldata)
        self.adj = adj = AdjacencyView(variables, self.iadj)
        self.quadratic = QuadraticView(self)

    def __contains__(self, v):
        return v in self.variables

    def __iter__(self):
        return iter(self.variables)

    def __repr__(self):
        raise NotImplementedError

    def __eq__(self, other):
        if self.vartype is not other.vartype:
            return False

        if self.offset != other.offset:
            return False

        if self.linear != other.linear:
            return False

        return self.adj == other.adj

    def __ne__(self, other):
        return not (self == other)


FastBQM = FastBinaryQuadraticModel
