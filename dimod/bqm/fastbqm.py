from __future__ import division

from itertools import chain
from numbers import Number

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import numpy as np

from dimod.bqm.vectorbqm import VectorBQM
from dimod.sampleset import as_samples
from dimod.views import Variables, LinearView, QuadraticView, AdjacencyView
from dimod.vartypes import Vartype

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
        self.adj = AdjacencyView(variables, self.iadj, self.qdata)
        self.quadratic = QuadraticView(self)

    def __contains__(self, v):
        return v in self.variables

    def __iter__(self):
        return iter(self.variables)

    def __repr__(self):
        return "{}({}, {}, {}, '{}', dtype='{}', index_dtype='{}')".format(
            self.__class__.__name__,
            self.linear,
            self.quadratic,
            self.offset,
            self.vartype.name,
            np.dtype(self.dtype).name, np.dtype(self.index_dtype).name)

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

    ##########################################################################
    # In-place
    ##########################################################################

    def scale(self, scalar, ignored_variables=None, ignored_interactions=None):
        """Multiply by the specified scalar all the biases and offset of a binary quadratic model.

        Args:
            scalar (number):
                Value by which to scale the energy range of the binary quadratic model.

        Examples:

            This example creates a binary quadratic model and then scales it to half
            the original energy range.

            >>> bqm = dimod.FastBQM({'a': -2.0, 'b': 2.0}, {('a', 'b'): -1.0}, 1.0, dimod.SPIN)
            >>> bqm.scale(0.5)
            >>> bqm.linear['a']
            -1.0
            >>> bqm.quadratic[('a', 'b')]
            -0.5
            >>> bqm.offset
            0.5

        """
        variables = self.variables

        ldata = self.ldata
        qdata = self.qdata

        if ignored_variables is None:
            ldata *= scalar
        else:
            idx = np.ones(len(self.linear), dtype=bool)
            idx[[variables.index(v) for v in ignored_variables]] = False
            ldata[idx] *= scalar

        if ignored_interactions is None:
            qdata *= scalar
        else:
            iadj = self.iadj
            idx = np.ones(len(self.quadratic), dtype=bool)
            idx[[iadj[variables.index(u)][variables.index(v)] for u, v in ignored_interactions]] = False
            qdata[idx] *= scalar

        self.offset *= scalar

    def normalize(self, bias_range=1, quadratic_range=None):
        """Normalizes the biases of the binary quadratic model such that they
        fall in the provided range(s), and adjusts the offset appropriately.

        If `quadratic_range` is provided, then `bias_range` will be treated as
        the range for the linear biases and `quadratic_range` will be used for
        the range of the quadratic biases.

        Args:
            bias_range (number/pair):
                Value/range by which to normalize the all the biases, or if
                `quadratic_range` is provided, just the linear biases.

            quadratic_range (number/pair):
                Value/range by which to normalize the quadratic biases.

        Examples:

            This example creates a binary quadratic model and then normalizes
            all the biases in the range [-0.4, 0.8].

            >>> bqm = dimod.FastBQM({'a': -2.0, 'b': 1.5}, {('a', 'b'): -1.0}, 1.0, dimod.SPIN)
            >>> bqm.normalize([-0.4, 0.8])
            >>> bqm.linear
            {'a': -0.4, 'b': 0.30000000000000004}
            >>> bqm.quadratic
            {('a', 'b'): -0.2}
            >>> bqm.offset
            0.2

        """

        def parse_range(r):
            if isinstance(r, Number):
                return -abs(r), abs(r)
            return r

        def min_and_max(iterable):
            if not iterable:
                return 0, 0
            return min(iterable), max(iterable)

        if quadratic_range is None:
            linear_range, quadratic_range = bias_range, bias_range
        else:
            linear_range = bias_range

        lin_range, quad_range = map(parse_range, (linear_range,
                                                  quadratic_range))

        lin_min, lin_max = min_and_max(self.linear.values())
        quad_min, quad_max = min_and_max(self.quadratic.values())

        inv_scalar = max(lin_min / lin_range[0], lin_max / lin_range[1],
                         quad_min / quad_range[0], quad_max / quad_range[1])

        if inv_scalar != 0:
            self.scale(1 / inv_scalar)

    def flip_variable(self, v):
        """Flip variable v in a binary quadratic model.

        Args:
            v (variable):
                Variable in the binary quadratic model. If v is not in the binary
                quadratic model, it is ignored.

        Examples:
            This example creates a binary quadratic model with two variables and inverts
            the value of one.

            >>> bqm = dimod.FastBQM({1: 1, 2: 2}, {(1, 2): 0.5}, 0.5, dimod.SPIN)
            >>> bqm.flip_variable(1)
            >>> bqm.linear[1], bqm.linear[2], bqm.quadratic[(1, 2)]
            (-1.0, 2, -0.5)

        """
        self.flip_variables([v])

    def flip_variables(self, variables):
        adj = self.adj
        linear = self.linear

        variables = set(v for v in variables if v in self)

        if self.vartype is Vartype.SPIN:
            for v in variables:
                linear[v] *= -1
                for u in adj[v]:
                    adj[u][v] *= -1
        else:
            for v in variables:

                self.offset += linear[v]
                linear[v] *= -1

                for u in adj[v]:
                    bias = adj[v][u]

                    adj[u][v] *= -1.

                    linear[u] += bias

    def relabel_variables(self, mapping):
        """Relabel variables of a binary quadratic model as specified by mapping.

        Args:
            mapping (dict):
                Dict mapping current variable labels to new ones. If an incomplete mapping is
                provided, unmapped variables retain their current labels.

        """
        self.variables.relabel(mapping)

    ##########################################################################
    # Create new bqms
    ##########################################################################

    def copy(self):
        return self.__class__(self.ldata,
                              (self.irow, self.icol, self.qdata),
                              self.offset,
                              self.vartype,
                              dtype=self.dtype, index_dtype=self.index_dtype,
                              labels=self.variables)

    def fix_variable(self, v, val):
        return self.fix_variables({v: val})

    def fix_variables(self, assignments):
        if not isinstance(assignments, abc.Mapping):
            raise TypeError("assignments should be a mapping")

        # dev note: it might be faster to do this in numpy in some cases

        linear = self.linear
        quadratic = self.quadratic
        adj = self.adj
        vartype = self.vartype

        fixed_linear = {v: bias for v, bias in linear.items() if v not in assignments}

        fixed_quadratic = {(u, v): bias for (u, v), bias in quadratic.items()
                           if u not in assignments and v not in assignments}

        fixed_offset = self.offset

        for v, val in assignments.items():
            if val not in vartype.value:
                raise ValueError("expected value to be in {}, received {} instead".format(vartype.value, val))

            fixed_offset += linear[v] * val

            for u, bias in adj[v].items():
                if u in assignments:
                    # becomes an offset, we also do this twice
                    fixed_offset += val * assignments[u] * bias / 2
                else:
                    fixed_linear[u] += bias * val

        return self.__class__(fixed_linear,
                              fixed_quadratic,
                              fixed_offset,
                              vartype,
                              dtype=self.dtype, index_dtype=self.index_dtype)

    ##########################################################################
    # Methods
    ##########################################################################

    def energy(self, samples_like, _use_cpp_ext=True):
        energies = self.energies(samples_like, _use_cpp_ext=_use_cpp_ext)

        if len(energies) != 1:
            raise ValueError("too many samples given, use 'energies' method instead")

        return energies[0]

    def energies(self, samples_like, _use_cpp_ext=True):
        samples, labels = as_samples(samples_like)

        variables = self.variables
        if labels == variables:
            order = None
        else:
            order = [variables.index(v) for v in labels]

        return VectorBQM.energies(self, samples, order=order, _use_cpp_ext=_use_cpp_ext)


FastBQM = FastBinaryQuadraticModel
