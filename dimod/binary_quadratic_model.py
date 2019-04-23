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
"""

The binary quadratic model (BQM) class contains
Ising and quadratic unconstrained binary optimization (QUBO) models
used by samplers such as the D-Wave system.

The :term:`Ising` model is an objective function of :math:`N` variables
:math:`s=[s_1,...,s_N]` corresponding to physical Ising spins, where :math:`h_i`
are the biases and :math:`J_{i,j}` the couplings (interactions) between spins.

.. math::

    \\text{Ising:} \\qquad  E(\\bf{s}|\\bf{h},\\bf{J})
    = \\left\\{ \\sum_{i=1}^N h_i s_i + \\sum_{i<j}^N J_{i,j} s_i s_j  \\right\\}
    \\qquad\\qquad s_i\\in\\{-1,+1\\}


The :term:`QUBO` model is an objective function of :math:`N` binary variables represented
as an upper-diagonal matrix :math:`Q`, where diagonal terms are the linear coefficients
and the nonzero off-diagonal terms the quadratic coefficients.

.. math::

    \\text{QUBO:} \\qquad E(\\bf{x}| \\bf{Q})  =  \\sum_{i\\le j}^N x_i Q_{i,j} x_j
    \\qquad\\qquad x_i\\in \\{0,1\\}

The :class:`.BinaryQuadraticModel` class can contain both these models and its methods provide
convenient utilities for working with, and interworking between, the two representations
of a problem.

"""
from __future__ import absolute_import, division

import itertools

try:
    import collections.abc as abc
except ImportError:
    import collections as abc

from numbers import Number

import numpy as np

from six import itervalues, iteritems

from dimod.decorators import vartype_argument
from dimod.serialization.utils import array2bytes, bytes2array
from dimod.sampleset import as_samples
from dimod.utilities import resolve_label_conflict
from dimod.views.bqm import LinearView, QuadraticView, AdjacencyView
from dimod.views.samples import SampleView
from dimod.vartypes import Vartype

__all__ = 'BinaryQuadraticModel', 'BQM'


class BinaryQuadraticModel(abc.Sized, abc.Container, abc.Iterable):
    """Encodes a binary quadratic model.

    Binary quadratic model is the superclass that contains the `Ising model`_ and the QUBO_.

    .. _Ising model: https://en.wikipedia.org/wiki/Ising_model
    .. _QUBO: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    Args:
        linear (dict[variable, bias]):
            Linear biases as a dict, where keys are the variables of
            the binary quadratic model and values the linear biases associated
            with these variables.
            A variable can be any python object that is valid as a dictionary key.
            Biases are generally numbers but this is not explicitly checked.

        quadratic (dict[(variable, variable), bias]):
            Quadratic biases as a dict, where keys are
            2-tuples of variables and values the quadratic biases associated
            with the pair of variables (the interaction).
            A variable can be any python object that is valid as a dictionary key.
            Biases are generally numbers but this is not explicitly checked.
            Interactions that are not unique are added.

        offset (number):
            Constant energy offset associated with the binary quadratic model.
            Any input type is allowed, but many applications assume that offset is a number.
            See :meth:`.BinaryQuadraticModel.energy`.

        vartype (:class:`.Vartype`/str/set):
            Variable type for the binary quadratic model. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        **kwargs:
            Any additional keyword parameters and their values are stored in
            :attr:`.BinaryQuadraticModel.info`.

    Notes:
        The :class:`.BinaryQuadraticModel` class does not enforce types on biases
        and offsets, but most applications that use this class assume that they are numeric.

    Examples:
        This example creates a binary quadratic model with three spin variables.

        >>> bqm = dimod.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
        ...                                  {(0, 1): .5, (1, 2): 1.5},
        ...                                  1.4,
        ...                                  dimod.SPIN)

        This example creates a binary quadratic model with non-numeric variables
        (variables can be any hashable object).

        >>> bqm = dimod.BinaryQuadraticModel({'a': 0.0, 'b': -1.0, 'c': 0.5},
        ...                                  {('a', 'b'): -1.0, ('b', 'c'): 1.5},
        ...                                  1.4,
        ...                                  dimod.SPIN)
        >>> len(bqm)
        3
        >>> 'b' in bqm
        True

    Attributes:
        linear (dict[variable, bias]):
            Linear biases as a dict, where keys are the variables of
            the binary quadratic model and values the linear biases associated
            with these variables.

        quadratic (dict[(variable, variable), bias]):
            Quadratic biases as a dict, where keys are 2-tuples of variables, which
            represent an interaction between the two variables, and values
            are the quadratic biases associated with the interactions.

        offset (number):
            The energy offset associated with the model. Same type as given
            on instantiation.

        vartype (:class:`.Vartype`):
            The model's type. One of :class:`.Vartype.SPIN` or :class:`.Vartype.BINARY`.

        variables (keysview):
            The variables in the binary quadratic model as a dictionary keys
            view object.

        adj (dict):
            The model's interactions as nested dicts.
            In graphic representation, where variables are nodes and interactions
            are edges or adjacencies, keys of the outer dict (`adj`) are all
            the model's nodes (e.g. `v`) and values are the inner dicts. For the
            inner dict associated with outer-key/node 'v', keys are all the nodes
            adjacent to `v` (e.g. `u`) and values are quadratic biases associated
            with the pair of inner and outer keys (`u, v`).

        info (dict):
            A place to store miscellaneous data about the binary quadratic model
            as a whole.

        SPIN (:class:`.Vartype`): An alias of :class:`.Vartype.SPIN` for easier access.

        BINARY (:class:`.Vartype`): An alias of :class:`.Vartype.BINARY` for easier access.

    Examples:
       This example creates an instance of the :class:`.BinaryQuadraticModel`
       class for the K4 complete graph, where the nodes have biases
       set equal to their sequential labels and interactions are the
       concatenations of the node pairs (e.g., 23 for u,v = 2,3).

       >>> import dimod
       ...
       >>> linear = {1: 1, 2: 2, 3: 3, 4: 4}
       >>> quadratic = {(1, 2): 12, (1, 3): 13, (1, 4): 14,
       ...              (2, 3): 23, (2, 4): 24,
       ...              (3, 4): 34}
       >>> offset = 0.0
       >>> vartype = dimod.BINARY
       >>> bqm_k4 = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)
       >>> bqm_k4.info = {'Complete K4 binary quadratic model.'}
       >>> bqm_k4.info.issubset({'Complete K3 binary quadratic model.',
       ...                       'Complete K4 binary quadratic model.',
       ...                       'Complete K5 binary quadratic model.'})
       True
       >>> bqm_k4.adj.viewitems()   # Show all adjacencies  # doctest: +SKIP
       [(1, {2: 12, 3: 13, 4: 14}),
        (2, {1: 12, 3: 23, 4: 24}),
        (3, {1: 13, 2: 23, 4: 34}),
        (4, {1: 14, 2: 24, 3: 34})]
       >>> bqm_k4.adj[2]            # Show adjacencies for node 2  # doctest: +SKIP
       {1: 12, 3: 23, 4: 24}
       >>> bqm_k4.adj[2][3]         # Show the quadratic bias for nodes 2,3 # doctest: +SKIP
       23

    """

    SPIN = Vartype.SPIN
    BINARY = Vartype.BINARY

    @vartype_argument('vartype')
    def __init__(self, linear, quadratic, offset, vartype, **kwargs):

        self._adj = {}

        self.linear = LinearView(self)
        self.quadratic = QuadraticView(self)
        self.adj = AdjacencyView(self)

        self.offset = offset  # we are agnostic to type, though generally should behave like a number
        self.vartype = vartype
        self.info = kwargs  # any additional kwargs are kept as info (metadata)

        # add linear, quadratic
        self.add_variables_from(linear)
        self.add_interactions_from(quadratic)

    @classmethod
    def empty(cls, vartype):
        """Create an empty binary quadratic model.

        Equivalent to instantiating a :class:`.BinaryQuadraticModel` with no bias values
        and zero offset for the defined :class:`vartype`:

        .. code-block:: python

            BinaryQuadraticModel({}, {}, 0.0, vartype)

        Args:
            vartype (:class:`.Vartype`/str/set):
                Variable type for the binary quadratic model. Accepted input values:

                * :attr:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :attr:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

        """
        return cls({}, {}, 0.0, vartype)

    def __repr__(self):
        return 'BinaryQuadraticModel({}, {}, {}, {})'.format(self.linear, self.quadratic, self.offset, self.vartype)

    def __eq__(self, other):
        """Model is equal if and only if linear, adj, offset and vartype are all equal."""

        try:
            if self.vartype is not other.vartype:
                return False

            if self.offset != other.offset:
                return False

            if self.linear != other.linear:
                return False

            return self.adj == other.adj
        except AttributeError:
            return False

    def __ne__(self, other):
        return not (self == other)

    def __len__(self):
        return len(self.adj)

    def __contains__(self, v):
        return v in self.adj

    def __iter__(self):
        return iter(self.adj)

    @property
    def variables(self):
        """Return binary quadratic model's variables as a dictionary view object."""
        return abc.KeysView(self.linear)

##################################################################################################
# vartype properties
##################################################################################################

    @property
    def spin(self):
        """:class:`.BinaryQuadraticModel`: An instance of the Ising model subclass
        of the :class:`.BinaryQuadraticModel` superclass, corresponding to
        a binary quadratic model with spins as its variables.

        Enables access to biases for the spin-valued binary quadratic model
        regardless of the :class:`vartype` set when the model was created.
        If the model was created with the :attr:`.binary` vartype,
        the Ising model subclass is instantiated upon the first use of the
        :attr:`.spin` property and used in any subsequent reads.

        Examples:
            This example creates a QUBO model and uses the :attr:`.spin` property
            to instantiate the corresponding Ising model.

            >>> import dimod
            ...
            >>> bqm_qubo = dimod.BinaryQuadraticModel({0: -1, 1: -1}, {(0, 1): 2}, 0.0, dimod.BINARY)
            >>> bqm_spin = bqm_qubo.spin
            >>> bqm_spin   # doctest: +SKIP
            BinaryQuadraticModel({0: 0.0, 1: 0.0}, {(0, 1): 0.5}, -0.5, Vartype.SPIN)
            >>> bqm_spin.spin is bqm_spin
            True

        Note:
            Methods like :meth:`.add_variable`, :meth:`.add_variables_from`,
            :meth:`.add_interaction`, etc. should only be used on the base model.

        """
        # NB: The existence of the _spin property implies that it is up to date, methods that
        # invalidate it will erase the property
        try:
            spin = self._spin
            if spin is not None:
                return spin
        except AttributeError:
            pass

        if self.vartype is Vartype.SPIN:
            self._spin = spin = self
        else:
            self._counterpart = self._spin = spin = self.change_vartype(Vartype.SPIN, inplace=False)

            # we also want to go ahead and set spin.binary to refer back to self
            spin._binary = self

        return spin

    @property
    def binary(self):
        """:class:`.BinaryQuadraticModel`: An instance of the QUBO model subclass of
        the :class:`.BinaryQuadraticModel` superclass, corresponding to a binary quadratic
        model with binary variables.

        Enables access to biases for the binary-valued binary quadratic model
        regardless of the :class:`vartype` set when the model was created. If the model
        was created with the :attr:`.spin` vartype, the QUBO model subclass is instantiated
        upon the first use of the :attr:`.binary` property and used in any subsequent reads.

        Examples:
           This example creates an Ising model and uses the :attr:`.binary` property
           to instantiate the corresponding QUBO model.

           >>> import dimod
           ...
           >>> bqm_spin = dimod.BinaryQuadraticModel({0: 0.0, 1: 0.0}, {(0, 1): 0.5}, -0.5, dimod.SPIN)
           >>> bqm_qubo = bqm_spin.binary
           >>> bqm_qubo  # doctest: +SKIP
           BinaryQuadraticModel({0: -1.0, 1: -1.0}, {(0, 1): 2.0}, 0.0, Vartype.BINARY)
           >>> bqm_qubo.binary is bqm_qubo
           True

        Note:
            Methods like :meth:`.add_variable`, :meth:`.add_variables_from`,
            :meth:`.add_interaction`, etc. should only be used on the base model.

        """
        # NB: The existence of the _binary property implies that it is up to date, methods that
        # invalidate it will erase the property
        try:
            binary = self._binary
            if binary is not None:
                return binary
        except AttributeError:
            pass

        if self.vartype is Vartype.BINARY:
            self._binary = binary = self
        else:
            self._counterpart = self._binary = binary = self.change_vartype(Vartype.BINARY, inplace=False)

            # we also want to go ahead and set binary.spin to refer back to self
            binary._spin = self

        return binary

###################################################################################################
# update methods
###################################################################################################

    def add_variable(self, v, bias, vartype=None):
        """Add variable v and/or its bias to a binary quadratic model.

        Args:
            v (variable):
                The variable to add to the model. Can be any python object
                that is a valid dict key.

            bias (bias):
                Linear bias associated with v. If v is already in the model, this value is added
                to its current linear bias. Many methods and functions expect `bias` to be a number
                but this is not explicitly checked.

            vartype (:class:`.Vartype`, optional, default=None):
                Vartype of the given bias. If None, the vartype of the binary
                quadratic model is used. Valid values are :class:`.Vartype.SPIN` or
                :class:`.Vartype.BINARY`.

        Examples:
            This example creates an Ising model with two variables, adds a third,
            and adds to the linear biases of the initial two.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 1.0}, {(0, 1): 0.5}, -0.5, dimod.SPIN)
            >>> len(bqm.linear)
            2
            >>> bqm.add_variable(2, 2.0, vartype=dimod.SPIN)        # Add a new variable
            >>> bqm.add_variable(1, 0.33, vartype=dimod.SPIN)
            >>> bqm.add_variable(0, 0.33, vartype=dimod.BINARY)     # Binary value is converted to spin value
            >>> len(bqm.linear)
            3
            >>> bqm.linear[1]
            1.33

        """

        # handle the case that a different vartype is provided
        if vartype is not None and vartype is not self.vartype:
            if self.vartype is Vartype.SPIN and vartype is Vartype.BINARY:
                # convert from binary to spin
                bias /= 2
                self.offset += bias
            elif self.vartype is Vartype.BINARY and vartype is Vartype.SPIN:
                # convert from spin to binary
                self.offset -= bias
                bias *= 2
            else:
                raise ValueError("unknown vartype")

        # we used to do this using self.linear but working directly with _adj
        # is much faster
        _adj = self._adj
        if v in _adj:
            if v in _adj[v]:
                _adj[v][v] += bias
            else:
                _adj[v][v] = bias
        else:
            _adj[v] = {v: bias}

        try:
            self._counterpart.add_variable(v, bias, vartype=self.vartype)
        except AttributeError:
            pass

    def add_variables_from(self, linear, vartype=None):
        """Add variables and/or linear biases to a binary quadratic model.

        Args:
            linear (dict[variable, bias]/iterable[(variable, bias)]):
                A collection of variables and their linear biases to add to the model.
                If a dict, keys are variables in the binary quadratic model and
                values are biases. Alternatively, an iterable of (variable, bias) pairs.
                Variables can be any python object that is a valid dict key.
                Many methods and functions expect the biases
                to be numbers but this is not explicitly checked.
                If any variable already exists in the model, its bias is added to
                the variable's current linear bias.

            vartype (:class:`.Vartype`, optional, default=None):
                Vartype of the given bias. If None, the vartype of the binary
                quadratic model is used. Valid values are :class:`.Vartype.SPIN` or
                :class:`.Vartype.BINARY`.

        Examples:
            This example creates creates an empty Ising model, adds two variables,
            and subsequently adds to the bias of the one while adding a new, third,
            variable.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
            >>> len(bqm.linear)
            0
            >>> bqm.add_variables_from({'a': .5, 'b': -1.})
            >>> 'b' in bqm
            True
            >>> bqm.add_variables_from({'b': -1., 'c': 2.0})
            >>> bqm.linear['b']
            -2.0

        """
        if isinstance(linear, abc.Mapping):
            for v, bias in iteritems(linear):
                self.add_variable(v, bias, vartype=vartype)
        else:
            try:
                for v, bias in linear:
                    self.add_variable(v, bias, vartype=vartype)
            except TypeError:
                raise TypeError("expected 'linear' to be a dict or an iterable of 2-tuples.")

    def add_interaction(self, u, v, bias, vartype=None):
        """Add an interaction and/or quadratic bias to a binary quadratic model.

        Args:
            v (variable):
                One of the pair of variables to add to the model. Can be any python object
                that is a valid dict key.

            u (variable):
                One of the pair of variables to add to the model. Can be any python object
                that is a valid dict key.

            bias (bias):
                Quadratic bias associated with u, v. If u, v is already in the model, this value
                is added to the current quadratic bias. Many methods and functions expect `bias` to
                be a number but this is not explicitly checked.

            vartype (:class:`.Vartype`, optional, default=None):
                Vartype of the given bias. If None, the vartype of the binary
                quadratic model is used. Valid values are :class:`.Vartype.SPIN` or
                :class:`.Vartype.BINARY`.

        Examples:
            This example creates an Ising model with two variables, adds a third,
            adds to the bias of the initial interaction, and creates
            a new interaction.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 1.0}, {(0, 1): 0.5}, -0.5, dimod.SPIN)
            >>> len(bqm.quadratic)
            1
            >>> bqm.add_interaction(0, 2, 2)        # Add new variable 2
            >>> bqm.add_interaction(0, 1, .25)
            >>> bqm.add_interaction(1, 2, .25, vartype=dimod.BINARY)     # Binary value is converted to spin value
            >>> len(bqm.quadratic)
            3
            >>> bqm.quadratic[(0, 1)]
            0.75

        """
        if u == v:
            raise ValueError("no self-loops allowed, therefore ({}, {}) is not an allowed interaction".format(u, v))

        _adj = self._adj

        if vartype is not None and vartype is not self.vartype:
            if self.vartype is Vartype.SPIN and vartype is Vartype.BINARY:
                # convert from binary to spin
                bias /= 4

                self.add_offset(bias)
                self.add_variable(u, bias)
                self.add_variable(v, bias)

            elif self.vartype is Vartype.BINARY and vartype is Vartype.SPIN:
                # convert from spin to binary

                self.add_offset(bias)
                self.add_variable(u, -2 * bias)
                self.add_variable(v, -2 * bias)

                bias *= 4
            else:
                raise ValueError("unknown vartype")
        else:
            # so that they exist.
            if u not in self:
                _adj[u] = {}
            if v not in self:
                _adj[v] = {}

        if u in _adj[v]:
            _adj[u][v] = _adj[v][u] = _adj[u][v] + bias
        else:
            _adj[u][v] = _adj[v][u] = bias

        try:
            self._counterpart.add_interaction(u, v, bias, vartype=self.vartype)
        except AttributeError:
            pass

    def add_interactions_from(self, quadratic, vartype=None):
        """Add interactions and/or quadratic biases to a binary quadratic model.

        Args:
            quadratic (dict[(variable, variable), bias]/iterable[(variable, variable, bias)]):
                A collection of variables that have an interaction and their quadratic
                bias to add to the model. If a dict, keys are 2-tuples of variables
                in the binary quadratic model and values are their corresponding
                bias. Alternatively, an iterable of 3-tuples. Each interaction in `quadratic` should be
                unique; that is, if `(u, v)` is a key, `(v, u)` should not be.
                Variables can be any python object that is a valid dict key.
                Many methods and functions expect the biases to be numbers but this is not
                explicitly checked.

            vartype (:class:`.Vartype`, optional, default=None):
                Vartype of the given bias. If None, the vartype of the binary
                quadratic model is used. Valid values are :class:`.Vartype.SPIN` or
                :class:`.Vartype.BINARY`.

        Examples:
            This example creates creates an empty Ising model, adds an interaction
            for two variables, adds to its bias while adding a new variable,
            then adds another interaction.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
            >>> bqm.add_interactions_from({('a', 'b'): -.5})
            >>> bqm.quadratic[('a', 'b')]
            -0.5
            >>> bqm.add_interactions_from({('a', 'b'): -.5, ('a', 'c'): 2})
            >>> bqm.add_interactions_from({('b', 'c'): 2}, vartype=dimod.BINARY)   # Binary value is converted to spin value
            >>> len(bqm.quadratic)
            3
            >>> bqm.quadratic[('a', 'b')]
            -1.0

        """
        if isinstance(quadratic, abc.Mapping):
            for (u, v), bias in iteritems(quadratic):
                self.add_interaction(u, v, bias, vartype=vartype)
        else:
            try:
                for u, v, bias in quadratic:
                    self.add_interaction(u, v, bias, vartype=vartype)
            except TypeError:
                raise TypeError("expected 'quadratic' to be a dict or an iterable of 3-tuples.")

    def remove_variable(self, v):
        """Remove variable v and all its interactions from a binary quadratic model.

        Args:
            v (variable):
                The variable to be removed from the binary quadratic model.

        Notes:
            If the specified variable is not in the binary quadratic model, this function does nothing.

        Examples:
            This example creates an Ising model and then removes one variable.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel({'a': 0.0, 'b': 1.0, 'c': 2.0},
            ...                            {('a', 'b'): 0.25, ('a','c'): 0.5, ('b','c'): 0.75},
            ...                            -0.5, dimod.SPIN)
            >>> bqm.remove_variable('a')
            >>> 'a' in bqm.linear
            False
            >>> ('b','c') in bqm.quadratic
            True

        """
        if v not in self:
            return

        adj = self.adj

        # first remove all the interactions associated with v
        while adj[v]:
            self.remove_interaction(v, next(iter(adj[v])))

        # remove the variable
        del self.linear[v]

        try:
            # invalidates counterpart
            del self._counterpart
            if self.vartype is not Vartype.BINARY and hasattr(self, '_binary'):
                del self._binary
            elif self.vartype is not Vartype.SPIN and hasattr(self, '_spin'):
                del self._spin
        except AttributeError:
            pass

    def remove_variables_from(self, variables):
        """Remove specified variables and all of their interactions from a binary quadratic model.

        Args:
            variables(iterable):
                A collection of variables to be removed from the binary quadratic model.
                If any variable is not in the model, it is ignored.

        Examples:
            This example creates an Ising model with three variables and interactions
            among all of them, and then removes two variables.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 1.0, 2: 2.0},
            ...                                  {(0, 1): 0.25, (0,2): 0.5, (1,2): 0.75},
            ...                                  -0.5, dimod.SPIN)
            >>> bqm.remove_variables_from([0, 1])
            >>> len(bqm.linear)
            1
            >>> len(bqm.quadratic)
            0

        """
        for v in variables:
            self.remove_variable(v)

    def remove_interaction(self, u, v):
        """Remove interaction of variables u, v from a binary quadratic model.

        Args:
            u (variable):
                One of the pair of variables in the binary quadratic model that
                has an interaction.

            v (variable):
                One of the pair of variables in the binary quadratic model that
                has an interaction.

        Notes:
            Any interaction not in the binary quadratic model is ignored.

        Examples:
            This example creates an Ising model with three variables that has interactions
            between two, and then removes an interaction.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1.0, ('b', 'c'): 1.0}, 0.0, dimod.SPIN)
            >>> bqm.remove_interaction('b', 'c')
            >>> ('b', 'c') in bqm.quadratic
            False
            >>> bqm.remove_interaction('a', 'c')  # not an interaction, so ignored
            >>> len(bqm.quadratic)
            1

        """

        try:
            del self.quadratic[(u, v)]
        except KeyError:
            return  # no interaction with that name

        try:
            # invalidates counterpart
            del self._counterpart
            if self.vartype is not Vartype.BINARY and hasattr(self, '_binary'):
                del self._binary
            elif self.vartype is not Vartype.SPIN and hasattr(self, '_spin'):
                del self._spin
        except AttributeError:
            pass

    def remove_interactions_from(self, interactions):
        """Remove all specified interactions from the binary quadratic model.

        Args:
            interactions (iterable[[variable, variable]]):
                A collection of interactions. Each interaction should be a 2-tuple of variables
                in the binary quadratic model.

        Notes:
            Any interaction not in the binary quadratic model is ignored.

        Examples:
            This example creates an Ising model with three variables that has interactions
            between two, and then removes an interaction.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1.0, ('b', 'c'): 1.0}, 0.0, dimod.SPIN)
            >>> bqm.remove_interactions_from([('b', 'c'), ('a', 'c')])  # ('a', 'c') is not an interaction, so ignored
            >>> len(bqm.quadratic)
            1

        """
        for u, v in interactions:
            self.remove_interaction(u, v)

    def add_offset(self, offset):
        """Add specified value to the offset of a binary quadratic model.

        Args:
            offset (number):
                Value to be added to the constant energy offset of the binary quadratic model.

        Examples:

            This example creates an Ising model with an offset of -0.5 and then
            adds to it.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 0.0}, {(0, 1): 0.5}, -0.5, dimod.SPIN)
            >>> bqm.add_offset(1.0)
            >>> bqm.offset
            0.5

        """
        self.offset += offset

        try:
            self._counterpart.add_offset(offset)
        except AttributeError:
            pass

    def remove_offset(self):
        """Set the binary quadratic model's offset to zero.

        Examples:
            This example creates an Ising model with a positive energy offset, and
            then removes it.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel({}, {}, 1.3, dimod.SPIN)
            >>> bqm.remove_offset()
            >>> bqm.offset
            0.0

        """
        self.add_offset(-self.offset)

    def scale(self, scalar, ignored_variables=None, ignored_interactions=None,
              ignore_offset=False):
        """Multiply by the specified scalar all the biases and offset of a binary quadratic model.

        Args:
            scalar (number):
                Value by which to scale the energy range of the binary quadratic model.

            ignored_variables (iterable, optional):
                Biases associated with these variables are not scaled.

            ignored_interactions (iterable[tuple], optional):
                As an iterable of 2-tuples. Biases associated with these interactions are not scaled.

            ignore_offset (bool, default=False):
                If True, the offset is not scaled.

        Examples:

            This example creates a binary quadratic model and then scales it to half
            the original energy range.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel({'a': -2.0, 'b': 2.0}, {('a', 'b'): -1.0}, 1.0, dimod.SPIN)
            >>> bqm.scale(0.5)
            >>> bqm.linear['a']
            -1.0
            >>> bqm.quadratic[('a', 'b')]
            -0.5
            >>> bqm.offset
            0.5

        """

        if ignored_variables is None:
            ignored_variables = set()
        elif not isinstance(ignored_variables, abc.Container):
            ignored_variables = set(ignored_variables)

        if ignored_interactions is None:
            ignored_interactions = set()
        elif not isinstance(ignored_interactions, abc.Container):
            ignored_interactions = set(ignored_interactions)

        linear = self.linear
        for v in linear:
            if v in ignored_variables:
                continue
            linear[v] *= scalar

        quadratic = self.quadratic
        for u, v in quadratic:
            if (u, v) in ignored_interactions or (v, u) in ignored_interactions:
                continue
            quadratic[(u, v)] *= scalar

        if not ignore_offset:
            self.offset *= scalar

        try:
            self._counterpart.scale(scalar, ignored_variables=ignored_variables,
                                    ignored_interactions=ignored_interactions)
        except AttributeError:
            pass

    def normalize(self, bias_range=1, quadratic_range=None,
                  ignored_variables=None, ignored_interactions=None,
                  ignore_offset=False):
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

            ignored_variables (iterable, optional):
                Biases associated with these variables are not scaled.

            ignored_interactions (iterable[tuple], optional):
                As an iterable of 2-tuples. Biases associated with these interactions are not scaled.

            ignore_offset (bool, default=False):
                If True, the offset is not scaled.

        Examples:

            >>> bqm = dimod.BinaryQuadraticModel({'a': -2.0, 'b': 1.5},
            ...                                  {('a', 'b'): -1.0},
            ...                                  1.0, dimod.SPIN)
            >>> max(abs(bias) for bias in bqm.linear.values())
            2.0
            >>> max(abs(bias) for bias in bqm.quadratic.values())
            1.0
            >>> bqm.normalize([-1.0, 1.0])
            >>> max(abs(bias) for bias in bqm.linear.values())
            1.0
            >>> max(abs(bias) for bias in bqm.quadratic.values())
            0.5

        """

        def parse_range(r):
            if isinstance(r, Number):
                return -abs(r), abs(r)
            return r

        def min_and_max(iterable):
            if not iterable:
                return 0, 0
            return min(iterable), max(iterable)

        if ignored_variables is None:
            ignored_variables = set()
        elif not isinstance(ignored_variables, abc.Container):
            ignored_variables = set(ignored_variables)

        if ignored_interactions is None:
            ignored_interactions = set()
        elif not isinstance(ignored_interactions, abc.Container):
            ignored_interactions = set(ignored_interactions)

        if quadratic_range is None:
            linear_range, quadratic_range = bias_range, bias_range
        else:
            linear_range = bias_range

        lin_range, quad_range = map(parse_range, (linear_range,
                                                  quadratic_range))

        lin_min, lin_max = min_and_max([v for k, v in self.linear.items()
                                        if k not in ignored_variables])
        quad_min, quad_max = min_and_max([v for (a, b), v in self.quadratic.items()
                                          if ((a, b) not in ignored_interactions
                                              and (b, a) not in
                                              ignored_interactions)])

        inv_scalar = max(lin_min / lin_range[0], lin_max / lin_range[1],
                         quad_min / quad_range[0], quad_max / quad_range[1])

        if inv_scalar != 0:
            self.scale(1 / inv_scalar, ignored_variables=ignored_variables,
                       ignored_interactions=ignored_interactions,
                       ignore_offset=ignore_offset)

    def fix_variable(self, v, value):
        """Fix the value of a variable and remove it from a binary quadratic model.

        Args:
            v (variable):
                Variable in the binary quadratic model to be fixed.

            value (int):
                Value assigned to the variable. Values must match the :class:`.Vartype` of the binary
                quadratic model.

        Examples:

            This example creates a binary quadratic model with one variable and fixes
            its value.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel({'a': -.5, 'b': 0.}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
            >>> bqm.fix_variable('a', -1)
            >>> bqm.offset
            0.5
            >>> bqm.linear['b']
            1.0
            >>> 'a' in bqm
            False

        """
        adj = self.adj
        linear = self.linear

        if value not in self.vartype.value:
            raise ValueError("expected value to be in {}, received {} instead".format(self.vartype.value, value))

        removed_interactions = []
        for u in adj[v]:
            self.add_variable(u, value * adj[v][u])
            removed_interactions.append((u, v))
        self.remove_interactions_from(removed_interactions)

        self.add_offset(value * linear[v])
        self.remove_variable(v)

    def fix_variables(self, fixed):
        """Fix the value of the variables and remove it from a binary quadratic model.

        Args:
            fixed (dict):
                A dictionary of variable assignments.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({'a': -.5, 'b': 0., 'c': 5}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
            >>> bqm.fix_variables({'a': -1, 'b': +1})

        """
        for v, val in fixed.items():
            self.fix_variable(v, val)


    def flip_variable(self, v):
        """Flip variable v in a binary quadratic model.

        Args:
            v (variable):
                Variable in the binary quadratic model. If v is not in the binary
                quadratic model, it is ignored.

        Examples:
            This example creates a binary quadratic model with two variables and inverts
            the value of one.

            >>> import dimod
            ...
            >>> bqm = dimod.BinaryQuadraticModel({1: 1, 2: 2}, {(1, 2): 0.5}, 0.5, dimod.SPIN)
            >>> bqm.flip_variable(1)
            >>> bqm.linear[1], bqm.linear[2], bqm.quadratic[(1, 2)]
            (-1.0, 2, -0.5)

        """
        adj = self.adj
        linear = self.linear
        quadratic = self.quadratic

        if v not in adj:
            return

        if self.vartype is Vartype.SPIN:
            # in this case we just multiply by -1
            linear[v] *= -1.
            for u in adj[v]:
                adj[v][u] *= -1.
                adj[u][v] *= -1.

                if (u, v) in quadratic:
                    quadratic[(u, v)] *= -1.
                elif (v, u) in quadratic:
                    quadratic[(v, u)] *= -1.
                else:
                    raise RuntimeError("quadratic is missing an interaction")

        elif self.vartype is Vartype.BINARY:
            self.offset += linear[v]
            linear[v] *= -1

            for u in adj[v]:
                bias = adj[v][u]

                adj[v][u] *= -1.
                adj[u][v] *= -1.

                linear[u] += bias

                if (u, v) in quadratic:
                    quadratic[(u, v)] *= -1.
                elif (v, u) in quadratic:
                    quadratic[(v, u)] *= -1.
                else:
                    raise RuntimeError("quadratic is missing an interaction")

        else:
            raise RuntimeError("Unexpected vartype")

        try:
            self._counterpart.flip_variable(v)
        except AttributeError:
            pass

    def update(self, bqm, ignore_info=True):
        """Update one binary quadratic model from another.

        Args:
            bqm (:class:`.BinaryQuadraticModel`):
                The updating binary quadratic model. Any variables in the updating
                model are added to the updated model. Values of biases and the offset
                in the updating model are added to the corresponding values in
                the updated model.

            ignore_info (bool, optional, default=True):
                If True, info in the given binary quadratic model is ignored, otherwise
                :attr:`.BinaryQuadraticModel.info` is updated with the given binary quadratic
                model's info, potentially overwriting values.

        Examples:
           This example creates two binary quadratic models and updates the first
           from the second.

           >>> import dimod
           ...
           >>> linear1 = {1: 1, 2: 2}
           >>> quadratic1 = {(1, 2): 12}
           >>> bqm1 = dimod.BinaryQuadraticModel(linear1, quadratic1, 0.5, dimod.SPIN)
           >>> bqm1.info = {'BQM number 1'}
           >>> linear2 = {2: 0.25, 3: 0.35}
           >>> quadratic2 = {(2, 3): 23}
           >>> bqm2 = dimod.BinaryQuadraticModel(linear2, quadratic2, 0.75, dimod.SPIN)
           >>> bqm2.info = {'BQM number 2'}
           >>> bqm1.update(bqm2)
           >>> bqm1.offset
           1.25
           >>> 'BQM number 2' in bqm1.info
           False
           >>> bqm1.update(bqm2, ignore_info=False)
           >>> 'BQM number 2' in bqm1.info
           True
           >>> bqm1.offset
           2.0

        """
        self.add_variables_from(bqm.linear, vartype=bqm.vartype)
        self.add_interactions_from(bqm.quadratic, vartype=bqm.vartype)
        self.add_offset(bqm.offset)

        if not ignore_info:
            self.info.update(bqm.info)

    def contract_variables(self, u, v):
        """Enforce u, v being the same variable in a binary quadratic model.

        The resulting variable is labeled 'u'. Values of interactions between `v` and
        variables that `u` interacts with are added to the corresponding interactions
        of `u`.

        Args:
            u (variable):
                Variable in the binary quadratic model.

            v (variable):
                Variable in the binary quadratic model.

        Examples:
           This example creates a binary quadratic model representing the K4 complete graph
           and contracts node (variable) 3 into node 2. The interactions between
           3 and its neighbors 1 and 4 are added to the corresponding interactions
           between 2 and those same neighbors.

           >>> import dimod
           ...
           >>> linear = {1: 1, 2: 2, 3: 3, 4: 4}
           >>> quadratic = {(1, 2): 12, (1, 3): 13, (1, 4): 14,
           ...              (2, 3): 23, (2, 4): 24,
           ...              (3, 4): 34}
           >>> bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.5, dimod.SPIN)
           >>> bqm.contract_variables(2, 3)
           >>> 3 in bqm.linear
           False
           >>> bqm.quadratic[(1, 2)]
           25

        """
        adj = self.adj

        if u not in adj:
            raise ValueError("{} is not a variable in the binary quadratic model".format(u))
        if v not in adj:
            raise ValueError("{} is not a variable in the binary quadratic model".format(v))

        # if there is an interaction between u, v it becomes linear for u
        if v in adj[u]:
            if self.vartype is Vartype.BINARY:
                self.add_variable(u, adj[u][v])
            elif self.vartype is Vartype.SPIN:
                self.add_offset(adj[u][v])
            else:
                raise RuntimeError("unexpected vartype")
            self.remove_interaction(u, v)

        # all of the interactions that v has become interactions for u
        neighbors = list(adj[v])
        for w in neighbors:
            self.add_interaction(u, w, adj[v][w])
            self.remove_interaction(v, w)

        # finally remove v
        self.remove_variable(v)

###################################################################################################
# transformations
###################################################################################################

    def relabel_variables(self, mapping, inplace=True):
        """Relabel variables of a binary quadratic model as specified by mapping.

        Args:
            mapping (dict):
                Dict mapping current variable labels to new ones. If an incomplete mapping is
                provided, unmapped variables retain their current labels.

            inplace (bool, optional, default=True):
                If True, the binary quadratic model is updated in-place; otherwise, a new binary
                quadratic model is returned.

        Returns:
            :class:`.BinaryQuadraticModel`: A binary quadratic model
            with the variables relabeled. If `inplace` is set to True, returns
            itself.

        Examples:
            This example creates a binary quadratic model with two variables and relables one.

            >>> import dimod
            ...
            >>> model = dimod.BinaryQuadraticModel({0: 0., 1: 1.}, {(0, 1): -1}, 0.0, vartype=dimod.SPIN)
            >>> model.relabel_variables({0: 'a'})   # doctest: +SKIP
            BinaryQuadraticModel({1: 1.0, 'a': 0.0}, {('a', 1): -1}, 0.0, Vartype.SPIN)

            This example creates a binary quadratic model with two variables and returns a new
            model with relabled variables.

            >>> import dimod
            ...
            >>> model = dimod.BinaryQuadraticModel({0: 0., 1: 1.}, {(0, 1): -1}, 0.0, vartype=dimod.SPIN)
            >>> new_model = model.relabel_variables({0: 'a', 1: 'b'}, inplace=False)  # doctest: +SKIP
            >>> new_model.quadratic       # doctest: +SKIP
            {('a', 'b'): -1}

        """
        try:
            old_labels = set(mapping)
            new_labels = set(itervalues(mapping))
        except TypeError:
            raise ValueError("mapping targets must be hashable objects")

        for v in new_labels:
            if v in self.linear and v not in old_labels:
                raise ValueError(('A variable cannot be relabeled "{}" without also relabeling '
                                  "the existing variable of the same name").format(v))

        if inplace:
            shared = old_labels & new_labels
            if shared:
                old_to_intermediate, intermediate_to_new = resolve_label_conflict(mapping, old_labels, new_labels)

                self.relabel_variables(old_to_intermediate, inplace=True)
                self.relabel_variables(intermediate_to_new, inplace=True)
                return self

            linear = self.linear
            quadratic = self.quadratic
            adj = self.adj

            # rebuild linear and adj with the new labels
            for old in list(linear):
                if old not in mapping:
                    continue

                new = mapping[old]

                # get the new interactions that need to be added
                new_interactions = [(new, v, adj[old][v]) for v in adj[old]]

                self.add_variable(new, linear[old])
                self.add_interactions_from(new_interactions)
                self.remove_variable(old)

            return self
        else:
            return BinaryQuadraticModel({mapping.get(v, v): bias for v, bias in iteritems(self.linear)},
                                        {(mapping.get(u, u), mapping.get(v, v)): bias
                                         for (u, v), bias in iteritems(self.quadratic)},
                                        self.offset, self.vartype)

    @vartype_argument('vartype')
    def change_vartype(self, vartype, inplace=True):
        """Create a binary quadratic model with the specified vartype.

        Args:
            vartype (:class:`.Vartype`/str/set, optional):
                Variable type for the changed model. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            inplace (bool, optional, default=True):
                If True, the binary quadratic model is updated in-place; otherwise, a new binary
                quadratic model is returned.

        Returns:
            :class:`.BinaryQuadraticModel`. A new binary quadratic model with
            vartype matching input 'vartype'.

        Examples:
            This example creates an Ising model and then creates a QUBO from it.

            >>> import dimod
            ...
            >>> bqm_spin = dimod.BinaryQuadraticModel({1: 1, 2: 2}, {(1, 2): 0.5}, 0.5, dimod.SPIN)
            >>> bqm_qubo = bqm_spin.change_vartype('BINARY', inplace=False)
            >>> bqm_spin.offset, bqm_spin.vartype
            (0.5, <Vartype.SPIN: frozenset({1, -1})>)
            >>> bqm_qubo.offset, bqm_qubo.vartype
            (-2.0, <Vartype.BINARY: frozenset({0, 1})>)

        """

        if not inplace:
            # create a new model of the appropriate type, then add self's biases to it
            new_model = BinaryQuadraticModel({}, {}, 0.0, vartype)

            new_model.add_variables_from(self.linear, vartype=self.vartype)
            new_model.add_interactions_from(self.quadratic, vartype=self.vartype)
            new_model.add_offset(self.offset)

            return new_model

        # in this case we are doing things in-place, if the desired vartype matches self.vartype,
        # then we don't need to do anything
        if vartype is self.vartype:
            return self

        if self.vartype is Vartype.SPIN and vartype is Vartype.BINARY:
            linear, quadratic, offset = self.spin_to_binary(self.linear, self.quadratic, self.offset)
        elif self.vartype is Vartype.BINARY and vartype is Vartype.SPIN:
            linear, quadratic, offset = self.binary_to_spin(self.linear, self.quadratic, self.offset)
        else:
            raise RuntimeError("something has gone wrong. unknown vartype conversion.")

        # drop everything
        for v in linear:
            self.remove_variable(v)
        self.add_offset(-self.offset)

        self.vartype = vartype
        self.add_variables_from(linear)
        self.add_interactions_from(quadratic)
        self.add_offset(offset)

        return self

##################################################################################################
# static method
##################################################################################################

    @staticmethod
    def spin_to_binary(linear, quadratic, offset):
        """convert linear, quadratic, and offset from spin to binary.
        Does no checking of vartype. Copies all of the values into new objects.
        """

        # the linear biases are the easiest
        new_linear = {v: 2. * bias for v, bias in iteritems(linear)}

        # next the quadratic biases
        new_quadratic = {}
        for (u, v), bias in iteritems(quadratic):
            new_quadratic[(u, v)] = 4. * bias
            new_linear[u] -= 2. * bias
            new_linear[v] -= 2. * bias

        # finally calculate the offset
        offset += sum(itervalues(quadratic)) - sum(itervalues(linear))

        return new_linear, new_quadratic, offset

    @staticmethod
    def binary_to_spin(linear, quadratic, offset):
        """convert linear, quadratic and offset from binary to spin.
        Does no checking of vartype. Copies all of the values into new objects.
        """
        h = {}
        J = {}
        linear_offset = 0.0
        quadratic_offset = 0.0

        for u, bias in iteritems(linear):
            h[u] = .5 * bias
            linear_offset += bias

        for (u, v), bias in iteritems(quadratic):

            J[(u, v)] = .25 * bias

            h[u] += .25 * bias
            h[v] += .25 * bias

            quadratic_offset += bias

        offset += .5 * linear_offset + .25 * quadratic_offset

        return h, J, offset

###################################################################################################
# Methods
###################################################################################################

    def copy(self):
        """Create a copy of a BinaryQuadraticModel.

        Returns:
            :class:`.BinaryQuadraticModel`

        Examples:

            >>> bqm = dimod.BinaryQuadraticModel({1: 1, 2: 2}, {(1, 2): 0.5}, 0.5, dimod.SPIN)
            >>> bqm2 = bqm.copy()


        """
        # new objects are constructed for each, so we just need to pass them in
        return BinaryQuadraticModel(self.linear, self.quadratic, self.offset, self.vartype, **self.info)

    def energy(self, sample):
        """Determine the energy of the specified sample of a binary quadratic model.

        Energy of a sample for a binary quadratic model is defined as a sum, offset
        by the constant energy offset associated with the binary quadratic model, of
        the sample multipled by the linear bias of the variable and
        all its interactions; that is,

        .. math::

            E(\mathbf{s}) = \sum_v h_v s_v + \sum_{u,v} J_{u,v} s_u s_v + c

        where :math:`s_v` is the sample, :math:`h_v` is the linear bias, :math:`J_{u,v}`
        the quadratic bias (interactions), and :math:`c` the energy offset.

        Code for the energy calculation might look like the following::

            energy = model.offset  # doctest: +SKIP
            for v in model:  # doctest: +SKIP
                energy += model.linear[v] * sample[v]
            for u, v in model.quadratic:  # doctest: +SKIP
                energy += model.quadratic[(u, v)] * sample[u] * sample[v]

        Args:
            sample (dict):
                Sample for which to calculate the energy, formatted as a dict where keys
                are variables and values are the value associated with each variable.

        Returns:
            float: Energy for the sample.

        Examples:
            This example creates a binary quadratic model and returns the energies for
            a couple of samples.

            >>> import dimod
            >>> bqm = dimod.BinaryQuadraticModel({1: 1, 2: 1}, {(1, 2): 1}, 0.5, dimod.SPIN)
            >>> bqm.energy({1: -1, 2: -1})
            -0.5
            >>> bqm.energy({1: 1, 2: 1})
            3.5

        """
        linear = self.linear
        quadratic = self.quadratic

        if isinstance(sample, SampleView):
            # because the SampleView object simply reads from an underlying matrix, each read
            # is relatively expensive.
            # However, sample.items() is ~10x faster than {sample[v] for v in sample}, therefore
            # it is much more efficient to dump sample into a dictionary for repeated reads
            sample = dict(sample)

        en = self.offset
        en += sum(linear[v] * sample[v] for v in linear)
        en += sum(sample[u] * sample[v] * quadratic[(u, v)] for u, v in quadratic)
        return en

    def energies(self, samples_like, dtype=np.float):
        """Determine the energies of the given samples.

        Args:
            samples_like (samples_like):
                A collection of raw samples. `samples_like` is an extension of NumPy's array_like
                structure. See :func:`.as_samples`.

            dtype (:class:`numpy.dtype`):
                The data type of the returned energies.

        Returns:
            :obj:`numpy.ndarray`: The energies.

        """
        samples, labels = as_samples(samples_like)

        if all(v == idx for idx, v in enumerate(labels)):
            ldata, (irow, icol, qdata), offset = self.to_numpy_vectors(dtype=dtype)
        else:
            ldata, (irow, icol, qdata), offset = self.to_numpy_vectors(variable_order=labels, dtype=dtype)

        energies = samples.dot(ldata) + (samples[:, irow]*samples[:, icol]).dot(qdata) + offset
        return np.asarray(energies, dtype=dtype)  # handle any type promotions

##################################################################################################
# conversions
##################################################################################################

    def to_coo(self, fp=None, vartype_header=False):
        """Serialize the binary quadratic model to a COOrdinate_ format encoding.

        .. _COOrdinate: https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)

        Args:
            fp (file, optional):
                `.write()`-supporting `file object`_ to save the linear and quadratic biases
                of a binary quadratic model to. The model is stored as a list of 3-tuples,
                (i, j, bias), where :math:`i=j` for linear biases. If not provided,
                returns a string.

            vartype_header (bool, optional, default=False):
                If true, the binary quadratic model's variable type as prepended to the
                string or file as a header.

        .. _file object: https://docs.python.org/3/glossary.html#term-file-object

        .. note:: Variables must use index lables (numeric lables). Binary quadratic
            models saved to COOrdinate format encoding do not preserve offsets.

        Examples:
            This is an example of a binary quadratic model encoded in COOrdinate format.

            .. code-block:: none

                0 0 0.50000
                0 1 0.50000
                1 1 -1.50000

            The Coordinate format with a header

            .. code-block:: none

                # vartype=SPIN
                0 0 0.50000
                0 1 0.50000
                1 1 -1.50000

            This is an example of writing a binary quadratic model to a COOrdinate-format
            file.

            >>> bqm = dimod.BinaryQuadraticModel({0: -1.0, 1: 1.0}, {(0, 1): -1.0}, 0.0, dimod.SPIN)
            >>> with open('tmp.ising', 'w') as file:  # doctest: +SKIP
            ...     bqm.to_coo(file)

            This is an example of writing a binary quadratic model to a COOrdinate-format string.

            >>> bqm = dimod.BinaryQuadraticModel({0: -1.0, 1: 1.0}, {(0, 1): -1.0}, 0.0, dimod.SPIN)
            >>> bqm.to_coo()  # doctest: +SKIP
            0 0 -1.000000
            0 1 -1.000000
            1 1 1.000000

        """
        import dimod.serialization.coo as coo

        if fp is None:
            return coo.dumps(self, vartype_header)
        else:
            coo.dump(self, fp, vartype_header)

    @classmethod
    def from_coo(cls, obj, vartype=None):
        """Deserialize a binary quadratic model from a COOrdinate_ format encoding.

        .. _COOrdinate: https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)

        Args:
            obj: (str/file):
                Either a string or a `.read()`-supporting `file object`_ that represents
                linear and quadratic biases for a binary quadratic model. This data
                is stored as a list of 3-tuples, (i, j, bias), where :math:`i=j`
                for linear biases.

            vartype (:class:`.Vartype`/str/set, optional):
                Variable type for the binary quadratic model. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

                If not provided, the vartype must be specified with a header in the
                file.

        .. _file object: https://docs.python.org/3/glossary.html#term-file-object

        .. note:: Variables must use index lables (numeric lables). Binary quadratic
            models created from COOrdinate format encoding have offsets set to
            zero.

        Examples:
            An example of a binary quadratic model encoded in COOrdinate format.

            .. code-block:: none

                0 0 0.50000
                0 1 0.50000
                1 1 -1.50000

            The Coordinate format with a header

            .. code-block:: none

                # vartype=SPIN
                0 0 0.50000
                0 1 0.50000
                1 1 -1.50000

            This example saves a binary quadratic model to a COOrdinate-format file
            and creates a new model by reading the saved file.

            >>> import dimod
            >>> bqm = dimod.BinaryQuadraticModel({0: -1.0, 1: 1.0}, {(0, 1): -1.0}, 0.0, dimod.BINARY)
            >>> with open('tmp.qubo', 'w') as file:      # doctest: +SKIP
            ...     bqm.to_coo(file)
            >>> with open('tmp.qubo', 'r') as file:      # doctest: +SKIP
            ...     new_bqm = dimod.BinaryQuadraticModel.from_coo(file, dimod.BINARY)
            >>> any(new_bqm)        # doctest: +SKIP
            True

        """
        import dimod.serialization.coo as coo

        if isinstance(obj, str):
            return coo.loads(obj, cls=cls, vartype=vartype)

        return coo.load(obj, cls=cls, vartype=vartype)

    def to_serializable(self, use_bytes=False, bias_dtype=np.float32,
                        bytes_type=bytes):
        """Convert the binary quadratic model to a serializable object.

        Args:
            use_bytes (bool, optional, default=False):
                If True, a compact representation representing the biases as bytes is used.

            bias_dtype (numpy.dtype, optional, default=numpy.float32):
                If `use_bytes` is True, this numpy dtype will be used to
                represent the bias values in the serialized format.

            bytes_type (class, optional, default=bytes):
                This class will be used to wrap the bytes objects in the
                serialization if `use_bytes` is true. Useful for when using
                Python 2 and using BSON encoding, which will not accept the raw
                `bytes` type, so `bson.Binary` can be used instead.

        Returns:
            dict: An object that can be serialized.

        Examples:

            Encode using JSON

            >>> import dimod
            >>> import json
            ...
            >>> bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0}, {('a', 'b'): -1.0}, 0.0, dimod.SPIN)
            >>> s = json.dumps(bqm.to_serializable())

            Encode using BSON_ in python 3.5+

            >>> import dimod
            >>> import bson
            ...
            >>> bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0}, {('a', 'b'): -1.0}, 0.0, dimod.SPIN)
            >>> doc = bqm.to_serializable(use_bytes=True)
            >>> b = bson.BSON.encode(doc)  # doctest: +SKIP

            Encode using BSON in python 2.7. Because :class:`bytes` is an alias for :class:`str`,
            we need to signal to the encoder that it should encode the biases and labels as binary
            data.

            >>> import dimod
            >>> import bson
            ...
            >>> bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0}, {('a', 'b'): -1.0}, 0.0, dimod.SPIN)
            >>> doc = bqm.to_serializable(use_bytes=True, bytes_type=bson.Binary)
            >>> b = bson.BSON.encode(doc)  # doctest: +SKIP

        See also:
            :meth:`~.BinaryQuadraticModel.from_serializable`

            :func:`json.dumps`, :func:`json.dump` JSON encoding functions

            :meth:`bson.BSON.encode` BSON encoding method

        .. _BSON: http://bsonspec.org/

        """
        from dimod.package_info import __version__
        schema_version = "2.0.0"

        try:
            variables = sorted(self.variables)
        except TypeError:
            # sorting unlike types in py3
            variables = list(self.variables)

        num_variables = len(variables)

        # when doing byte encoding we can use less space depending on the
        # total number of variables
        index_dtype = np.uint16 if num_variables <= 2**16 else np.uint32

        ldata, (irow, icol, qdata), offset = self.to_numpy_vectors(
            dtype=bias_dtype,
            index_dtype=index_dtype,
            sort_indices=True,
            variable_order=variables)

        doc = {"basetype": "BinaryQuadraticModel",
               "type": type(self).__name__,
               "version": {"dimod": __version__,
                           "bqm_schema": schema_version},
               "variable_labels": variables,
               "variable_type": self.vartype.name,
               "info": self.info,
               "offset": float(offset),
               "use_bytes": bool(use_bytes)
               }

        if use_bytes:
            doc.update({'linear_biases': array2bytes(ldata, bytes_type=bytes_type),
                        'quadratic_biases': array2bytes(qdata, bytes_type=bytes_type),
                        'quadratic_head': array2bytes(irow, bytes_type=bytes_type),
                        'quadratic_tail': array2bytes(icol, bytes_type=bytes_type)})
        else:
            doc.update({'linear_biases': ldata.tolist(),
                        'quadratic_biases': qdata.tolist(),
                        'quadratic_head': irow.tolist(),
                        'quadratic_tail': icol.tolist()})

        return doc

    def _asdict(self):
        # support simplejson encoding
        return self.to_serializable()

    @classmethod
    def _from_serializable_v1(cls, obj):
        # deprecated
        import warnings

        msg = ("bqm is serialized with a deprecated format and will no longer "
               "work in dimod 0.9.0.")
        warnings.warn(msg)

        from dimod.serialization.json import bqm_decode_hook

        # try decoding with json
        dct = bqm_decode_hook(obj, cls=cls)
        if isinstance(dct, cls):
            return dct

        # assume if not json then binary-type
        bias_dtype, index_dtype = obj["bias_dtype"], obj["index_dtype"]
        lin = np.frombuffer(obj["linear"], dtype=bias_dtype)
        num_variables = len(lin)
        vals = np.frombuffer(obj["quadratic_vals"], dtype=bias_dtype)
        if obj["as_complete"]:
            i, j = zip(*itertools.combinations(range(num_variables), 2))
        else:
            i = np.frombuffer(obj["quadratic_head"], dtype=index_dtype)
            j = np.frombuffer(obj["quadratic_tail"], dtype=index_dtype)

        off = obj["offset"]

        return cls.from_numpy_vectors(lin, (i, j, vals), off,
                                      str(obj["variable_type"]),
                                      variable_order=obj["variable_order"])

    @classmethod
    def from_serializable(cls, obj):
        """Deserialize a binary quadratic model.

        Args:
            obj (dict):
                A binary quadratic model serialized by :meth:`~.BinaryQuadraticModel.to_serializable`.

        Returns:
            :obj:`.BinaryQuadraticModel`

        Examples:

            Encode and decode using JSON

            >>> import dimod
            >>> import json
            ...
            >>> bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0}, {('a', 'b'): -1.0}, 0.0, dimod.SPIN)
            >>> s = json.dumps(bqm.to_serializable())
            >>> new_bqm = dimod.BinaryQuadraticModel.from_serializable(json.loads(s))

        See also:
            :meth:`~.BinaryQuadraticModel.to_serializable`

            :func:`json.loads`, :func:`json.load` JSON deserialization functions

        """
        if obj.get("version", {"bqm_schema": "1.0.0"})["bqm_schema"] != "2.0.0":
            return cls._from_serializable_v1(obj)

        variables = [tuple(v) if isinstance(v, list) else v
                     for v in obj["variable_labels"]]

        if obj["use_bytes"]:
            ldata = bytes2array(obj["linear_biases"])
            qdata = bytes2array(obj["quadratic_biases"])
            irow = bytes2array(obj["quadratic_head"])
            icol = bytes2array(obj["quadratic_tail"])
        else:
            ldata = obj["linear_biases"]
            qdata = obj["quadratic_biases"]
            irow = obj["quadratic_head"]
            icol = obj["quadratic_tail"]

        offset = obj["offset"]
        vartype = obj["variable_type"]

        bqm = cls.from_numpy_vectors(ldata,
                                     (irow, icol, qdata),
                                     offset,
                                     str(vartype),  # handle unicode for py2
                                     variable_order=variables)

        bqm.info.update(obj["info"])
        return bqm

    def to_networkx_graph(self, node_attribute_name='bias', edge_attribute_name='bias'):
        """Convert a binary quadratic model to NetworkX graph format.

        Args:
            node_attribute_name (hashable, optional, default='bias'):
                Attribute name for linear biases.

            edge_attribute_name (hashable, optional, default='bias'):
                Attribute name for quadratic biases.

        Returns:
            :class:`networkx.Graph`: A NetworkX graph with biases stored as
            node/edge attributes.

        Examples:
            This example converts a binary quadratic model to a NetworkX graph, using first
            the default attribute name for quadratic biases then "weight".

            >>> import networkx as nx
            >>> bqm = dimod.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
            ...                                  {(0, 1): .5, (1, 2): 1.5},
            ...                                  1.4,
            ...                                  dimod.SPIN)
            >>> BQM = bqm.to_networkx_graph()
            >>> BQM[0][1]['bias']
            0.5
            >>> BQM.node[0]['bias']
            1
            >>> BQM_w = bqm.to_networkx_graph(edge_attribute_name='weight')
            >>> BQM_w[0][1]['weight']
            0.5

        """
        import networkx as nx

        BQM = nx.Graph()

        # add the linear biases
        BQM.add_nodes_from(((v, {node_attribute_name: bias, 'vartype': self.vartype})
                            for v, bias in iteritems(self.linear)))

        # add the quadratic biases
        BQM.add_edges_from(((u, v, {edge_attribute_name: bias}) for (u, v), bias in iteritems(self.quadratic)))

        # set the offset and vartype properties for the graph
        BQM.offset = self.offset
        BQM.vartype = self.vartype

        return BQM

    @classmethod
    def from_networkx_graph(cls, G, vartype=None, node_attribute_name='bias',
                            edge_attribute_name='bias'):
        """Create a binary quadratic model from a NetworkX graph.

        Args:
            G (:obj:`networkx.Graph`):
                A NetworkX graph with biases stored as node/edge attributes.

            vartype (:class:`.Vartype`/str/set, optional):
                Variable type for the binary quadratic model. Accepted input
                values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

                If not provided, the `G` should have a vartype attribute. If
                `vartype` is provided and `G.vartype` exists then the argument
                overrides the property.

            node_attribute_name (hashable, optional, default='bias'):
                Attribute name for linear biases. If the node does not have a
                matching attribute then the bias defaults to 0.

            edge_attribute_name (hashable, optional, default='bias'):
                Attribute name for quadratic biases. If the edge does not have a
                matching attribute then the bias defaults to 0.

        Returns:
            :obj:`.BinaryQuadraticModel`

        Examples:

            >>> import networkx as nx
            ...
            >>> G = nx.Graph()
            >>> G.add_node('a', bias=.5)
            >>> G.add_edge('a', 'b', bias=-1)
            >>> bqm = dimod.BinaryQuadraticModel.from_networkx_graph(G, 'SPIN')
            >>> bqm.adj['a']['b']
            -1

        """
        if vartype is None:
            if not hasattr(G, 'vartype'):
                msg = ("either 'vartype' argument must be provided or "
                       "the given graph should have a vartype attribute.")
                raise ValueError(msg)
            vartype = G.vartype

        linear = G.nodes(data=node_attribute_name, default=0)
        quadratic = G.edges(data=edge_attribute_name, default=0)
        offset = getattr(G, 'offset', 0)

        return cls(linear, quadratic, offset, vartype)

    def to_ising(self):
        """Converts a binary quadratic model to Ising format.

        If the binary quadratic model's vartype is not :class:`.Vartype.SPIN`,
        values are converted.

        Returns:
            tuple: 3-tuple of form (`linear`, `quadratic`, `offset`), where `linear`
            is a dict of linear biases, `quadratic` is a dict of quadratic biases,
            and `offset` is a number that represents the constant offset of the
            binary quadratic model.

        Examples:
            This example converts a binary quadratic model to an Ising problem.

            >>> import dimod
            >>> model = dimod.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
            ...                                    {(0, 1): .5, (1, 2): 1.5},
            ...                                    1.4,
            ...                                    dimod.SPIN)
            >>> model.to_ising()    # doctest: +SKIP
            ({0: 1, 1: -1, 2: 0.5}, {(0, 1): 0.5, (1, 2): 1.5}, 1.4)

        """
        # cast to a dict
        return dict(self.spin.linear), dict(self.spin.quadratic), self.spin.offset

    @classmethod
    def from_ising(cls, h, J, offset=0.0):
        """Create a binary quadratic model from an Ising problem.


        Args:
            h (dict/list):
                Linear biases of the Ising problem. If a dict, should be of the
                form `{v: bias, ...}` where v is a spin-valued variable and `bias`
                is its associated bias. If a list, it is treated as a list of
                biases where the indices are the variable labels.

            J (dict[(variable, variable), bias]):
                Quadratic biases of the Ising problem.

            offset (optional, default=0.0):
                Constant offset applied to the model.

        Returns:
            :class:`.BinaryQuadraticModel`: Binary quadratic model with vartype set to
            :class:`.Vartype.SPIN`.

        Examples:
            This example creates a binary quadratic model from an Ising problem.

            >>> import dimod
            >>> h = {1: 1, 2: 2, 3: 3, 4: 4}
            >>> J = {(1, 2): 12, (1, 3): 13, (1, 4): 14,
            ...      (2, 3): 23, (2, 4): 24,
            ...      (3, 4): 34}
            >>> model = dimod.BinaryQuadraticModel.from_ising(h, J, offset = 0.0)
            >>> model      # doctest: +SKIP
            BinaryQuadraticModel({1: 1, 2: 2, 3: 3, 4: 4}, {(1, 2): 12, (1, 3): 13, (1, 4): 14, (2, 3): 23, (3, 4): 34, (2, 4): 24}, 0.0, Vartype.SPIN)

        """
        if isinstance(h, abc.Sequence):
            h = dict(enumerate(h))

        return cls(h, J, offset, Vartype.SPIN)

    def to_qubo(self):
        """Convert a binary quadratic model to QUBO format.

        If the binary quadratic model's vartype is not :class:`.Vartype.BINARY`,
        values are converted.

        Returns:
            tuple: 2-tuple of form (`biases`, `offset`), where `biases` is a dict
            in which keys are pairs of variables and values are the associated linear or
            quadratic bias and `offset` is a number that represents the constant offset
            of the binary quadratic model.

        Examples:
            This example converts a binary quadratic model with spin variables to QUBO format
            with binary variables.

            >>> import dimod
            >>> model = dimod.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
            ...                                    {(0, 1): .5, (1, 2): 1.5},
            ...                                    1.4,
            ...                                    dimod.SPIN)
            >>> model.to_qubo()   # doctest: +SKIP
            ({(0, 0): 1.0, (0, 1): 2.0, (1, 1): -6.0, (1, 2): 6.0, (2, 2): -2.0}, 2.9)

        """
        qubo = dict(self.binary.quadratic)
        qubo.update(((v, v), bias) for v, bias in iteritems(self.binary.linear))
        return qubo, self.binary.offset

    @classmethod
    def from_qubo(cls, Q, offset=0.0):
        """Create a binary quadratic model from a QUBO model.

        Args:
            Q (dict):
                Coefficients of a quadratic unconstrained binary optimization
                (QUBO) problem. Should be a dict of the form `{(u, v): bias, ...}`
                where `u`, `v`, are binary-valued variables and `bias` is their
                associated coefficient.

            offset (optional, default=0.0):
                Constant offset applied to the model.

        Returns:
            :class:`.BinaryQuadraticModel`: Binary quadratic model with vartype set to
            :class:`.Vartype.BINARY`.

        Examples:
            This example creates a binary quadratic model from a QUBO model.

            >>> import dimod
            >>> Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
            >>> model = dimod.BinaryQuadraticModel.from_qubo(Q, offset = 0.0)
            >>> model.linear    # doctest: +SKIP
            {0: -1, 1: -1}
            >>> model.vartype
            <Vartype.BINARY: frozenset({0, 1})>

        """
        linear = {}
        quadratic = {}
        for (u, v), bias in iteritems(Q):
            if u == v:
                linear[u] = bias
            else:
                quadratic[(u, v)] = bias

        return cls(linear, quadratic, offset, Vartype.BINARY)

    def to_numpy_matrix(self, variable_order=None):
        """Convert a binary quadratic model to NumPy 2D array.

        Args:
            variable_order (list, optional):
                If provided, indexes the rows/columns of the NumPy array. If `variable_order` includes
                any variables not in the binary quadratic model, these are added to the NumPy array.

        Returns:
            :class:`numpy.ndarray`: The binary quadratic model as a NumPy 2D array. Note that the
            binary quadratic model is converted to :class:`~.Vartype.BINARY` vartype.

        Notes:
            The matrix representation of a binary quadratic model only makes sense for binary models.
            For a binary sample x, the energy of the model is given by:

            .. math::

                E(x) = x^T Q x

            The offset is dropped when converting to a NumPy array.

        Examples:
            This example converts a binary quadratic model to NumPy array format while
            ordering variables and adding one ('d').

            >>> import dimod
            >>> import numpy as np
            ...
            >>> model = dimod.BinaryQuadraticModel({'a': 1, 'b': -1, 'c': .5},
            ...                                    {('a', 'b'): .5, ('b', 'c'): 1.5},
            ...                                    1.4,
            ...                                    dimod.BINARY)
            >>> model.to_numpy_matrix(variable_order=['d', 'c', 'b', 'a'])
            array([[ 0. ,  0. ,  0. ,  0. ],
                   [ 0. ,  0.5,  1.5,  0. ],
                   [ 0. ,  0. , -1. ,  0.5],
                   [ 0. ,  0. ,  0. ,  1. ]])

        """
        import numpy as np

        if variable_order is None:
            # just use the existing variable labels, assuming that they are [0, N)
            num_variables = len(self)
            mat = np.zeros((num_variables, num_variables), dtype=float)

            try:
                for v, bias in iteritems(self.binary.linear):
                    mat[v, v] = bias
            except IndexError:
                raise ValueError(("if 'variable_order' is not provided, binary quadratic model must be "
                                  "index labeled [0, ..., N-1]"))

            for (u, v), bias in iteritems(self.binary.quadratic):
                if u < v:
                    mat[u, v] = bias
                else:
                    mat[v, u] = bias

        else:
            num_variables = len(variable_order)
            idx = {v: i for i, v in enumerate(variable_order)}

            mat = np.zeros((num_variables, num_variables), dtype=float)

            try:
                for v, bias in iteritems(self.binary.linear):
                    mat[idx[v], idx[v]] = bias
            except KeyError as e:
                raise ValueError(("variable {} is missing from variable_order".format(e)))

            for (u, v), bias in iteritems(self.binary.quadratic):
                iu, iv = idx[u], idx[v]
                if iu < iv:
                    mat[iu, iv] = bias
                else:
                    mat[iv, iu] = bias

        return mat

    @classmethod
    def from_numpy_matrix(cls, mat, variable_order=None, offset=0.0, interactions=None):
        """Create a binary quadratic model from a NumPy array.

        Args:
            mat (:class:`numpy.ndarray`):
                Coefficients of a quadratic unconstrained binary optimization (QUBO)
                model formatted as a square NumPy 2D array.

            variable_order (list, optional):
                If provided, labels the QUBO variables; otherwise, row/column indices are used.
                If `variable_order` is longer than the array, extra values are ignored.

            offset (optional, default=0.0):
                Constant offset for the binary quadratic model.

            interactions (iterable, optional, default=[]):
                Any additional 0.0-bias interactions to be added to the binary quadratic model.

        Returns:
            :class:`.BinaryQuadraticModel`: Binary quadratic model with vartype set to
            :class:`.Vartype.BINARY`.


        Examples:
            This example creates a binary quadratic model from a QUBO in NumPy format while
            adding an interaction with a new variable ('f'), ignoring an extra variable
            ('g'), and setting an offset.

            >>> import dimod
            >>> import numpy as np
            >>> Q = np.array([[1, 0, 0, 10, 11],
            ...               [0, 2, 0, 12, 13],
            ...               [0, 0, 3, 14, 15],
            ...               [0, 0, 0, 4, 0],
            ...               [0, 0, 0, 0, 5]]).astype(np.float32)
            >>> model = dimod.BinaryQuadraticModel.from_numpy_matrix(Q,
            ...         variable_order = ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            ...         offset = 2.5,
            ...         interactions = {('a', 'f')})
            >>> model.linear   # doctest: +SKIP
            {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0, 'e': 5.0, 'f': 0.0}
            >>> model.quadratic[('a', 'd')]
            10.0
            >>> model.quadratic[('a', 'f')]
            0.0
            >>> model.offset
            2.5

        """
        import numpy as np

        if mat.ndim != 2:
            raise ValueError("expected input mat to be a square 2D numpy array")

        num_row, num_col = mat.shape
        if num_col != num_row:
            raise ValueError("expected input mat to be a square 2D numpy array")

        if variable_order is None:
            variable_order = list(range(num_row))

        if interactions is None:
            interactions = []

        bqm = cls({}, {}, offset, Vartype.BINARY)

        for (row, col), bias in np.ndenumerate(mat):
            if row == col:
                bqm.add_variable(variable_order[row], bias)
            elif bias:
                bqm.add_interaction(variable_order[row], variable_order[col], bias)

        for u, v in interactions:
            bqm.add_interaction(u, v, 0.0)

        return bqm

    def to_numpy_vectors(self, variable_order=None, dtype=np.float, index_dtype=np.int64, sort_indices=False):
        """Convert a binary quadratic model to numpy arrays.

        Args:
            variable_order (iterable, optional):
                If provided, labels the variables; otherwise, row/column indices are used.

            dtype (:class:`numpy.dtype`, optional):
                Data-type of the biases. By default, the data-type is inferred from the biases.

            index_dtype (:class:`numpy.dtype`, optional):
                Data-type of the indices. By default, the data-type is inferred from the labels.

            sort_indices (bool, optional, default=False):
                If True, the indices are sorted, first by row then by column. Otherwise they
                match :attr:`~.BinaryQuadraticModel.quadratic`.

        Returns:
            :obj:`~numpy.ndarray`: A numpy array of the linear biases.

            tuple: The quadratic biases in COOrdinate format.

                :obj:`~numpy.ndarray`: A numpy array of the row indices of the quadratic matrix
                entries

                :obj:`~numpy.ndarray`: A numpy array of the column indices of the quadratic matrix
                entries

                :obj:`~numpy.ndarray`: A numpy array of the values of the quadratic matrix
                entries

            The offset

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({}, {(0, 1): .5, (3, 2): -1, (0, 3): 1.5}, 0.0, dimod.SPIN)
            >>> lin, (i, j, vals), off = bqm.to_numpy_vectors(sort_indices=True)
            >>> lin
            array([0., 0., 0., 0.])
            >>> i
            array([0, 0, 2])
            >>> j
            array([1, 3, 3])
            >>> vals
            array([ 0.5,  1.5, -1. ])

        """
        linear = self.linear
        quadratic = self.quadratic

        num_variables = len(linear)
        num_interactions = len(quadratic)

        irow = np.empty(num_interactions, dtype=index_dtype)
        icol = np.empty(num_interactions, dtype=index_dtype)
        qdata = np.empty(num_interactions, dtype=dtype)

        if variable_order is None:
            try:
                ldata = np.fromiter((linear[v] for v in range(num_variables)), count=num_variables, dtype=dtype)
            except KeyError:
                raise ValueError(("if 'variable_order' is not provided, binary quadratic model must be "
                                  "index labeled [0, ..., N-1]"))

            # we could speed this up a lot with cython
            for idx, ((u, v), bias) in enumerate(quadratic.items()):
                irow[idx] = u
                icol[idx] = v
                qdata[idx] = bias

        else:
            try:
                ldata = np.fromiter((linear[v] for v in variable_order), count=num_variables, dtype=dtype)
            except KeyError:
                raise ValueError("provided 'variable_order' does not match binary quadratic model")

            label_to_idx = {v: idx for idx, v in enumerate(variable_order)}

            # we could speed this up a lot with cython
            for idx, ((u, v), bias) in enumerate(quadratic.items()):
                irow[idx] = label_to_idx[u]
                icol[idx] = label_to_idx[v]
                qdata[idx] = bias

        if sort_indices:
            # row index should be less than col index, this handles upper-triangular vs lower-triangular
            swaps = irow > icol
            if swaps.any():
                # in-place
                irow[swaps], icol[swaps] = icol[swaps], irow[swaps]

            # sort lexigraphically
            order = np.lexsort((irow, icol))
            if not (order == range(len(order))).all():
                # copy
                irow = irow[order]
                icol = icol[order]
                qdata = qdata[order]

        return ldata, (irow, icol, qdata), ldata.dtype.type(self.offset)

    @classmethod
    def from_numpy_vectors(cls, linear, quadratic, offset, vartype, variable_order=None):
        """Create a binary quadratic model from vectors.

        Args:
            linear (array_like):
                A 1D array-like iterable of linear biases.

            quadratic (tuple[array_like, array_like, array_like]):
                A 3-tuple of 1D array_like vectors of the form (row, col, bias).

            offset (numeric, optional):
                Constant offset for the binary quadratic model.

            vartype (:class:`.Vartype`/str/set):
                Variable type for the binary quadratic model. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            variable_order (iterable, optional):
                If provided, labels the variables; otherwise, indices are used.

        Returns:
            :obj:`.BinaryQuadraticModel`

        Examples:
            >>> import dimod
            >>> import numpy as np
            ...
            >>> linear_vector = np.asarray([-1, 1])
            >>> quadratic_vectors = (np.asarray([0]), np.asarray([1]), np.asarray([-1.0]))
            >>> bqm = dimod.BinaryQuadraticModel.from_numpy_vectors(linear_vector, quadratic_vectors, 0.0, dimod.SPIN)
            >>> print(bqm.quadratic)
            {(0, 1): -1.0}

        """

        try:
            heads, tails, values = quadratic
        except ValueError:
            raise ValueError("quadratic should be a 3-tuple")

        if variable_order is None:
            variable_order = list(range(len(linear)))

        linear = {v: float(bias) for v, bias in zip(variable_order, linear)}
        quadratic = {(variable_order[u], variable_order[v]): float(bias)
                     for u, v, bias in zip(heads, tails, values)}

        return cls(linear, quadratic, offset, vartype)

    def to_pandas_dataframe(self):
        """Convert a binary quadratic model to pandas DataFrame format.

        Returns:
            :class:`pandas.DataFrame`: The binary quadratic model as a DataFrame. The DataFrame has
            binary vartype. The rows and columns are labeled by the variables in the binary quadratic
            model.

        Notes:
            The DataFrame representation of a binary quadratic model only makes sense for binary models.
            For a binary sample x, the energy of the model is given by:

            .. math::

                E(x) = x^T Q x


            The offset is dropped when converting to a pandas DataFrame.

        Examples:
            This example converts a binary quadratic model to pandas DataFrame format.

            >>> import dimod
            >>> model = dimod.BinaryQuadraticModel({'a': 1.1, 'b': -1., 'c': .5},
            ...                                    {('a', 'b'): .5, ('b', 'c'): 1.5},
            ...                                    1.4,
            ...                                    dimod.BINARY)
            >>> model.to_pandas_dataframe()  # doctest: +SKIP
                 a    b    c
            a  1.1  0.5  0.0
            b  0.0 -1.0  1.5
            c  0.0  0.0  0.5

        """
        import pandas as pd

        try:
            variable_order = sorted(self.linear)
        except TypeError:
            variable_order = list(self.linear)

        return pd.DataFrame(self.to_numpy_matrix(variable_order=variable_order),
                            index=variable_order,
                            columns=variable_order)  # let it choose its own datatype

    @classmethod
    def from_pandas_dataframe(cls, bqm_df, offset=0.0, interactions=None):
        """Create a binary quadratic model from a QUBO model formatted as a pandas DataFrame.

        Args:
            bqm_df (:class:`pandas.DataFrame`):
                Quadratic unconstrained binary optimization (QUBO) model formatted
                as a pandas DataFrame. Row and column indices label the QUBO variables;
                values are QUBO coefficients.

            offset (optional, default=0.0):
                Constant offset for the binary quadratic model.

            interactions (iterable, optional, default=[]):
                Any additional 0.0-bias interactions to be added to the binary quadratic model.

        Returns:
            :class:`.BinaryQuadraticModel`: Binary quadratic model with vartype set to
            :class:`vartype.BINARY`.

        Examples:
            This example creates a binary quadratic model from a QUBO in pandas DataFrame format
            while adding an interaction and setting a constant offset.

            >>> import dimod
            >>> import pandas as pd
            >>> pd_qubo = pd.DataFrame(data={0: [-1, 0], 1: [2, -1]})
            >>> pd_qubo
               0  1
            0 -1  2
            1  0 -1
            >>> model = dimod.BinaryQuadraticModel.from_pandas_dataframe(pd_qubo,
            ...         offset = 2.5,
            ...         interactions = {(0,2), (1,2)})
            >>> model.linear        # doctest: +SKIP
            {0: -1, 1: -1.0, 2: 0.0}
            >>> model.quadratic     # doctest: +SKIP
            {(0, 1): 2, (0, 2): 0.0, (1, 2): 0.0}
            >>> model.offset
            2.5
            >>> model.vartype
            <Vartype.BINARY: frozenset({0, 1})>

        """
        if interactions is None:
            interactions = []

        bqm = cls({}, {}, offset, Vartype.BINARY)

        for u, row in bqm_df.iterrows():
            for v, bias in row.iteritems():
                if u == v:
                    bqm.add_variable(u, bias)
                elif bias:
                    bqm.add_interaction(u, v, bias)

        for u, v in interactions:
            bqm.add_interaction(u, v, 0.0)

        return bqm


BQM = BinaryQuadraticModel
"""Alias for :obj:`.BinaryQuadraticModel`"""
