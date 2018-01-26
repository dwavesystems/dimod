"""
BinaryQuadraticModel
====================
"""
from __future__ import absolute_import, division

import itertools

from numbers import Number

from dimod.compatibility23 import itervalues, iteritems, iterkeys
from dimod.vartypes import Vartype


class BinaryQuadraticModel(object):
    """Encodes a binary quadratic model.

    Binary quadratic model is the superclass that contains the `Ising model`_ and the QUBO_.

    .. _Ising model: https://en.wikipedia.org/wiki/Ising_model
    .. _QUBO: https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization

    Args:
        linear (dict[variable, bias]):
            The linear biases as a dict.
            The keys should be the variables of the binary quadratic model. A variable can be any
            python object that can be used as the key of a dictionary.
            The values should be the linear bias associated with each variable. Biases are generally
            a number but this is not explicitly checked.

        quadratic (dict[(variable, variable), bias]):
            The quadratic biases as a dict.
            The keys should be 2-tuples of variables.  A variable can be any python object that can
            be used as the key of a dictionary. A pair of variables is called an interaction.
            The values should be the quadratic bias associated with the interaction. Biases are
            generally a number but this is not explicitly checked.
            Interactions that are not unique are added.

        offset (number):
            The constant energy offset associated with the binary quadratic model. Any type input is
            allowed, but many applications will assume that offset is a number.
            See :meth:`.BinaryQuadraticModel.energy`

        vartype (:class:`.Vartype`/str/set):
            The variable type desired for the binary quadratic model. Accepted input values:
            :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

    Notes:
        The BinaryQuadraticModel does not enforce types on the biases
        and the offset, but most applications that use BinaryQuadraticModel
        will assume that they are numeric.

    Examples:
        >>> model = pm.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
        ...                                 {(0, 1): .5, (1, 2): 1.5},
        ...                                 1.4,
        ...                                 pm.SPIN)

    Attributes:
        linear (dict[variable, bias]):
            The linear biases as a dict. The keys are the variables of the binary quadratic model.
            The values are the linear biases associated with each variable.

        quadratic (dict[(variable, variable), bias]):
            The quadratic biases as a dict. The keys are 2-tuples of variables. Each 2-tuple
            represents an interaction between two variables in the model. The values are the
            quadratic biases associated with each interaction.

        offset (number):
            The energy offset associated with the model. Same type as given
            on instantiation.

        vartype (:class:`.Vartype`):
            The model's type. One of :class:`.Vartype.SPIN` or :class:`.Vartype.BINARY`.

        adj (dict):
            Encodes the interactions of the model in nested dicts. The keys of adj
            are the variables of the model and the values are neighbor-dicts.
            For a node `v`, the keys of the neighbor-dict associated with `v` are
            the neighbors of `v` and for each `u` in the neighbor-dict the value
            associated with `u` is the quadratic bias associated with `u, v`.

            Examples:
                If we create a BinaryQuadraticModel with a single interaction

                >>> bqm = pm.BinaryQuadraticModel({'a': 0, 'b': 0}, {('a', 'b'): -1}, 0.0, pm.SPIN)

                Then we can see the neighbors of each variable

                >>> bqm.adj['a']
                {'b': -1}
                >>> bqm.adj['b']
                {'a': -1}

                In this way if we know that there is an interaction between :code:`'a', 'b'`
                we can easily find the quadratic bias

                >>> bqm.adj['a']['b']
                -1
                >>> bqm.adj['b']['a']
                -1

        SPIN (:class:`.Vartype`): An alias for :class:`.Vartype.SPIN` for easier access.

        BINARY (:class:`.Vartype`): An alias for :class:`.Vartype.BINARY` for easier access.

    """

    SPIN = Vartype.SPIN
    BINARY = Vartype.BINARY

    def __init__(self, linear, quadratic, offset, vartype):
        self.linear = {}
        self.quadratic = {}
        self.adj = {}
        self.offset = offset  # we are agnostic to type, though generally should behave like a number

        # make sure that we are dealing with a known vartype.
        try:
            if isinstance(vartype, str):
                vartype = Vartype[vartype]
            else:
                vartype = Vartype(vartype)

            if not (vartype is Vartype.SPIN or vartype is Vartype.BINARY):
                raise ValueError  # this gets caught

        except (ValueError, KeyError):
            raise TypeError(("expected input vartype to be one of: "
                             "Vartype.SPIN, 'SPIN', {-1, 1}, "
                             "Vartype.BINARY, 'BINARY', or {0, 1}."))
        self.vartype = vartype

        # add linear, quadratic
        self.add_variables_from(linear)
        self.add_interactions_from(quadratic)

    def __repr__(self):
        return 'BinaryQuadraticModel({}, {}, {}, {})'.format(self.linear, self.quadratic, self.offset, self.vartype)

    def __eq__(self, bqm):
        """Model is equal if and only if linear, adj, offset and vartype are all equal."""
        if not isinstance(bqm, BinaryQuadraticModel):
            return False

        if self.vartype == bqm.vartype:
            return all([self.linear == bqm.linear,
                        self.adj == bqm.adj,  # adj is invariant of edge order, so check that instead of quadratic
                        self.offset == bqm.offset])
        else:
            # different vartypes are not equal
            return False

    def __ne__(self, bqm):
        """Inversion of equality."""
        return not self.__eq__(bqm)

    def __len__(self):
        """The length is number of variables."""
        return len(self.linear)

##################################################################################################
# vartype properties
##################################################################################################

    @property
    def spin(self):
        """:class:`.BinaryQuadraticModel`: The spin-valued version of the binary quadratic model.

        This property allows the user to access the biases for the appropriate vartype without
        needing to check the given binary quadratic model.

        Examples:
            Create a spin-valued binary quadratic model. In this case the :attr:`.spin` attribute
            refers back to itself.

            >>> bqm = dimod.BinaryQuadraticModel({'a': 0, 'b': 0}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
            >>> bqm.spin is bqm
            True

            For a binary-valued binary quadratic model, it's spin-valued counterpart will be built
            the first time the :attr:`.spin` property is accessed and subsequent reads will use
            it.

            >>> bqm = dimod.BinaryQuadraticModel({'a': .5, 'b': .5}, {('a', 'b'): -1}, 0.0, dimod.BINARY)
            >>> bqm.spin.linear
            {'a': 0, 'b': 0}
            >>> bqm.spin.quadratic
            {('a', 'b'): -.25}
            >>> bqm.spin.offset
            .25

            The energy will correspond.

            >>> bqm = dimod.BinaryQuadraticModel({'a': .5, 'b': .5}, {('a', 'b'): -1}, 0.0, dimod.BINARY)
            >>> bqm.energy({'a': 0, 'b': 1})
            .5
            >>> bqm.spin.energy({'a': -1, 'b': +1})
            .5

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
            self._counterpart = self._spin = spin = self.change_vartype(Vartype.SPIN)

            # we also want to go ahead and set spin.binary to refer back to self
            spin._binary = self

        return spin

    @property
    def binary(self):
        """:class:`.BinaryQuadraticModel`: The binary-valued version of the binary quadratic model.

        This property allows the user to access the biases for the appropriate vartype without
        needing to check the given binary quadratic model.

        Examples:
            Create a binary-valued binary quadratic model. In this case the :attr:`.binary` attribute
            refers back to itself.

            >>> bqm = dimod.BinaryQuadraticModel({'a': .5, 'b': .5}, {('a', 'b'): -1}, 0.0, dimod.BINARY)
            >>> bqm.binary is bqm
            True

            For a spin-valued binary quadratic model, it's binary-valued counterpart will be built
            the first time the :attr:`.binary` property is accessed and subsequent reads will use
            it.

            >>> bqm = dimod.BinaryQuadraticModel({'a': 0, 'b': 0}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
            >>> bqm.binary.linear
            {'a': 2.0, 'b': 2.0}
            >>> bqm.binary.quadratic
            {('a', 'b'): -4.0}
            >>> bqm.binary.offset
            -1.0

            The energy will correspond.

            >>> bqm = dimod.BinaryQuadraticModel({'a': 0, 'b': 0}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
            >>> bqm.binary.energy({'a': 0, 'b': 1})
            1.0
            >>> bqm.energy({'a': -1, 'b': +1})
            1.0

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
            self._counterpart = self._binary = binary = self.change_vartype(Vartype.BINARY)

            # we also want to go ahead and set binary.spin to refer back to self
            binary._spin = self

        return binary

###################################################################################################
# update methods
###################################################################################################

    def add_variable(self, v, bias, vartype=None):
        """Add a variable v and its bias.

        Args:
            v (variable):
                A variable can be any python object that could be used as a key of a dict.

            bias (bias):
                The linear bias associated with v. If v already is in the model, the bias is added
                to the existing linear bias. Many methods and functions expect bias to be a number
                but this is not explicitly checked.

            vartype (:class:`.Vartype`, optional, default=None):
                The vartype of the given bias. If None will be the same vartype as the binary
                quadratic model. If given, should be :class:`.Vartype.SPIN` or
                :class:`.Vartype.BINARY`.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
            >>> bqm.add_variable('a', .5)
            >>> bqm.linear
            {'a': .5}

            Variables that already exist have their bias added.

            >>> bqm = dimod.BinaryQuadraticModel({'b': -1}, {}, 0.0, dimod.SPIN)
            >>> bqm.add_variable('b', .5)
            >>> bqm.linear
            {'b': -.5}

        """

        # handle the case that a different vartype is provided
        if vartype is not None and vartype is not self.vartype:
            if self.vartype is Vartype.SPIN and vartype is Vartype.BINARY:
                # convert from binary to spin
                bias /= 2.
                self.offset += bias
            elif self.vartype is Vartype.BINARY and vartype is Vartype.SPIN:
                # convert from spin to binary
                self.offset -= bias
                bias *= 2.
            else:
                raise ValueError("unknown vartype")

        # add the variable to linear and adj
        linear = self.linear
        if v in linear:
            linear[v] += bias
        else:
            linear[v] = bias
            self.adj[v] = {}

        try:
            self._counterpart.add_variable(v, bias, vartype=self.vartype)
        except AttributeError:
            pass

    def add_variables_from(self, linear, vartype=None):
        """Add linear biases.

        Args:
            linear (dict[variable, bias]/iterable[(variable, bias)]):
                A collection of linear biases. If a dict, the keys should be variables in the
                binary quadratic model and the values should be biases. Otherwise should be
                an iterable of (variable, bias) pairs. The variables can be any python object
                that could be used as a key in a dict. Many methods and functions expect the biases
                to be numbers but this is not explicitly checked.
                If any of the variables already exist in the model, their bias is added to the
                existing linear bias.

            vartype (:class:`.Vartype`, optional, default=None):
                The vartype of the given bias. If None will be the same vartype as the binary
                quadratic model. If given, should be :class:`.Vartype.SPIN` or
                :class:`.Vartype.BINARY`.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
            >>> bqm.add_variables_from({'a': .5, 'b': -1.})
            >>> bqm.linear
            {'a': .5, 'b': -1.}

            Variables that already exist have their bias added.

            >>> bqm = dimod.BinaryQuadraticModel({'b': -1.}, {}, 0.0, dimod.SPIN)
            >>> bqm.add_variables_from({'a': .5, 'b': -1.})
            >>> bqm.linear
            {'a': .5, 'b': -2.}

        """
        if isinstance(linear, dict):
            for v, bias in iteritems(linear):
                self.add_variable(v, bias, vartype=vartype)
        else:
            try:
                for v, bias in linear:
                    self.add_variable(v, bias, vartype=vartype)
            except TypeError:
                raise TypeError("expected 'linear' to be a dict or an iterable of 2-tuples.")

    def add_interaction(self, u, v, bias, vartype=None):
        """Add a variable interaction and its quadratic bias.

        Args:
            v (variable):
                A variable can be any python object that could be used as a key of a dict.

            u (variable):
                A variable can be any python object that could be used as a key of a dict.

            bias (bias):
                The quadratic bias associated with u, v. If u, v already is in the model, the bias
                is added to the existing quadratic bias. Many methods and functions expect bias to
                be a number but this is not explicitly checked.

            vartype (:class:`.Vartype`, optional, default=None):
                The vartype of the given bias. If None will be the same vartype as the binary
                quadratic model. If given, should be :class:`.Vartype.SPIN` or
                :class:`.Vartype.BINARY`.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
            >>> bqm.add_interaction('a', 'b', -.5)
            >>> bqm.quadratic
            {('a', 'b'): -.5}

            Variables that already exist have their bias added.

            >>> bqm = dimod.BinaryQuadraticModel({}, {('b', 'a'): -.5}, 0.0, dimod.SPIN)
            >>> bqm.add_interaction('a', 'b', -.5)
            >>> bqm.quadratic
            {('b', 'a'): -1.}

        """
        if u == v:
            raise ValueError("no self-loops allowed, therefore ({}, {}) is not an allowed interaction".format(u, v))

        linear = self.linear
        quadratic = self.quadratic
        adj = self.adj

        if vartype is not None and vartype is not self.vartype:
            if self.vartype is Vartype.SPIN and vartype is Vartype.BINARY:
                # convert from binary to spin
                bias /= 4.

                self.offset += bias

                if u in linear:
                    linear[u] += bias
                else:
                    linear[u] = bias
                    self.adj[u] = {}

                if v in linear:
                    linear[v] += bias
                else:
                    linear[v] = bias
                    self.adj[v] = {}

            elif self.vartype is Vartype.BINARY and vartype is Vartype.SPIN:
                # convert from spin to binary
                self.offset += bias

                if u in linear:
                    linear[u] += -2. * bias
                else:
                    linear[u] = -2. * bias
                    self.adj[u] = {}

                if v in linear:
                    linear[v] += -2. * bias
                else:
                    linear[v] = -2. * bias
                    self.adj[v] = {}

                bias *= 4.
            else:
                raise ValueError("unknown vartype")
        else:
            if u not in linear:
                linear[u] = 0.
                adj[u] = {}
            if v not in linear:
                linear[v] = 0.
                adj[v] = {}

        if (v, u) in quadratic:
            quadratic[(v, u)] += bias
            adj[u][v] += bias
            adj[v][u] += bias
        elif (u, v) in quadratic:
            quadratic[(u, v)] += bias
            adj[u][v] += bias
            adj[v][u] += bias
        else:
            quadratic[(u, v)] = bias
            adj[u][v] = bias
            adj[v][u] = bias

        try:
            self._counterpart.add_interaction(u, v, bias, vartype=self.vartype)
        except AttributeError:
            pass

    def add_interactions_from(self, quadratic, vartype=None):
        """Add quadratic biases.

        Args:
            quadratic (dict[(variable, variable), bias]/iterable[(variable, variable, bias)]):
                Variables that have an interaction and their quadratic bias. If a dict, the keys
                should be 2-tuples of the variables and the values should be their corresponding
                bias. Can also be an iterable of 3-tuples. Each interaction in quadratic should be
                unique - that is if `(u, v)` is a key in quadratic, then `(v, u)` should not be.
                The variables can be any python object that could be used as a key in a dict.
                Many methods and functions expect the biases to be numbers but this is not
                explicitly checked.

            vartype (:class:`.Vartype`, optional, default=None):
                The vartype of the given bias. If None will be the same vartype as the binary
                quadratic model. If given, should be :class:`.Vartype.SPIN` or
                :class:`.Vartype.BINARY`.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
            >>> bqm.add_interactions_from({('a', 'b'): -.5})
            >>> bqm.quadratic
            {('a', 'b'): -.5}

            Variables that already exist have their bias added.

            >>> bqm = dimod.BinaryQuadraticModel({}, {('b', 'a'): -.5}, 0.0, dimod.SPIN)
            >>> bqm.add_interactions_from({('a', 'b'): -.5})
            >>> bqm.quadratic
            {('b', 'a'): -1.}

        """
        if isinstance(quadratic, dict):
            for (u, v), bias in iteritems(quadratic):
                self.add_interaction(u, v, bias, vartype=vartype)
        else:
            try:
                for u, v, bias in quadratic:
                    self.add_interaction(u, v, bias, vartype=vartype)
            except TypeError:
                raise TypeError("expected 'quadratic' to be a dict or an iterable of 3-tuples.")

    def remove_variable(self, v):
        """Remove the variable v and all of its interactions.

        Args:
            v (variable):
                A variable in the binary quadratic model.

        Notes:
            If the given variable is not in the binary quadratic model, this function does nothing.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({'a': 1., 'b': 2.}, {}, 0.0, dimod.SPIN)
            >>> bqm.remove_variable('a')
            >>> bqm.linear
            {'b': 2.}

        """
        linear = self.linear
        if v in linear:
            del linear[v]
        else:
            # nothing to remove
            return

        quadratic = self.quadratic
        adj = self.adj

        for u in adj[v]:
            if (u, v) in quadratic:
                del quadratic[(u, v)]
            else:
                del quadratic[(v, u)]

            del adj[u][v]

        del adj[v]

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
        """Remove the given variables and all of their interactions.

        Args:
            variables(iterable):
                A collection of variables to be removed from the binary quadratic model.

        Notes:
            If any variable is not in the binary quadratic model, it is ignored.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({'a': 1., 'b': 2., 'c': 3.}, {}, 0.0, dimod.SPIN)
            >>> bqm.remove_variables_from(['a', 'c'])
            >>> bqm.linear
            {'b': 2.}

        """
        for v in variables:
            self.remove_variable(v)

    def remove_interaction(self, u, v):
        """Remove the interaction between u, v.

        Args:
            u (variable):
                A variable in the binary quadratic model that has an interaction with v.

            v (variable):
                A variable in the binary quadratic model that has an interaction with u.

        Notes:
            Any interaction not in the binary quadratic model is ignored.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1, ('b', 'c'): 1}, 0.0, dimod.SPIN)
            >>> bqm.remove_interaction('b', 'c')
            >>> bqm.quadratic
            {('a', 'b'): -1}
            >>> bqm.remove_interaction('a', 'c')  # not an interaction, so ignored
            >>> bqm.quadratic
            {('a', 'b'): -1}

        """
        quadratic = self.quadratic
        adj = self.adj

        try:
            del adj[v][u]
        except KeyError:
            return  # no interaction with that name
        del adj[u][v]

        if (u, v) in quadratic:
            del quadratic[(u, v)]
        else:
            del quadratic[(v, u)]

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
        """Remove all of the given interactions from the binary quadratic model.

        Args:
            interactions (iterable[[variable, variable]]):
                A collections of interactions. Each interaction should be a 2-tuple of variables
                in the binary quadratic model.

        Notes:
            Any interaction not in the binary quadratic model is ignored.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1, ('b', 'c'): 1}, 0.0, dimod.SPIN)
            >>> bqm.remove_interactions_from([('b', 'c'), ('a', 'c')])  # ('a', 'c') is not an interaction, so ignored
            >>> bqm.quadratic
            {('a', 'b'): -1}

        """
        for u, v in interactions:
            self.remove_interaction(u, v)

    def add_offset(self, offset):
        """Add given value to the offset.

        Args:
            offset (number):
                A value to be added to the constant energy offset for the binary quadratic model.

        """
        self.offset += offset

        try:
            self._counterpart.add_offset(offset)
        except AttributeError:
            pass

    def remove_offset(self):
        """Set the binary quadratic model's offset to zero.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({}, {}, 1.3, dimod.SPIN)
            >>> bqm.remove_offset()
            >>> bqm.offset
            0.0

        """
        self.add_offset(-self.offset)

    def scale(self, scalar):
        """Multiply all of the biases and the offset by the given scalar.

        Args:
            scalar (number):
                The value to scale the energy range of the binary quadratic model by.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({'a': -2, 'b': 2}, {('a', 'b'): -1}, 1., dimod.SPIN)
            >>> bqm.scale(.5)
            >>> bqm.linear
            {'a': -1., 'b': 1.}
            >>> bqm.quadratic
            {('a', 'b'): -.5}
            >>> bqm.offset
            .5

        """
        if not isinstance(scalar, Number):
            raise TypeError("expected scalar to be a Number")

        linear = self.linear
        for v in linear:
            linear[v] *= scalar

        quadratic = self.quadratic
        for edge in quadratic:
            quadratic[edge] *= scalar

        adj = self.adj
        for u in adj:
            for v in adj[u]:
                adj[u][v] *= scalar

        self.offset *= scalar

        try:
            self._counterpart.scale(scalar)
        except AttributeError:
            pass

    def fix_variable(self, v, value):
        """Fix the value of a variable in the binary quadratic model and remove it.

        Args:
            v (variable):
                A variable in the binary quadratic model that has an interaction with u.

            value (int):
                The value assigned to the variable, must match the :class:`.Vartype` of the binary
                quadratic model.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({'a': -.5}, {}, 0.0, dimod.SPIN)
            >>> bqm.fix_variable('a', -1)
            >>> bqm.offset
            .5
            >>> bqm.linear
            {}

            >>> bqm = dimod.BinaryQuadraticModel({'a': -.5, 'b': 0.}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
            >>> bqm.fix_variable('a', -1)
            >>> bqm.offset
            .5
            >>> bqm.linear
            {'b': 1}
            >>> bqm.quadratic
            {}

        Notes:
            Acts on the binary quadratic model in place.

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

    def flip_variable(self, v):
        """Flips a single variable v.

        Args:
            v (variable):
                A variable in the binary quadratic model.

        Notes:
            If v is not in the binary quadratic model then it is ignored.

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({'a': -1}, {}, 0.0, dimod.SPIN)
            >>> original = bqm.copy()
            >>> bqm.flip_variable('a')
            >>> bqm.energy({'a': -1}) == bqm.energy({'a': 1})
            True

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

    def update(self, bqm):
        """Update with the values from another binary quadratic model.

        Args:
            bqm (:class:`.BinaryQuadraticModel`):
                A binary quadratic model. All of the biases are added to self.

        """
        self.add_variables_from(bqm.linear, vartype=bqm.vartype)
        self.add_interactions_from(bqm.quadratic, vartype=bqm.vartype)
        self.add_offset(bqm.offset)

    def contract_variables(self, u, v):
        """Enforces u, v are the same variable.

        The resulting variable will be labeled as 'u'.

        Args:
            u (variable):
                A variable in the binary quadratic model.

            v (variable):
                A variable in the binary quadratic model.

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

    def relabel_variables(self, mapping, copy=True):
        """Relabel the variables according to the given mapping.

        Args:
            mapping (dict): a dict mapping the current variable labels
                to new ones. If an incomplete mapping is provided,
                unmapped variables will keep their labels
            copy (bool, default): If True, return a copy of BinaryQuadraticModel
                with the variables relabeled, otherwise apply the relabeling in
                place.

        Returns:
            :class:`.BinaryQuadraticModel`: A BinaryQuadraticModel with the
            variables relabeled. If copy=False, returns itself.

        Examples:
            >>> model = pm.BinaryQuadraticModel({0: 0., 1: 1.}, {(0, 1): -1}, 0.0, vartype=pm.SPIN)
            >>> new_model = model.relabel_variables({0: 'a'})
            >>> new_model.quadratic
            {('a', 1): -1}
            >>> new_model = model.relabel_variables({0: 'a', 1: 'b'}, copy=False)
            >>> model.quadratic
            {('a', 'b'): -1}
            >>> new_model is model
            True

        """
        try:
            old_labels = set(iterkeys(mapping))
            new_labels = set(itervalues(mapping))
        except TypeError:
            raise ValueError("mapping targets must be hashable objects")

        for v in new_labels:
            if v in self.linear and v not in old_labels:
                raise ValueError(('A variable cannot be relabeled "{}" without also relabeling '
                                  "the existing variable of the same name").format(v))

        if copy:
            return BinaryQuadraticModel({mapping.get(v, v): bias for v, bias in iteritems(self.linear)},
                                        {(mapping.get(u, u), mapping.get(v, v)): bias
                                         for (u, v), bias in iteritems(self.quadratic)},
                                        self.offset, self.vartype)
        else:
            shared = old_labels & new_labels
            if shared:
                # in this case relabel to a new intermediate labeling, then map from the intermediate
                # labeling to the desired labeling

                # counter will be used to generate the intermediate labels, as an easy optimization
                # we start the counter with a high number because often variables are labeled by
                # integers starting from 0
                counter = itertools.count(2 * len(self))

                old_to_intermediate = {}
                intermediate_to_new = {}

                for old, new in iteritems(mapping):
                    if old == new:
                        # we can remove self-labels
                        continue

                    if old in new_labels or new in old_labels:

                        # try to get a new unique label
                        lbl = next(counter)
                        while lbl in new_labels or lbl in old_labels:
                            lbl = next(counter)

                        # add it to the mapping
                        old_to_intermediate[old] = lbl
                        intermediate_to_new[lbl] = new

                    else:
                        old_to_intermediate[old] = new
                        # don't need to add it to intermediate_to_new because it is a self-label

                self.relabel_variables(old_to_intermediate, copy=False)
                self.relabel_variables(intermediate_to_new, copy=False)
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

    def change_vartype(self, vartype):
        """Creates a new BinaryQuadraticModel with the given vartype.

        Args:
            vartype (:class:`.Vartype`/str/set, optional):
                The variable type desired for the penalty model. Accepted input values:
                :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        Returns:
            :class:`.BinaryQuadraticModel`. A new BinaryQuadraticModel with
            vartype matching input 'vartype'.

        """
        try:
            if isinstance(vartype, str):
                vartype = Vartype[vartype]
            else:
                vartype = Vartype(vartype)
            if not (vartype is Vartype.SPIN or vartype is Vartype.BINARY):
                raise ValueError  # pragma: no cover
        except (ValueError, KeyError):
            raise TypeError(("expected input vartype to be one of: "
                             "Vartype.SPIN, 'SPIN', {-1, 1}, "
                             "Vartype.BINARY, 'BINARY', or {0, 1}."))

        # vartype matches so we are done
        if vartype is self.vartype:
            return self.copy()

        if self.vartype is Vartype.SPIN and vartype is Vartype.BINARY:
            linear, quadratic, offset = self.spin_to_binary(self.linear, self.quadratic, self.offset)
            return BinaryQuadraticModel(linear, quadratic, offset, vartype=Vartype.BINARY)
        elif self.vartype is Vartype.BINARY and vartype is Vartype.SPIN:
            linear, quadratic, offset = self.binary_to_spin(self.linear, self.quadratic, self.offset)
            return BinaryQuadraticModel(linear, quadratic, offset, vartype=Vartype.SPIN)
        else:
            raise RuntimeError("something has gone wrong. unknown vartype conversion.")

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
        """Create a copy of the BinaryQuadraticModel.

        Returns:
            :class:`.BinaryQuadraticModel`

        """
        # new objects are constructed for each, so we just need to pass them in
        return BinaryQuadraticModel(self.linear, self.quadratic, self.offset, self.vartype)

    def energy(self, sample):
        """Determines the energy of the given sample.

        The energy is calculated:

        >>> energy = model.offset  # doctest: +SKIP
        >>> for v in model:  # doctest: +SKIP
        ...     energy += model.linear[v] * sample[v]
        >>> for u, v in model.quadratic:  # doctest: +SKIP
        ...     energy += model.quadratic[(u, v)] * sample[u] * sample[v]

        Or equivalently, let us define:

            :code:`sample[v]` as :math:`s_v`

            :code:`model.linear[v]` as :math:`h_v`

            :code:`model.quadratic[(u, v)]` as :math:`J_{u,v}`

            :code:`model.offset` as :math:`c`

        then,

        .. math::

            E(\mathbf{s}) = \sum_v h_v s_v + \sum_{u,v} J_{u,v} s_u s_v + c

        Args:
            sample (dict): The sample. The keys should be the variables and
                the values should be the value associated with each variable.

        Returns:
            float: The energy.

        """
        linear = self.linear
        quadratic = self.quadratic

        en = self.offset
        en += sum(linear[v] * sample[v] for v in linear)
        en += sum(sample[u] * sample[v] * quadratic[(u, v)] for u, v in quadratic)
        return en
