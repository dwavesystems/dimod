"""

todo - describe Ising, QUBO and BQM

"""
from __future__ import absolute_import, division

from numbers import Number

from six import itervalues, iteritems, iterkeys

from dimod.decorators import vartype_argument
from dimod.utilities import resolve_label_conflict
from dimod.vartypes import Vartype


class BinaryQuadraticModel(object):
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
        The BinaryQuadraticModel class does not enforce types on biases
        and offsets, but most applications that use the BinaryQuadraticModel
        class assume that they are numeric.

    Examples:
        This example creates a model with three spin variables.

        >>> model = dimod.BinaryQuadraticModel({0: 1, 1: -1, 2: .5},
        ...                                    {(0, 1): .5, (1, 2): 1.5},
        ...                                    1.4,
        ...                                    dimod.SPIN)

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

        adj (dict):
            The model's interactions as nested dicts.
            In graphic representation, where variables are nodes and interactions
            are edges or adjacencies, keys of the outer dict (`adj`) are all
            the model's nodes (e.g. `v`) and values are the inner dicts. For the
            inner dict associated with outer-key/node 'v', keys are all the nodes
            adjacent to `v` (e.g. `u`) and values are quadratic biases associated
            with the pair of inner and outer keys (`u, v`).

            Examples:
               This example creates an instance of the BinaryQuadraticModel()
               class for the K4 complete graph, where the nodes have biases
               set equal to their sequential labels and interactions are the
               concatenations of the node pairs (e.g., 23 for u,v = 2,3).

               >>> import dimod
               >>> linear = {1: 1, 2: 2, 3: 3, 4: 4}
               >>> quadratic = {(1, 2): 12, (1, 3): 13, (1, 4): 14,
               ...              (2, 3): 23, (2, 4): 24,
               ...              (3, 4): 34}
               >>> offset = 0.0
               >>> vt = dimod.BINARY
               >>> bqm_k4 = dimod.BinaryQuadraticModel(linear, quadratic, offset, vt)
               >>> bqm_k4.adj.viewitems()   # Show all adjacencies  # doctest: +SKIP
               dict_items([(1, {2: 12, 3: 13, 4: 14}),
                           (2, {1: 12, 3: 23, 4: 24}),
                           (3, {1: 13, 2: 23, 4: 34}),
                           (4, {1: 14, 2: 24, 3: 34})])
               >>> bqm_k4.adj[2]            # Show adjacencies for node 2
               {1: 12, 3: 23, 4: 24}
               >>> bqm_k4.adj[2][3]         # Show the quadratic bias for nodes 2,3
               23

        info (dict):
            A place to store miscellaneous data about the BinaryQuadraticModel as a whole.

        SPIN (:class:`.Vartype`): An alias of :class:`.Vartype.SPIN` for easier access.

        BINARY (:class:`.Vartype`): An alias of :class:`.Vartype.BINARY` for easier access.

    """

    SPIN = Vartype.SPIN
    BINARY = Vartype.BINARY

    @vartype_argument('vartype')
    def __init__(self, linear, quadratic, offset, vartype, **kwargs):
        self.linear = {}
        self.quadratic = {}
        self.adj = {}
        self.offset = offset  # we are agnostic to type, though generally should behave like a number
        self.vartype = vartype
        self.info = kwargs  # any additional kwargs are kept as info (metadata)

        # add linear, quadratic
        self.add_variables_from(linear)
        self.add_interactions_from(quadratic)

    @classmethod
    def empty(cls, vartype):
        """Create an empty BinaryQuadraticModel.

        Equivalent to

        .. code-block:: python

            BinaryQuadraticModel({}, {}, 0.0, vartype)

        Args:
            vartype (:class:`.Vartype`/str/set):
                Variable type for the binary quadratic model. Accepted input values:

                * :attr:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :attr:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        Examples:

            >>> bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)
            >>> bqm.linear
            {}
            >>> bqm.quadratic
            {}
            >>> bqm.offset
            0.0

        """
        return cls({}, {}, 0.0, vartype)

    def __repr__(self):
        return 'BinaryQuadraticModel({}, {}, {}, {})'.format(self.linear, self.quadratic, self.offset, self.vartype)

    def __eq__(self, other):
        """Model is equal if and only if linear, adj, offset and vartype are all equal."""
        if not isinstance(other, BinaryQuadraticModel):
            return False

        if self.vartype == other.vartype:
            return all([self.linear == other.linear,
                        self.adj == other.adj,  # adj is invariant of edge order, so check that instead of quadratic
                        self.offset == other.offset])
        else:
            # different vartypes are not equal
            return False

    def __ne__(self, other):
        """Inversion of equality."""
        return not self.__eq__(other)

    def __len__(self):
        """The length is number of variables."""
        return len(self.linear)

##################################################################################################
# vartype properties
##################################################################################################

    @property
    def spin(self):
        """:class:`.BinaryQuadraticModel`: An instance of the Ising model subclass
        of the :class:`.BinaryQuadraticModel` superclass, corresponding to
        a binary quadratic model with spins as its variables.

        Enables access to biases for the spin-valued binary quadratic model
        regardless of the vartype set when the model was created.
        If the model was created with the :attr:`.binary` vartype,
        the Ising model subclass is instantiated upon the first use of the
        :attr:`.spin` property and used in any subsequent reads.

        Examples:
            This example creates a QUBO model and uses the :attr:`.spin` property
            to instantiate the corresponding Ising model.

            >>> import dimod
            >>> bqm_qubo = dimod.BinaryQuadraticModel({0: -1, 1: -1}, {(0, 1): 2}, 0.0, dimod.BINARY)
            >>> bqm_spin = bqm_qubo.spin
            >>> bqm_spin
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
        regardless of the vartype set when the model was created. If the model
        was created with the :attr:`.spin` vartype, the QUBO model subclass is instantiated
        upon the first use of the :attr:`.binary` property and used in any subsequent reads.

        Examples:
           This example creates an Ising model and uses the :attr:`.binary` property
           to instantiate the corresponding QUBO model.

           >>> import dimod
           >>> bqm_spin = dimod.BinaryQuadraticModel({0: 0.0, 1: 0.0}, {(0, 1): 0.5}, -0.5, dimod.SPIN)
           >>> bqm_qubo = bqm_spin.binary
           >>> bqm_qubo
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
            >>> bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 1.0}, {(0, 1): 0.5}, -0.5, dimod.SPIN)
            >>> bqm.linear
            {0: 0.0, 1: 1.0}
            >>> bqm.add_variable(2, 2.0, vartype=dimod.SPIN)        # Add a new variable
            >>> bqm.add_variable(1, 0.33, vartype=dimod.SPIN)
            >>> bqm.add_variable(0, 0.33, vartype=dimod.BINARY)     # Binary value is converted to spin value
            >>> bqm.linear
            {0: 0.165, 1: 1.33, 2: 2.0}

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

            >>> bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
            >>> bqm.add_variables_from({'a': .5, 'b': -1.})
            >>> bqm.linear
            {'a': 0.5, 'b': -1.0}
            >>> bqm.add_variables_from({'b': -1., 'c': 2.0})
            >>> bqm.linear  # doctest: +SKIP
            {'a': 0.5, 'b': -2.0, 'c': 2.0}

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
            >>> bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 1.0}, {(0, 1): 0.5}, -0.5, dimod.SPIN)
            >>> bqm.quadratic
            {(0, 1): 0.5}
            >>> bqm.add_interaction(0, 2, 2)        # Add new variable 2
            >>> bqm.add_interaction(0, 1, .25)
            >>> bqm.add_interaction(1, 2, .25, vartype=dimod.BINARY)     # Binary value is converted to spin value
            >>> bqm.quadratic  # doctest: +SKIP
            {(0, 1): 0.75, (0, 2): 2, (1, 2): 0.0625}

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

            >>> bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.SPIN)
            >>> bqm.add_interactions_from({('a', 'b'): -.5})
            >>> bqm.quadratic
            {('a', 'b'): -0.5}
            >>> bqm.add_interactions_from({('a', 'b'): -.5, ('a', 'c'): 2})
            >>> bqm.add_interactions_from({('b', 'c'): 2}, vartype=dimod.BINARY)   # Binary value is converted to spin value
            >>> bqm.quadratic  # doctest: +SKIP
            {('a', 'b'): -1.0, ('a', 'c'): 2, ('b', 'c'): 0.5}

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
        """Remove variable v and all its interactions from a binary quadratic model.

        Args:
            v (variable):
                The variable to be removed from the binary quadratic model.

        Notes:
            If the specified variable is not in the binary quadratic model, this function does nothing.

        Examples:
            This example creates an Ising model and then removes one variable.

            >>> bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 1.0, 2: 2.0},
            ...                                  {(0, 1): 0.25, (0,2): 0.5, (1,2): 0.75},
            ...                                  -0.5, dimod.SPIN)
            >>> bqm.remove_variable(0)
            >>> bqm.linear
            {1: 1.0, 2: 2.0}
            >>> bqm.quadratic
            {(1, 2): 0.75}

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
        """Remove specified variables and all of their interactions from a binary quadratic model.

        Args:
            variables(iterable):
                A collection of variables to be removed from the binary quadratic model.
                If any variable is not in the model, it is ignored.

        Examples:
            This example creates an Ising model with three variables and interactions
            among all of them, and then removes two variables.

            >>> bqm = dimod.BinaryQuadraticModel({0: 0.0, 1: 1.0, 2: 2.0},
            ...                                  {(0, 1): 0.25, (0,2): 0.5, (1,2): 0.75},
            ...                                  -0.5, dimod.SPIN)
            >>> bqm.remove_variables_from([0, 1])
            >>> bqm.linear
            {2: 2.0}
            >>> bqm.quadratic
            {}

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

            >>> bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1.0, ('b', 'c'): 1.0}, 0.0, dimod.SPIN)
            >>> bqm.remove_interaction('b', 'c')
            >>> bqm.quadratic
            {('a', 'b'): -1.0}
            >>> bqm.remove_interaction('a', 'c')  # not an interaction, so ignored
            >>> bqm.quadratic
            {('a', 'b'): -1.0}

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

            >>> bqm = dimod.BinaryQuadraticModel({}, {('a', 'b'): -1.0, ('b', 'c'): 1.0}, 0.0, dimod.SPIN)
            >>> bqm.remove_interactions_from([('b', 'c'), ('a', 'c')])  # ('a', 'c') is not an interaction, so ignored
            >>> bqm.quadratic
            {('a', 'b'): -1.0}

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

            >>> bqm = dimod.BinaryQuadraticModel({}, {}, 1.3, dimod.SPIN)
            >>> bqm.remove_offset()
            >>> bqm.offset
            0.0

        """
        self.add_offset(-self.offset)

    def scale(self, scalar):
        """Multiply by the specified scalar all the biases and offset of a binary quadratic model.

        Args:
            scalar (number):
                Value by which to scale the energy range of the binary quadratic model.

        Examples:

            This example creates a binary quadratic model and then scales it to half
            the original energy range.

            >>> bqm = dimod.BinaryQuadraticModel({'a': -2.0, 'b': 2.0}, {('a', 'b'): -1.0}, 1.0, dimod.SPIN)
            >>> bqm.scale(0.5)
            >>> bqm.linear
            {'a': -1.0, 'b': 1.0}
            >>> bqm.quadratic
            {('a', 'b'): -0.5}
            >>> bqm.offset
            0.5

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
            >>> bqm = dimod.BinaryQuadraticModel({'a': -.5}, {}, 0.0, dimod.SPIN)
            >>> bqm.fix_variable('a', -1)
            >>> bqm.offset
            0.5
            >>> bqm.linear
            {}

            This example creates a binary quadratic model with two variables and fixes
            the value of one.

            >>> import dimod
            >>> bqm = dimod.BinaryQuadraticModel({'a': -.5, 'b': 0.}, {('a', 'b'): -1}, 0.0, dimod.SPIN)
            >>> bqm.fix_variable('a', -1)
            >>> bqm.offset
            0.5
            >>> bqm.linear
            {'b': 1.0}
            >>> bqm.quadratic
            {}

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
        """Flips variable v in a binary quadratic model.

        Args:
            v (variable):
                Variable in the binary quadratic model. If v is not in the binary
                quadratic model, it is ignored.

        Examples:
            This example creates a binary quadratic model with two variables and inverts the value of one.

            >>> import dimod
            >>> bqm = dimod.BinaryQuadraticModel({1: 1, 2: 2}, {(1, 2): 0.5}, 0.5, dimod.SPIN)
            >>> bqm.flip_variable(1)
            >>> bqm
            BinaryQuadraticModel({1: -1.0, 2: 2}, {(1, 2): -0.5}, 0.5, Vartype.SPIN)

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
           >>> linear = {1: 1, 2: 2}
           >>> quadratic = {(1, 2): 12}
           >>> bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.5, dimod.SPIN)
           >>> linear2 = {2: 0.25, 3: 0.35}
           >>> quadratic2 = {(2, 3): 23}
           >>> bqm2 = dimod.BinaryQuadraticModel(linear2, quadratic2, 0.75, dimod.SPIN)
           >>> bqm.update(bqm2)
           >>> bqm.linear
           {1: 1, 2: 2.25, 3: 0.35}
           >>> bqm.quadratic
           {(1, 2): 12, (2, 3): 23}
           >>> bqm.offset
           1.25

        """
        self.add_variables_from(bqm.linear, vartype=bqm.vartype)
        self.add_interactions_from(bqm.quadratic, vartype=bqm.vartype)
        self.add_offset(bqm.offset)

        if not ignore_info:
            self.info.update(bqm.info)

    def contract_variables(self, u, v):
        """Enforces u, v being the same variable in a binary quadratic model.

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
           >>> linear = {1: 1, 2: 2, 3: 3, 4: 4}
           >>> quadratic = {(1, 2): 12, (1, 3): 13, (1, 4): 14,
           ...              (2, 3): 23, (2, 4): 24,
           ...              (3, 4): 34}
           >>> bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.5, dimod.SPIN)
           >>> bqm.contract_variables(2, 3)
           >>> bqm.linear
           {1: 1, 2: 2, 4: 4}
           >>> bqm.quadratic
           {(1, 2): 25, (1, 4): 14, (2, 4): 58}

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
            :class:`.BinaryQuadraticModel`: A BinaryQuadraticModel with the variables relabeled.
            If inplace=True, returns itself.

        Examples:
            This example creates a binary quadratic model with two variables and relables one.

            >>> import dimod
            >>> model = dimod.BinaryQuadraticModel({0: 0., 1: 1.}, {(0, 1): -1}, 0.0, vartype=dimod.SPIN)
            >>> model.relabel_variables({0: 'a'})
            BinaryQuadraticModel({1: 1.0, 'a': 0.0}, {('a', 1): -1}, 0.0, Vartype.SPIN)

            This example creates a binary quadratic model with two variables and returns a new
            model with relabled variables.

            >>> import dimod
            >>> model = dimod.BinaryQuadraticModel({0: 0., 1: 1.}, {(0, 1): -1}, 0.0, vartype=dimod.SPIN)
            >>> new_model = model.relabel_variables({0: 'a', 1: 'b'}, inplace=False)
            >>> new_model.quadratic
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
        """Create a BinaryQuadraticModel with the specified vartype.

        Args:
            vartype (:class:`.Vartype`/str/set, optional):
                Variable type for the changed model. Accepted input values:
                :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            inplace (bool, optional, default=True):
                If True, the binary quadratic model is updated in-place; otherwise, a new binary
                quadratic model is returned.

        Returns:
            :class:`.BinaryQuadraticModel`. A new BinaryQuadraticModel with
            vartype matching input 'vartype'.

        Examples:
            This example creates an Ising model and then creates a QUBO from it.

            >>> import dimod
            >>> bqm_spin = dimod.BinaryQuadraticModel({1: 1, 2: 2}, {(1, 2): 0.5}, 0.5, dimod.SPIN)
            >>> bqm_qubo = bqm_spin.change_vartype('BINARY', inplace=False)
            >>> bqm_spin
            BinaryQuadraticModel({1: 1, 2: 2}, {(1, 2): 0.5}, 0.5, Vartype.SPIN)
            >>> bqm_qubo
            BinaryQuadraticModel({1: 1.0, 2: 3.0}, {(1, 2): 2.0}, -2.0, Vartype.BINARY)

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
            This example creates a binary quadratic model and copies it.

            >>> import dimod
            >>> bqm = dimod.BinaryQuadraticModel({1: 1, 2: 2}, {(1, 2): 0.5}, 0.5, dimod.SPIN)
            >>> bqm2 = bqm.copy()
            >>> bqm2
            BinaryQuadraticModel({1: 1, 2: 2}, {(1, 2): 0.5}, 0.5, Vartype.SPIN)

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

        Code for the energy calculation might look like the following:

        >>> energy = model.offset  # doctest: +SKIP
        >>> for v in model:  # doctest: +SKIP
        ...     energy += model.linear[v] * sample[v]
        >>> for u, v in model.quadratic:  # doctest: +SKIP
        ...     energy += model.quadratic[(u, v)] * sample[u] * sample[v]

        Args:
            sample (dict):
                Sample for which to calculate the energy as a dict. Keys are variables
                and values are the value associated with each variable.

        Returns:
            float: The energy.

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

        en = self.offset
        en += sum(linear[v] * sample[v] for v in linear)
        en += sum(sample[u] * sample[v] * quadratic[(u, v)] for u, v in quadratic)
        return en

##################################################################################################
# conversions
##################################################################################################

    def to_json(self, fp=None):
        """Serialize the binary quadratic model using JSON.

        Args:
            fp (file, optional):
                A `.write()`-supporting `file object`_. If not provided, the method will return
                a string.

        .. _file object: https://docs.python.org/3/glossary.html#term-file-object

        An example of a serialized BinaryQuadraticModel

        .. code-block:: json

            {
                "linear_terms": [
                    {"bias": 1.0, "label": 0},
                    {"bias": -1.0, "label": 1}
                ],
                "info": {},
                "offset": 0.5,
                "quadratic_terms": [
                    {"bias": 0.5, "label_head": 1, "label_tail": 0}
                ],
                "variable_labels": [0, 1],
                "variable_type": "SPIN",
                "version": {
                    "bqm_schema": "1.0.0",
                    "dimod": "0.6.3"
                }
            }

        Examples:
            Example of writing the binary quadratic model to a file

            >>> bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0}, {('a', 'b'): -1.0, 0.0, dimod.SPIN)
            >>> with open('tmp.txt', 'w') as file:  # doctest: +SKIP
            ...     bqm.to_json(file)

            Example of writing to a string

            >>> bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0}, {('a', 'b'): -1.0}, 0.0, dimod.SPIN)
            >>> bqm.to_json()  # doctest: +SKIP
            {"info": {},
             "linear_terms": [{"bias": -1.0, "label": "a"},
                              {"bias": 1.0, "label": "b"}],
             "offset": 0.0,
             "quadratic_terms": [{"bias": -1.0, "label_head": "b", "label_tail": "a"}],
             "variable_labels": ["a", "b"], "variable_type": "SPIN",
             "version": {"bqm_schema": "1.0.0", "dimod": "0.6.3"}}

        """
        import json
        from dimod.io.json import DimodEncoder

        if fp is None:
            return json.dumps(self, cls=DimodEncoder, sort_keys=True)
        else:
            return json.dump(self, fp, cls=DimodEncoder, sort_keys=True)

    @classmethod
    def from_json(cls, obj):
        """Deserialize a binary quadratic model from a JSON encoding.

        Args:
            obj: (str/file):
                Either a string or a  A `.read()`-supporting `file object`_.

        .. _file object: https://docs.python.org/3/glossary.html#term-file-object

        Examples:
            >>> bqm = dimod.BinaryQuadraticModel({'a': -1.0, 'b': 1.0}, {('a', 'b'): -1.0, 0.0, dimod.SPIN)
            >>> with open('tmp.txt', 'w') as file:  # doctest: +SKIP
            ...     bqm.to_json(file)
            >>> with open('tmp.txt', 'r') as file:  # doctest: +SKIP
            ...     new_bqm = dimod.BinaryQuadraticModel.from_json(file)

        """
        import json

        from dimod.io.json import bqm_decode_hook

        if isinstance(obj, str):
            return json.loads(obj, object_hook=bqm_decode_hook)

        return json.load(obj,  object_hook=bqm_decode_hook)

    def to_networkx_graph(self, node_attribute_name='bias', edge_attribute_name='bias'):
        """Convert a binary quadratic model to NetworkX graph format.

        Args:
            node_attribute_name (hashable):
                Attribute name for linear biases.
            edge_attribute_name (hashable):
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

    def to_ising(self):
        """Converts a binary quadratic model to Ising format.

        If the binary quadratic model's vartype is not spin, values are converted.

        Returns:
            tuple: A 3-tuple of the form (`linear`, `quadratic`, `offset`) where `linear`
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
            >>> model.to_ising()
            ({0: 1, 1: -1, 2: 0.5}, {(0, 1): 0.5, (1, 2): 1.5}, 1.4)

        """
        return self.spin.linear, self.spin.quadratic, self.spin.offset

    @classmethod
    def from_ising(cls, h, J, offset=0.0):
        """Create a binary quadratic model from an Ising problem.


        Args:
            h (dict[variable, bias]/list[bias]):
                Linear biases of the Ising problem. If a list, the list's indices are used
                as variable labels.

            J (dict[(variable, variable), bias]):
                Quadratic biases of the Ising problem.

            offset (optional, default=0.0):
                Constant offset applied to the model.

        Returns:
            :class:`.BinaryQuadraticModel`

        Examples:
            This example creates a binary quadratic model from an Ising problem.

            >>> import dimod
            >>> h = {1: 1, 2: 2, 3: 3, 4: 4}
            >>> J = {(1, 2): 12, (1, 3): 13, (1, 4): 14,
            ...      (2, 3): 23, (2, 4): 24,
            ...      (3, 4): 34}
            >>> model = dimod.BinaryQuadraticModel.from_ising(h, J, offset = 0.0)
            >>> model
            BinaryQuadraticModel({1: 1, 2: 2, 3: 3, 4: 4}, {(1, 2): 12, (1, 3): 13, (1, 4): 14, (2, 3): 23, (3, 4): 34, (2, 4): 24}, 0.0, Vartype.SPIN)

        """
        if isinstance(h, list):
            h = dict(enumerate(h))

        return cls(h, J, offset, Vartype.SPIN)

    def to_qubo(self):
        """Convert a binary quadratic model to QUBO format.

        If the binary quadratic model's vartype is not binary, values are converted.

        Returns:
            tuple: A 2-tuple of the form (`biases`, `offset`) where `biases` is a dict
            where keys are pairs of variables and values are the associated linear or
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
            >>> model.to_qubo()
            ({(0, 0): 1.0, (0, 1): 2.0, (1, 1): -6.0, (1, 2): 6.0, (2, 2): -2.0}, 2.9)

        """
        qubo = {}

        for v, bias in iteritems(self.binary.linear):
            qubo[(v, v)] = bias

        for edge, bias in iteritems(self.binary.quadratic):
            qubo[edge] = bias

        return qubo, self.binary.offset

    @classmethod
    def from_qubo(cls, Q, offset=0.0):
        """Create a binary quadratic model from a QUBO model.

        Args:
            Q (dict):
                Coefficients of a quadratic unconstrained binary optimization (QUBO) model.

            offset (optional, default=0.0):
                Constant offset applied to the model.

        Returns:
            :class:`.BinaryQuadraticModel`

        Examples:
            This example creates a binary quadratic model from a QUBO model.

            >>> import dimod
            >>> Q = {(0, 0): -1, (1, 1): -1, (0, 1): 2}
            >>> model = dimod.BinaryQuadraticModel.from_qubo(Q, offset = 0.0)
            >>> model.linear
            {0: -1, 1: -1}
            >>> model.vartype
            <Vartype.BINARY: frozenset([0, 1])>

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
        """Convert a binary quadratic model to NumPy matrix format.

        Args:
            variable_order (list, optional):
                If provided, indexes the rows/columns of the NumPy array. If `variable_order` includes
                any variables not in the binary quadratic model, these are added to the NumPy matrix.

        Returns:
            :class:`numpy.matrix`: The binary quadratic model as a NumPy matrix. The matrix has binary
            vartype.

        Notes:
            The matrix representation of a binary quadratic model only makes sense for binary models.
            For a binary sample x, the energy of the model is given by:

            .. math::

                E(x) = x^T Q x

            The offset is dropped when converting to a NumPy matrix.

        Examples:
            This example converts a binary quadratic model to NumPy matrix format while
            ordering variables and adding one.

            >>> import dimod
            >>> import numpy as np
            >>> model = dimod.BinaryQuadraticModel({'a': 1, 'b': -1, 'c': .5},
            ...                                    {('a', 'b'): .5, ('b', 'c'): 1.5},
            ...                                    1.4,
            ...                                    dimod.BINARY)
            >>> model.to_numpy_matrix(variable_order=['d', 'c', 'b', 'a'])
            matrix([[ 0. ,  0. ,  0. ,  0. ],
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

        return np.asmatrix(mat)

    @classmethod
    def from_numpy_matrix(cls, mat, variable_order=None, offset=0.0, interactions=None):
        """Create a binary quadratic model from a NumPy matrix.

        Args:
            mat (:class:`numpy.matrix`):
                Coefficients of a quadratic unconstrained binary optimization (QUBO)
                model formatted as a square NumPy matrix.

            variable_order (list, optional):
                If provided, labels the QUBO variables; otherwise, row/column indices are used.
                If `variable_order` is longer than the matrix, extra values are ignored.

            offset (optional, default=0.0):
                Constant offset for the binary quadratic model.

            interactions (iterable, optional, default=[]):
                Any additional 0.0-bias interactions to be added to the binary quadratic model.

        Returns:
            :class:`.BinaryQuadraticModel`

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
            >>> model.linear
            {'a': 1.0, 'b': 2.0, 'c': 3.0, 'd': 4.0, 'e': 5.0, 'f': 0.0}
            >>> model.quadratic
            {('a', 'd'): 10.0,
             ('a', 'e'): 11.0,
             ('a', 'f'): 0.0,
             ('b', 'd'): 12.0,
             ('b', 'e'): 13.0,
             ('c', 'd'): 14.0,
             ('c', 'e'): 15.0}
            >>> model.offset
            2.5

        """
        import numpy as np

        if mat.ndim != 2:
            raise ValueError("expected input mat to be a square matrix")  # pragma: no cover

        num_row, num_col = mat.shape
        if num_col != num_row:
            raise ValueError("expected input mat to be a square matrix")  # pragma: no cover

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
            >>> model = dimod.BinaryQuadraticModel({'a': 1, 'b': -1, 'c': .5},
            ...                                    {('a', 'b'): .5, ('b', 'c'): 1.5},
            ...                                    1.4,
            ...                                    dimod.BINARY)
            >>> model.to_pandas_dataframe()
                 a    b    c
            a  1.0  0.5  0.0
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
            :class:`.BinaryQuadraticModel`

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
            >>> model.linear
            {0: -1, 1: -1.0, 2: 0.0}
            >>> model.quadratic
            {(0, 1): 2, (0, 2): 0.0, (1, 2): 0.0}
            >>> model.offset
            2.5
            >>> model.vartype
            <Vartype.BINARY: frozenset([0, 1])>

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
