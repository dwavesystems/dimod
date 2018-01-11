"""
BinaryQuadraticModel
--------------------
"""
from __future__ import absolute_import

import itertools

from dimod import _PY2
from dimod.vartypes import Vartype

if _PY2:
    def iteritems(d):
        return d.iteritems()

    def itervalues(d):
        return d.itervalues()

    def iterkeys(d):
        return d.iterkeys()

else:
    def iteritems(d):
        return d.items()

    def itervalues(d):
        return d.values()

    def iterkeys(d):
        return d.keys()


class BinaryQuadraticModel(object):
    """Encodes a binary quadratic model.

    Binary quadratic models are the superclass that contains Ising models
    and QUBOs.

    The energy of a binary quadratic model is given by:

    Args:
        linear (dict):
            The linear biases as a dict. The keys should be the
            variables of the binary quadratic model. The values should be
            the linear bias associated with each variable.

        quadratic (dict):
            The quadratic biases as a dict. The keys should
            be 2-tuples of variables. The values should be the quadratic
            bias associated with interaction of variables.
            Each interaction in quadratic should be unique - that is if
            `(u, v)` is a key in quadratic, then `(v, u)` should
            not be.

        offset (number):
            The energy offset associated with the model. Any type input
            is allowed, but many applications that use BinaryQuadraticModel
            will assume that offset is a number.
            See :meth:`.BinaryQuadraticModel.energy`

        vartype (:class:`.Vartype`/str/set):
            The variable type desired for the penalty model.
            Accepted input values:
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
        linear (dict):
            The linear biases as a dict. The keys are the
            variables of the binary quadratic model. The values are
            the linear biases associated with each variable.

        quadratic (dict):
            The quadratic biases as a dict. The keys are 2-tuples of variables.
            Each 2-tuple represents an interaction between two variables in the
            model. The values are the quadratic biases associated with each
            interaction.

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

                >>> model = pm.BinaryQuadraticModel({'a': 0, 'b': 0}, {('a', 'b'): -1}, 0.0, pm.SPIN)

                Then we can see the neighbors of each variable

                >>> model.adj['a']
                {'b': -1}
                >>> model.adj['b']
                {'a': -1}

                In this way if we know that there is an interaction between :code:`'a', 'b'`
                we can easily find the quadratic bias

                >>> model.adj['a']['b']
                -1
                >>> model.adj['b']['a']
                -1

        SPIN (:class:`.Vartype`): An alias for :class:`.Vartype.SPIN` for easier access.

        BINARY (:class:`.Vartype`): An alias for :class:`.Vartype.BINARY` for easier access.

    """

    SPIN = Vartype.SPIN
    BINARY = Vartype.BINARY

    def __init__(self, linear, quadratic, offset, vartype):
        # make sure that we are dealing with a known vartype.
        self.linear = {}
        self.quadratic = {}
        self.adj = {}

        try:
            if isinstance(vartype, str):
                vartype = Vartype[vartype]
            else:
                vartype = Vartype(vartype)
            if not (vartype is Vartype.SPIN or vartype is Vartype.BINARY):
                raise ValueError
        except (ValueError, KeyError):
            raise TypeError(("expected input vartype to be one of: "
                             "Vartype.SPIN, 'SPIN', {-1, 1}, "
                             "Vartype.BINARY, 'BINARY', or {0, 1}."))
        self.vartype = vartype

        # add linear, quadratic
        self.add_variables_from(linear)
        self.add_interactions_from(quadratic)

        # we will also be agnostic to the offset type, the user can determine what makes sense
        self.offset = offset

    def __repr__(self):
        return 'BinaryQuadraticModel({}, {}, {}, {})'.format(self.linear, self.quadratic, self.offset, self.vartype)

    def __eq__(self, model):
        """Model is equal if linear, quadratic, offset and vartype are all equal."""
        if not isinstance(model, BinaryQuadraticModel):
            return False

        if self.vartype == model.vartype:
            return all([self.linear == model.linear,
                        self.adj == model.adj,  # adj is invariant of edge order, so check that instead of quadratic
                        self.offset == model.offset])
        else:
            # different vartypes are not equal
            return False

    def __ne__(self, model):
        """Inversion of equality"""
        return not self.__eq__(model)

    def __len__(self):
        """The length is number of variables."""
        return len(self.linear)

##################################################################################################
# vartype properties
##################################################################################################

    # @property
    # def spin(self):
    #     try:
    #         spin = self._spin
    #         if spin is not None:
    #             return spin
    #     except AttributeError:
    #         pass

    #     if self.vartype is Vartype.SPIN:
    #         self._spin = spin = self
    #     else:
    #         self._spin = spin = self.change_vartype(Vartype.SPIN)

    #         # we also want to go ahead and set spin.binary to refer back to self
    #         spin._binary = self

    #     return spin

    # @property
    # def binary(self):
    #     try:
    #         binary = self._binary
    #         if binary is not None:
    #             return binary
    #     except AttributeError:
    #         pass

    #     if self.vartype is Vartype.BINARY:
    #         self._binary = binary = self
    #     else:
    #         self._binary = binary = self.change_vartype(Vartype.BINARY)

    #         # we also want to go ahead and set binary.spin to refer back to self
    #         binary._spin = self

    #     return binary

###################################################################################################
# update methods
###################################################################################################

    def add_variable(self, v, bias, vartype=None):
        """todo"""
        linear = self.linear

        if v in linear:
            linear[v] += bias
        else:
            linear[v] = bias
            self.adj[v] = {}

    def add_variables_from(self, linear):
        """todo"""
        # We want the linear terms to be a dict.
        # The keys are the variables and the values are the linear biases.
        # Model is deliberately agnostic to the type of the variable names
        # and the biases.
        if not isinstance(linear, dict):
            raise TypeError("expected `linear` to be a dict")

        for v, bias in iteritems(linear):
            self.add_variable(v, bias)

    def add_interaction(self, u, v, bias):
        """todo"""
        if u == v:
            raise ValueError("no self-loops allowed, therefore ({}, {}) is not an allowed interaction".format(u, v))

        linear = self.linear
        quadratic = self.quadratic
        adj = self.adj

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

    def add_interactions_from(self, quadratic):
        """todo"""
        # We want quadratic to be a dict.
        # The keys should be 2-tuples of the form (u, v) where both u and v
        # are in linear.
        # We are agnostic to the type of the bias.
        if isinstance(quadratic, dict):
            for (u, v), bias in iteritems(quadratic):
                self.add_interaction(u, v, bias)
        else:
            try:
                for u, v, bias in quadratic:
                    self.add_interaction(u, v, bias)
            except TypeError:
                raise TypeError("expected 'quadratic' to be a dict or an iterable of 3-tuples.")

    def remove_variable(self, v):
        """todo"""
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

    def remove_variables_from(self, variables):
        """todo"""
        for v in variabels:
            self.remove_variable(v)

    def remove_interaction(u, v):
        """todo"""
        quadratic = self.quadratic
        adj = self.adj
        if (u, v) in quadratic:
            del quadratic[(u, v)]
        if (v, u) in quadratic:
            del quadratic[(v, u)]

        del adj[v][u]
        del adj[u][v]

    def remove_interactions_from(interactions):
        """todo"""
        for u, v in interactions:
            self.remove_interaction(u, v)

###################################################################################################
# transformations
###################################################################################################

    def relabel_variables(self, mapping, copy=True):
        """Relabel the variables according to the given mapping.

        Args:
            mapping (dict): a dict mapping the current variable labels
                to new ones. If an incomplete mapping is provided,
                variables will keep their labels
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
                raise ValueError
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
            raise RuntimeError("something has gone wrong. unknown vartype conversion.")  # pragma: no cover

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
        en += sum(quadratic[(u, v)] * sample[u] * sample[v] for u, v in quadratic)
        return en
