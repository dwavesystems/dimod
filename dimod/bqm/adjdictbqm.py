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
import collections.abc as abc

from collections import OrderedDict
from copy import deepcopy
from numbers import Integral

import numpy as np

from dimod.core.bqm import BQM, ShapeableBQM
from dimod.utilities import iter_safe_relabels
from dimod.vartypes import as_vartype, Vartype

try:
    from dimod.bqm.common import itype, ntype
except ImportError:
    itype = np.dtype(np.uint32)
    ntype = np.dtype(np.uint64)


__all__ = ['AdjDictBQM']


class AdjDictBQM(ShapeableBQM):
    """A binary quadratic model structured as a dict-of-dicts.

    Can be created in several ways:

        AdjDictBQM(vartype)
            Creates an empty binary quadratic model (BQM).

        AdjDictBQM(bqm)
            Creates a BQM from another BQM. See `copy` and `cls` kwargs below.

        AdjDictBQM(bqm, vartype)
            Creates a BQM from another BQM, changing to the appropriate
            `vartype` if necessary.

        AdjDictBQM(n, vartype)
            Creates a BQM with `n` variables, indexed linearly from zero,
            setting all biases to zero.

        AdjDictBQM(quadratic, vartype)
            Creates a BQM from quadratic biases given as a square array_like_
            or a dictionary of the form `{(u, v): b, ...}`. Note that when
            formed with SPIN-variables, biases on the diagonal are added to the
            offset.

        AdjDictBQM(linear, quadratic, vartype)
            Creates a BQM from linear and quadratic biases, where `linear` is a
            one-dimensional array_like_ or a dictionary of the form
            `{v: b, ...}`, and `quadratic` is a square array_like_ or a
            dictionary of the form `{(u, v): b, ...}`. Note that when formed
            with SPIN-variables, biases on the diagonal are added to the offset.

        AdjDictBQM(linear, quadratic, offset, vartype)
            Creates a BQM from linear and quadratic biases, where `linear` is a
            one-dimensional array_like_ or a dictionary of the form
            `{v: b, ...}`, and `quadratic` is a square array_like_ or a
            dictionary of the form `{(u, v): b, ...}`, and `offset` is a
            numerical offset. Note that when formed with SPIN-variables, biases
            on the diagonal are added to the offset.

    Notes:

        The AdjDictBQM is implemented using a dict-of-dicts structure. The
        outer dict contains the BQM's variables as keys and the neighborhoods
        as values. Each neighborhood dict contains the neighbors as keys and
        the quadratic biases as values. The linear biases are stored as
        self-interactions.

        Advantages:

        - Pure python implementation
        - Supports arbitrary python types as biases
        - Low complexity for lookup operations
        - Supports incremental construction

        Disadvantages:

        - Slow iteration
        - High memory usage

        Intended Use:

        - For small problems or when flexibility is important

    Examples:
        The first example constructs a BQM from a dict.

        >>> dimod.AdjDictBQM({'a': -1.0}, {('a', 'b'): 1.0}, 'SPIN')
        AdjDictBQM({a: -1.0, b: 0.0}, {('a', 'b'): 1.0}, 0.0, 'SPIN')

        The next example demonstrates incremental construction:

        >>> bqm = dimod.AdjDictBQM('SPIN')
        >>> bqm.add_variable('a')
        'a'
        >>> bqm.add_variable()
        1
        >>> bqm.set_quadratic('a', 1, 3.0)
        >>> bqm
        AdjDictBQM({a: 0.0, 1: 0.0}, {('a', 1): 3.0}, 0.0, 'SPIN')

        This example shows support for arbitrary types.

        >>> import numpy as np
        >>> from fractions import Fraction
        >>> dimod.AdjDictBQM({('a', 'b'): Fraction(1, 3)}, 'BINARY')
        AdjDictBQM({a: 0.0, b: 0.0}, {('a', 'b'): 1/3}, 0.0, 'BINARY')

    .. _array_like: https://numpy.org/doc/stable/user/basics.creation.html

    """
    # developer note: we should add a FAQ for why spin-valued diagonal biases
    # are offsets and reference it here

    def __init__(self, *args, vartype=None):

        if vartype is not None:
            # pass in as a positional argument
            self.__init__(*args, vartype)
            return

        self.dtype = np.dtype(object)

        # we use ordered dict because the other BQM types are ordered. However
        # collection's OrderedDict is optimized for both FILO and FIFO so has
        # a larger memory footprint than we need. We could instead store a list
        # with the dict which would be more performant but complicate the
        # implementation
        self._adj = adj = OrderedDict()

        if len(args) == 0:
            raise TypeError("A valid vartype or another bqm must be provided")
        if len(args) == 1:
            # BQM(bqm) or BQM(vartype)
            obj, = args
            if isinstance(obj, BQM):
                self._init_bqm(obj)
            else:
                self._init_number(0, obj)
        elif len(args) == 2:
            # BQM(bqm, vartype), BQM(n, vartype) or BQM(M, vartype)
            obj, vartype = args
            if isinstance(obj, BQM):
                self._init_bqm(obj, vartype)
            elif isinstance(obj, Integral):
                self._init_number(obj, vartype)
            else:
                self._init_components({}, obj, 0.0, vartype)
        elif len(args) == 3:
            # BQM(linear, quadratic, vartype)
            linear, quadratic, vartype = args
            self._init_components(linear, quadratic, 0.0, vartype)
        elif len(args) == 4:
            # BQM(linear, quadratic, offset, vartype)
            self._init_components(*args)
        else:
            msg = "__init__() takes 4 positional arguments but {} were given"
            raise TypeError(msg.format(len(args)))

    def _init_bqm(self, bqm, vartype=None):
        self.linear.update(bqm.linear)
        self.quadratic.update(bqm.quadratic)
        self.offset = bqm.offset
        self._vartype = bqm.vartype

        if vartype is not None:
            self.change_vartype(as_vartype(vartype), inplace=True)

    def _init_components(self, linear, quadratic, offset, vartype):
        self._vartype = vartype = as_vartype(vartype)

        if isinstance(linear, (abc.Mapping, abc.Iterator)):
            self.linear.update(linear)
        else:
            # assume a sequence
            self.linear.update(enumerate(linear))

        adj = self._adj

        if isinstance(quadratic, abc.Mapping):
            for (u, v), bias in quadratic.items():
                self.add_variable(u)
                self.add_variable(v)

                if u == v and vartype is Vartype.SPIN:
                    offset = offset + bias  # not += on off-chance it's mutable
                elif u in adj[v]:
                    adj[u][v] = adj[v][u] = adj[u][v] + bias
                else:
                    adj[u][v] = adj[v][u] = bias
        elif isinstance(quadratic, abc.Iterator):
            for u, v, bias in quadratic:
                self.add_variable(u)
                self.add_variable(v)

                if u == v and vartype is Vartype.SPIN:
                    offset = offset + bias  # not += on off-chance it's mutable
                elif u in adj[v]:
                    adj[u][v] = adj[v][u] = adj[u][v] + bias
                else:
                    adj[u][v] = adj[v][u] = bias
        else:
            # unlike the other BQM types we let numpy handle the typing
            if isinstance(quadratic, np.ndarray):
                dtype = quadratic.dtype
            else:
                quadratic = np.asarray(quadratic, dtype=np.object)

            D = np.atleast_2d(quadratic)

            num_variables = D.shape[0]

            if D.ndim != 2 or num_variables != D.shape[1]:
                raise ValueError("expected dense to be a 2 dim square array")

            # make sure all the variables are present
            for v in range(num_variables):
                self.add_variable(v)

            it = np.nditer(D, flags=['multi_index', 'refs_ok'], op_flags=['readonly'])
            while not it.finished:
                u, v = it.multi_index
                bias = it.value[()]

                if bias:
                    if u == v and vartype is Vartype.SPIN:
                        # not += on off-chance it's mutable
                        offset = offset + bias
                    elif u in adj[v]:
                        adj[u][v] = adj[v][u] = adj[u][v] + bias
                    else:
                        adj[u][v] = adj[v][u] = bias

                it.iternext()

        self.offset = offset

    def _init_number(self, n, vartype):
        self.linear.update((v, 0.0) for v in range(n))
        self.offset = 0.0
        self._vartype = as_vartype(vartype)

    @property
    def num_variables(self):
        """int: The number of variables in the model."""
        return len(self._adj)

    @property
    def num_interactions(self):
        """int: The number of interactions in the model."""
        return (sum(map(len, self._adj.values())) - len(self._adj)) // 2

    @property
    def offset(self):
        """The constant energy offset associated with the model."""
        return self._offset

    @offset.setter
    def offset(self, offset):
        self._offset = offset

    @property
    def vartype(self):
        """:class:`.Vartype`: The vartype of the binary quadratic model. One of
        :class:`.Vartype.SPIN` or :class:`.Vartype.BINARY`.
        """
        # so it's readonly
        return self._vartype

    def add_variable(self, v=None, bias=0.0):
        """Add a variable to the binary quadratic model.

        Args:
            v (hashable, optional):
                A label for the variable. Defaults to the length of the binary
                quadratic model, if that label is available. Otherwise defaults
                to the lowest available positive integer label.

            bias (numeric, optional, default=0):
                The initial bias value for the added variable. If `v` is already
                a variable, then `bias` (if any) is adding to its existing
                linear bias.

        Returns:
            hashable: The label of the added variable.

        Raises:
            TypeError: If the label is not hashable.

        Examples:

            >>> bqm = dimod.AdjDictBQM('SPIN')
            >>> bqm.add_variable()
            0
            >>> bqm.add_variable('a')
            'a'
            >>> bqm.add_variable()
            2

            >>> bqm = dimod.AdjDictBQM('SPIN')
            >>> bqm.add_variable(1)
            1
            >>> bqm.add_variable()  # 1 is taken
            0
            >>> bqm.add_variable()
            2

        """
        if v is None:

            v = self.num_variables

            if self.has_variable(v):
                # if v already exists there must be a smaller one that's free
                for v in range(self.num_variables):
                    if not self.has_variable(v):
                        break

        self._adj.setdefault(v, OrderedDict({v: 0.0}))[v] += bias
        return v

    def change_vartype(self, vartype, inplace=True):
        """Return a binary quadratic model with the specified vartype.

        Args:
            vartype (:class:`.Vartype`/str/set, optional):
                Variable type for the changed model. Accepted input values:

                * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
                * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            inplace (bool, optional, default=True):
                If True, the binary quadratic model is updated in-place;
                otherwise, a new binary quadratic model is returned.

        Returns:
            :obj:`.AdjDictBQM`: A binary quadratic model with the specified
            vartype.

        """
        if not inplace:
            return self.copy().change_vartype(vartype, inplace=True)

        vartype = as_vartype(vartype)

        # in place and we are already correct, so nothing to do
        if self.vartype == vartype:
            return self

        if vartype is Vartype.BINARY:
            lin_mp, lin_offset_mp = 2.0, -1.0
            quad_mp, lin_quad_mp, quad_offset_mp = 4.0, -2.0, 1.0
        elif vartype is Vartype.SPIN:
            lin_mp, lin_offset_mp = 0.5, .5
            quad_mp, lin_quad_mp, quad_offset_mp = 0.25, 0.25, 0.25
        else:
            raise ValueError("unkown vartype")

        for v, bias in self.linear.items():
            self.linear[v] = lin_mp * bias
            self.offset += lin_offset_mp * bias

        for (u, v), bias in self.quadratic.items():
            self.adj[u][v] = quad_mp * bias

            self.linear[u] += lin_quad_mp * bias
            self.linear[v] += lin_quad_mp * bias

            self.offset += quad_offset_mp * bias

        self._vartype = vartype

        return self

    def __copy__(self):
        # We want to make new nested dictionaries, but keep the biases
        # in case they are mutable (e.g. sympy variables)
        return type(self)(self)

    def degree(self, v):
        """Return degree of the specified variable.

        The degree is the number of variables sharing an interaction with ``v``.

        Args:
            v (hashable):
                Variable in the binary quadratic model.

        Returns:
            Degree of ``v``.

        Raises:
            ValueError: If ``v`` is not a variable in the binary quadratic model.
        """
        try:
            return len(self._adj[v]) - 1
        except KeyError:
            raise ValueError("{} is not a variable".format(v))

    def get_linear(self, v):
        """Get the linear bias of the specified variable.

        Args:
            v (hashable):
                A variable in the binary quadratic model.

        Returns:
            float: The linear bias of ``v``.

        Raises:
            ValueError: If ``v`` is not a variable in the binary quadratic model.

        """
        try:
            return self._adj[v][v]
        except KeyError:
            pass
        raise ValueError("{} is not a variable".format(v))

    def get_quadratic(self, u, v, default=None):
        """Get the quadratic bias of the specified interaction.

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

            default (optional):
                Value to return if there is no interactions between ``u`` and ``v``.

        Returns:
            The quadratic bias of ``(u, v)``.

        Raises:
            ValueError: If either ``u`` or ``v`` is not a variable in the binary
                quadratic model or if ``u == v``

            ValueError: If ``(u, v)`` is not an interaction and `default` is
                `None`.

        """
        if u == v:
            raise ValueError("No interaction between {} and itself".format(u))
        try:
            return self._adj[u][v]
        except KeyError:
            pass
        if default is not None:
            return default
        raise ValueError('No interaction between {} and {}'.format(u, v))

    def iter_linear(self):
        """Iterate over the linear biases of the binary quadratic model.

        Yields:
            tuple: A variable in the binary quadratic model and its linear bias.
        """
        for u, neighborhood in self._adj.items():
            yield u, neighborhood[u]  # u and it's bias

    def iter_quadratic(self, variables=None):
        """Iterate over the quadratic biases of the binary quadratic model.

        Args:
            variables (iterable):
                Variables in the binary quadratic model. Iterates only over
                interactions of these variables.

        Yields:
            3-tuple: Interaction variables in the binary quadratic model and their
            bias.
        """

        if variables is None:
            variables = self._adj
        elif self.has_variable(variables):
            variables = [variables]

        seen = set()
        for u in variables:
            neighborhood = self._adj[u]
            seen.add(u)  # also avoids self-loops
            for v, bias in neighborhood.items():
                if v not in seen:
                    yield (u, v, bias)

    def remove_variable(self, v=None):
        """Remove a variable and its associated interactions.

        Args:
            v (variable, optional):
                The variable to be removed from the binary quadratic model
                (BQM). If not provided, the last variable added is removed.

        Returns:
            variable: The removed variable.

        Raises:
            ValueError: If the BQM is empty or if ``v`` is not
                a variable.

        """
        if len(self._adj) == 0:
            raise ValueError("pop from empty binary quadratic model")

        if v is None:
            v, neighborhood = self._adj.popitem(last=True)
        else:
            try:
                neighborhood = self._adj.pop(v)
            except KeyError:
                raise ValueError('{!r} is not a variable'.format(v))

        for u in neighborhood:
            if u != v:
                self._adj[u].pop(v)

        return v

    def remove_interaction(self, u, v):
        """Remove the interaction between the specified variables.

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

        Returns:
            bool: If there was an interaction to remove.

        Raises:
            ValueError: If either ``u`` or ``v`` is not a variable in the binary
                quadratic model.

        """
        if u == v:
            raise ValueError("No interaction between {} and itself".format(u))

        try:
            self._adj[u].pop(v)
        except KeyError:
            # nothing pop
            raise ValueError('No interaction between {} and {}'.format(u, v))

        # want to raise an exception in the case that they got out of sync
        # as a sanity check
        self._adj[v].pop(u)

        return True

    def relabel_variables(self, mapping, inplace=True):
        """Relabel variables of a binary quadratic model as specified by mapping.

        Args:
            mapping (dict):
                Dict mapping current variable labels to new ones. If an
                incomplete mapping is provided, unmapped variables retain their
                current labels.

            inplace (bool, optional, default=True):
                If True, the binary quadratic model is updated in-place;
                otherwise, a new binary quadratic model is returned.

        Returns:
            A binary quadratic model with the variables relabeled. If `inplace`
            is set to True, returns itself.

        """
        if not inplace:
            return self.copy().relabel_variables(mapping, inplace=True)

        adj = self._adj

        for submap in iter_safe_relabels(mapping, self.linear):
            for old, new in submap.items():
                if old == new:
                    continue

                # replace the linear bias
                adj[new] = {new: adj[old].pop(old)}

                # copy the quadratic biases
                for v in adj[old]:
                    adj[new][v] = adj[v][new] = adj[v].pop(old)

                # remove the old adj for old
                del adj[old]

        return self

    def set_linear(self, v, bias):
        """Set the linear biase of the specified variable.

        Args:
            v (hashable):
                A variable in the binary quadratic model. It is added if not
                already in the model.

            b (numeric):
                The linear bias of ``v``.

        Raises:
            TypeError: If ``v`` is not hashable

        """
        if v in self._adj:
            self._adj[v][v] = bias
        else:
            self._adj[v] = {v: bias}

    def set_quadratic(self, u, v, bias):
        """Set the quadratic bias of an interaction specified by its variables.

        Args:
            u (hashable):
                A variable in the binary quadratic model.

            v (hashable):
                A variable in the binary quadratic model.

            b (numeric):
                The quadratic bias to set for interaction ``(u, v)``

        Raises:
            TypeError: If ``u`` or ``v`` is not hashable.

        """
        # make sure the variables exist
        self.add_variable(u)
        self.add_variable(v)

        adj = self._adj
        adj[u][v] = adj[v][u] = bias
