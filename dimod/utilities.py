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
# =============================================================================
import copy
import os
import itertools

from dimod.decorators import lockable_method

__all__ = ['ising_energy',
           'qubo_energy',
           'ising_to_qubo',
           'qubo_to_ising',
           'child_structure_dfs',
           'get_include',
           ]


def ising_energy(sample, h, J, offset=0.0):
    """Calculate the energy for the specified sample of an Ising model.

    Energy of a sample for a binary quadratic model is defined as a sum, offset
    by the constant energy offset associated with the model, of
    the sample multipled by the linear bias of the variable and
    all its interactions. For an Ising model,

    .. math::

        E(\mathbf{s}) = \sum_v h_v s_v + \sum_{u,v} J_{u,v} s_u s_v + c

    where :math:`s_v` is the sample, :math:`h_v` is the linear bias, :math:`J_{u,v}`
    the quadratic bias (interactions), and :math:`c` the energy offset.

    Args:
        sample (dict[variable, spin]):
            Sample for a binary quadratic model as a dict of form {v: spin, ...},
            where keys are variables of the model and values are spins (either -1 or 1).
        h (dict[variable, bias]):
            Linear biases as a dict of the form {v: bias, ...}, where keys are variables of
            the model and values are biases.
        J (dict[(variable, variable), bias]):
           Quadratic biases as a dict of the form {(u, v): bias, ...}, where keys
           are 2-tuples of variables of the model and values are quadratic biases
           associated with the pair of variables (the interaction).
        offset (numeric, optional, default=0):
            Constant offset to be applied to the energy. Default 0.

    Returns:
        float: The induced energy.

    Notes:
        No input checking is performed.

    Examples:
        This example calculates the energy of a sample representing two down spins for
        an Ising model of two variables that have positive biases of value 1 and
        are positively coupled with an interaction of value 1.

        >>> sample = {1: -1, 2: -1}
        >>> h = {1: 1, 2: 1}
        >>> J = {(1, 2): 1}
        >>> dimod.ising_energy(sample, h, J, 0.5)
        -0.5

    References
    ----------

    `Ising model on Wikipedia <https://en.wikipedia.org/wiki/Ising_model>`_

    """
    # add the contribution from the linear biases
    for v in h:
        offset += h[v] * sample[v]

    # add the contribution from the quadratic biases
    for v0, v1 in J:
        offset += J[(v0, v1)] * sample[v0] * sample[v1]

    return offset


def qubo_energy(sample, Q, offset=0.0):
    """Calculate the energy for the specified sample of a QUBO model.

    Energy of a sample for a binary quadratic model is defined as a sum, offset
    by the constant energy offset associated with the model, of
    the sample multipled by the linear bias of the variable and
    all its interactions. For a quadratic unconstrained binary optimization (QUBO)
    model,

    .. math::

        E(\mathbf{x}) = \sum_{u,v} Q_{u,v} x_u x_v + c

    where :math:`x_v` is the sample, :math:`Q_{u,v}`
    a matrix of biases, and :math:`c` the energy offset.

    Args:
        sample (dict[variable, spin]):
            Sample for a binary quadratic model as a dict of form {v: bin, ...},
            where keys are variables of the model and values are binary (either 0 or 1).
        Q (dict[(variable, variable), coefficient]):
            QUBO coefficients in a dict of form {(u, v): coefficient, ...}, where keys
            are 2-tuples of variables of the model and values are biases
            associated with the pair of variables. Tuples (u, v) represent interactions
            and (v, v) linear biases.
        offset (numeric, optional, default=0):
            Constant offset to be applied to the energy. Default 0.

    Returns:
        float: The induced energy.

    Notes:
        No input checking is performed.

    Examples:
        This example calculates the energy of a sample representing two zeros for
        a QUBO model of two variables that have positive biases of value 1 and
        are positively coupled with an interaction of value 1.

        >>> sample = {1: 0, 2: 0}
        >>> Q = {(1, 1): 1, (2, 2): 1, (1, 2): 1}
        >>> dimod.qubo_energy(sample, Q, 0.5)
        0.5

    References
    ----------

    `QUBO model on Wikipedia <https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization>`_

    """
    for v0, v1 in Q:
        offset += sample[v0] * sample[v1] * Q[(v0, v1)]

    return offset


def ising_to_qubo(h, J, offset=0.0):
    """Convert an Ising problem to a QUBO problem.

    Map an Ising model defined on spins (variables with {-1, +1} values) to quadratic
    unconstrained binary optimization (QUBO) formulation :math:`x'  Q  x` defined over
    binary variables (0 or 1 values), where the linear term is contained along the diagonal of Q.
    Return matrix Q that defines the model as well as the offset in energy between the two
    problem formulations:

    .. math::

         s'  J  s + h'  s = offset + x'  Q  x

    See :meth:`~dimod.utilities.qubo_to_ising` for the inverse function.

    Args:
        h (dict[variable, bias]):
            Linear biases as a dict of the form {v: bias, ...}, where keys are variables of
            the model and values are biases.
        J (dict[(variable, variable), bias]):
           Quadratic biases as a dict of the form {(u, v): bias, ...}, where keys
           are 2-tuples of variables of the model and values are quadratic biases
           associated with the pair of variables (the interaction).
        offset (numeric, optional, default=0):
            Constant offset to be applied to the energy. Default 0.

    Returns:
        (dict, float): A 2-tuple containing:

            dict: QUBO coefficients.

            float: New energy offset.

    Examples:
        This example converts an Ising problem of two variables that have positive
        biases of value 1 and are positively coupled with an interaction of value 1
        to a QUBO problem and prints the resulting energy offset.

        >>> h = {1: 1, 2: 1}
        >>> J = {(1, 2): 1}
        >>> dimod.ising_to_qubo(h, J, 0.5)[1]
        -0.5

    """
    # the linear biases are the easiest
    q = {(v, v): 2. * bias for v, bias in h.items()}

    # next the quadratic biases
    for (u, v), bias in J.items():
        if bias == 0.0:
            continue
        q[(u, v)] = 4. * bias
        q[(u, u)] = q.setdefault((u, u), 0) - 2. * bias
        q[(v, v)] = q.setdefault((v, v), 0) - 2. * bias

    # finally calculate the offset
    offset += sum(J.values()) - sum(h.values())

    return q, offset


def qubo_to_ising(Q, offset=0.0):
    """Convert a QUBO problem to an Ising problem.

    Map a quadratic unconstrained binary optimization (QUBO) problem :math:`x'  Q  x`
    defined over binary variables (0 or 1 values), where the linear term is contained along
    the diagonal of Q, to an Ising model defined on spins (variables with {-1, +1} values).
    Return h and J that define the Ising model as well as the offset in energy
    between the two problem formulations:

    .. math::

         x'  Q  x  = offset + s'  J  s + h'  s

    See :meth:`~dimod.utilities.ising_to_qubo` for the inverse function.

    Args:
        Q (dict[(variable, variable), coefficient]):
            QUBO coefficients in a dict of form {(u, v): coefficient, ...}, where keys
            are 2-tuples of variables of the model and values are biases
            associated with the pair of variables. Tuples (u, v) represent interactions
            and (v, v) linear biases.
        offset (numeric, optional, default=0):
            Constant offset to be applied to the energy. Default 0.

    Returns:
        (dict, dict, float): A 3-tuple containing:

            dict: Linear coefficients of the Ising problem.

            dict: Quadratic coefficients of the Ising problem.

            float: New energy offset.

    Examples:
        This example converts a QUBO problem of two variables that have positive
        biases of value 1 and are positively coupled with an interaction of value 1
        to an Ising problem, and shows the new energy offset.

        >>> Q = {(1, 1): 1, (2, 2): 1, (1, 2): 1}
        >>> dimod.qubo_to_ising(Q, 0.5)[2]
        1.75

    """
    h = {}
    J = {}
    linear_offset = 0.0
    quadratic_offset = 0.0

    for (u, v), bias in Q.items():
        if u == v:
            if u in h:
                h[u] += .5 * bias
            else:
                h[u] = .5 * bias
            linear_offset += bias

        else:
            if bias != 0.0:
                J[(u, v)] = .25 * bias

            if u in h:
                h[u] += .25 * bias
            else:
                h[u] = .25 * bias

            if v in h:
                h[v] += .25 * bias
            else:
                h[v] = .25 * bias

            quadratic_offset += bias

    offset += .5 * linear_offset + .25 * quadratic_offset

    return h, J, offset


def resolve_label_conflict(mapping, old_labels=None, new_labels=None):
    """Resolve a self-labeling conflict by creating an intermediate labeling.

    Args:
        mapping (dict):
            A dict mapping the current variable labels to new ones.

        old_labels (set, optional, default=None):
            The keys of mapping. Can be passed in for performance reasons. These are not checked.

        new_labels (set, optional, default=None):
            The values of mapping. Can be passed in for performance reasons. These are not checked.

    Returns:
        tuple: A 2-tuple containing:

            dict: A map from the keys of mapping to an intermediate labeling

            dict: A map from the intermediate labeling to the values of mapping.

    """

    if old_labels is None:
        old_labels = set(mapping)
    if new_labels is None:
        new_labels = set(mapping.values())

    # counter will be used to generate the intermediate labels, as an easy optimization
    # we start the counter with a high number because often variables are labeled by
    # integers starting from 0
    counter = itertools.count(2 * len(mapping))

    old_to_intermediate = {}
    intermediate_to_new = {}

    for old, new in mapping.items():
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

    return old_to_intermediate, intermediate_to_new


def iter_safe_relabels(mapping, existing):
    """Iterator over "safe" intermediate relabelings.

    Args:
        mapping (dict):
            A map from old lables to new.

        existing (set):
            A container of existing labels.

    Yields:
        dict: A "safe" relabelling.

    """

    # put the new labels into a set for fast lookup, also ensures that the
    # values are valid labels
    try:
        new_labels = set(mapping.values())
    except TypeError:
        raise ValueError("mapping targets must be hashable objects")

    old_labels = mapping.keys()

    for v in new_labels:
        if v in existing and v not in old_labels:
            msg = ("A variable cannot be relabeled {!r} without also "
                   "relabeling the existing variable of the same name")
            raise ValueError(msg.format(v))

    if any(v in new_labels for v in old_labels):
        yield from resolve_label_conflict(mapping, old_labels, new_labels)
    else:
        yield mapping


def child_structure_dfs(sampler, seen=None):
    """Return the structure of a composed sampler using a depth-first search on its
    children.

    Args:
        sampler (:obj:`.Sampler`):
            :class:`.Structured` or composed sampler with at least
            one structured child.

        seen (set, optional, default=False):
            IDs of already checked child samplers.

    Returns:
        :class:`~collections.namedtuple`: A named tuple of the form
        `Structure(nodelist, edgelist, adjacency)`, where the 3-tuple values
        are the :attr:`.Structured.nodelist`, :attr:`.Structured.edgelist`
        and :attr:`.Structured.adjacency` attributes of the first structured
        sampler found.

    Raises:
        ValueError: If no structured sampler is found.

    Examples:

    >>> sampler = dimod.TrackingComposite(
    ...                 dimod.StructureComposite(
    ...                 dimod.ExactSolver(), [0, 1], [(0, 1)]))
    >>> print(dimod.child_structure_dfs(sampler).nodelist)
    [0, 1]


    """
    seen = set() if seen is None else seen

    if sampler not in seen:
        try:
            return sampler.structure
        except AttributeError:
            # hasattr just tries to access anyway...
            pass

    seen.add(sampler)

    for child in getattr(sampler, 'children', ()):  # getattr handles samplers
        if child in seen:
            continue

        try:
            return child_structure_dfs(child, seen=seen)
        except ValueError:
            # tree has no child samplers
            pass

    raise ValueError("no structured sampler found")


class LockableDict(dict):
    """A dict that can turn writeablity on and off"""

    # methods like update, clear etc are not wrappers for __setitem__,
    # __delitem__ so they need to be overwritten

    @property
    def is_writeable(self):
        return getattr(self, '_writeable', True)

    @is_writeable.setter
    def is_writeable(self, b):
        self._writeable = bool(b)

    @lockable_method
    def __setitem__(self, key, value):
        return super(LockableDict, self).__setitem__(key, value)

    @lockable_method
    def __delitem__(self, key):
        return super(LockableDict, self).__delitem__(key)

    def __deepcopy__(self, memo):
        new = type(self)()
        memo[id(self)] = new
        new.update((copy.deepcopy(key, memo), copy.deepcopy(value, memo))
                   for key, value in self.items())
        new.is_writeable = self.is_writeable
        return new

    @lockable_method
    def clear(self):
        return super(LockableDict, self).clear()

    @lockable_method
    def pop(self, *args, **kwargs):
        return super(LockableDict, self).pop(*args, **kwargs)

    @lockable_method
    def popitem(self):
        return super(LockableDict, self).popitem()

    @lockable_method
    def setdefault(self, *args, **kwargs):
        return super(LockableDict, self).setdefault(*args, **kwargs)

    @lockable_method
    def update(self, *args, **kwargs):
        return super(LockableDict, self).update(*args, **kwargs)


def get_include():
    """Return the directory with dimod's header files."""
    return os.path.join(os.path.dirname(__file__), 'include')
