"""
Decorators that provide input checking and format for Ising and QUBO
functions and methods.
"""
from decorator import decorator

from dimod import _PY2

__all__ = ['qubo', 'ising', 'qubo_index_labels', 'ising_index_labels']

if _PY2:
    range = xrange

    def iteritems(d):
        return d.iteritems()

else:
    def iteritems(d):
        return d.items()


def qubo(Q_arg):
    """Provides input checking for quadratic unconstrained binary
    optimization (QUBO) methods.

    A dictionary of QUBO coefficients `Q` should be a dict. The keys
    should be tuples of the form (u, v) where u, v are variables in
    the problem. The variables can themselves be any hashable object.

    If both (u, v) and (v, u) are in `Q`, then their values are added.

    Args:
        Q_arg (int): Location of the `Q` argument in args.

    Returns:
        Function which checks the form of the `Q` argument before
        running the decorated function.

    Raises:
        TypeError: If `Q` is not a dict.
        TypeError: If any of the keys of `Q` are not tuples.
        ValueError: If any of the keys of `Q` are not length 2.

    Examples:
    Decorate functions like this::

        @qubo(1)
        def sample_qubo(self, Q):
            pass

        @qubo(0)
        def qubo_energy(Q, sample):
            pass

    """
    @decorator
    def _qubo(f, *args, **kw):
        Q = args[Q_arg]

        if not isinstance(Q, dict):
            raise TypeError("expected first input 'Q' to be of type 'dict'")

        if not all(isinstance(edge, tuple) for edge in Q):
            raise TypeError("the keys of 'Q' should all be tuples of length 2")

        if not all(len(edge) == 2 for edge in Q):
            raise ValueError("the keys of 'Q' should all be tuples of length 2")

        return f(*args, **kw)
    return _qubo


def ising(h_arg, J_arg):
    """Provides input checking for Ising methods.

    Ising problems are defined by a collection of linear biases `h`,
    and a collection of quadratic biases `J`. `J` should be a dict
    where each key is a tuple of the form (u, v) such that u, v are
    variables in the Ising problem and u != v. `h` should be a dict
    where the keys are the variables in the Ising problem.

    If `h` is provided as a list, it will be converted to a dict
    where the keys are the indices. Additionally, if any variable
    is referenced by `J` but is not a key of `h`, it will be added
    to `h` with bias 0.

    Args:
        h_arg (int): Location of the `h` argument in args.
        J_arg (int): Location of the `J` argument in args.

    Returns:
        Function which checks the form of the `h` and `J` arguments
        before running the decorated function.

    Raises:
        TypeError: If `h` is not a dict.
        TypeError: If `J` is not a dict.
        TypeError: If the keys of `J` are not tuples.
        ValueError: If the keys of `J` are not of length 2.

    Examples:
    Decorate functions like this::

        @ising(1, 2)
        def sample_ising(self, h, J):
            pass

        @ising(0, 1)
        def ising_energy(h, J, sample):
            pass

    """
    @decorator
    def _ising(f, *args, **kw):

        h = args[h_arg]
        J = args[J_arg]

        # if h is provided as a list, make it a dict where the keys of h are
        # the indices.
        if isinstance(h, (list, tuple)):
            h = dict(enumerate(h))
        elif not isinstance(h, dict):
            raise TypeError("expected input 'h' to be a dict")

        # now all the checking of J
        if not isinstance(J, dict):
            raise TypeError("expected input 'J' to be a dict")

        if not all(isinstance(edge, tuple) for edge in J):
            raise TypeError("expected the keys of 'J' to be tuples of length 2")

        if not all(len(edge) == 2 for edge in J):
            raise ValueError("expected the keys of 'J' to be tuples of length 2")

        # finally any nodes in J that are not in h are added to h
        for u, v in J:
            if u not in h:
                h[u] = 0.
            if v not in h:
                h[v] = 0.

        args = list(args)
        args[h_arg] = h
        # args[J_arg] = J

        return f(*args, **kw)
    return _ising


def qubo_index_labels(Q_arg):
    """Replaces the labels in the dictionary of QUBO coefficients with
    integers.

    Args:
        Q_arg (int): Location of the `Q` argument in args.

    Returns:
        Function that replaces the user-given `Q` with one with integer
        labels, runs the decorated method, then restores the previous
        labels.

    Notes:
        The integer labels start with 0.

        If the given labels are orderable, the relabeling is applied
        to the nodes sorted lexicographically.

    Examples:
    Decorate functions like this::

        @qubo_index_labels(1)
        def sample_qubo(self, Q):
            pass

        @qubo_index_labels(0)
        def qubo_energy(Q, sample):
            pass

    """
    @decorator
    def _qubo_index_labels(f, *args, **kw):

        Q = args[Q_arg]

        # get all of the node labels from Q
        nodes = set().union(*Q)

        # if the nodes are already index labeled from (0, n-1) then we are already
        # done
        if all(idx in nodes for idx in range(len(nodes))):
            return f(*args, **kw)

        # let's relabel them, sorting the node labels lexicographically.
        # In python 3, unlike types cannot be sorted, so let's just handle
        # that case with a try/catch.
        try:
            inv_relabel = dict(enumerate(sorted(nodes)))
        except TypeError:
            inv_relabel = dict(enumerate(nodes))
        relabel = {v: idx for idx, v in iteritems(inv_relabel)}

        # with relabel in hand, let's make a new Q with the nodes
        # labeled appropriately
        newQ = {(relabel[u], relabel[v]): coeff for (u, v), coeff in iteritems(Q)}

        # now substitute our newQ into the arguments and run the function
        newargs = list(args)
        newargs[Q_arg] = newQ

        response = f(*newargs, **kw)

        # the returned response will need to be relabelled with inv_relabel
        return response.relabel_samples(inv_relabel)

    return _qubo_index_labels


def ising_index_labels(h_arg, J_arg):
    """Replaces the variable labels in h and J with integers.

    Args:
        h_arg (int): Location of the `h` argument in args.
        J_arg (int): Location of the `J` argument in args.

    Returns:
        Function that replaces the user-given Ising problem with one
        that has integer labels, runs the decorated method, then
        restores the original labels.

    Notes:
        The integer labels start with 0.

        If the given labels are orderable, the relabeling is applied
        to the nodes sorted lexicographically.

    Examples:
    Decorate functions like this::

        @ising_index_labels(1, 2)
        def sample_ising(self, h, J):
            pass

        @ising_index_labels(0, 1)
        def ising_energy(h, J, sample):
            pass

    """
    @decorator
    def _ising_index_labels(f, *args, **kw):

        # get h and J out of the given arguments
        h = args[h_arg]
        J = args[J_arg]

        # we want to know all of the nodes used in h and J
        nodes = set().union(*J) | set(h)

        # if the nodes are already index labeled from (0, n-1) then we are already
        # done
        if all(idx in nodes for idx in range(len(nodes))):
            return f(*args, **kw)

        # let's relabel them, sorting the node labels lexicographically.
        # In python 3, unlike types cannot be sorted, so let's just handle
        # that case with a try/catch.
        try:
            inv_relabel = dict(enumerate(sorted(nodes)))
        except TypeError:
            inv_relabel = dict(enumerate(nodes))
        relabel = {v: idx for idx, v in iteritems(inv_relabel)}

        # now apply this mapping to h and J
        h = {relabel[v]: bias for v, bias in iteritems(h)}
        J = {(relabel[u], relabel[v]): bias for (u, v), bias in iteritems(J)}

        # finally run the function with the new h, J
        newargs = [arg for arg in args]
        newargs[h_arg] = h
        newargs[J_arg] = J

        response = f(*newargs, **kw)

        # finally unapply the relabeling
        return response.relabel_samples(inv_relabel)

    return _ising_index_labels
