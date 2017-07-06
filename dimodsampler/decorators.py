import sys

from decorator import decorator

__all__ = ['qubo', 'ising', 'qubo_index_labels', 'ising_index_labels']

PY2 = sys.version_info[0] == 2
if PY2:
    range = xrange


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
        Decorate function like this:

        @qubo(1)
        def solve_qubo(self, Q):
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
        Decorate function like this:

        @ising(1, 2)
        def solve_ising(self, h, J):
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


def qubo_index_labels():
    """Replaces the labels in the dictionary of QUBO coefficients with
    integers.

    Returns:
        Function that replaces the user-given Q with one with integer
        labels, runs the decorated method, then restores the previous
        labels.

    Notes:
        The integer labels start with 0.

        The relabelling is applied to the given labels sorted
        lexigraphically.

    """
    @decorator
    def _qubo_index_labels(f, *args, **kw):

        Q = args[1]

        # get all of the node labels from Q
        nodes = set.union(*[set(edge) for edge in Q])

        # if the nodes are already index labelled (from 0, n) then we are already
        # done
        if all(idx in nodes for idx in range(len(nodes))):
            return f(*args, **kw)

        # let's relabel them, sorting the node labels lexigraphically
        relabel = {node: idx for idx, node in enumerate(sorted(nodes))}
        inv_relabel = {relabel[node]: node for node in relabel}
        newQ = {(relabel[node0], relabel[node1]): Q[(node0, node1)]
                for node0, node1 in Q}

        newargs = [arg for arg in args]
        newargs[1] = newQ

        solutions = f(*newargs, **kw)

        return solutions.relabel_variables(inv_relabel)

    return _qubo_index_labels


def ising_index_labels():
    """Replaces the variable labels in h and J with integers.

    Returns:
        Function that replaces the user-given Ising problem with one
        that has integer labels, runs the decorated method, then
        restores the original labels.

    Notes:
        The integer labels start with 0.

        The relabelling is applied to the given labels sorted
        lexigraphically.

    """
    @decorator
    def _ising_index_labels(f, *args, **kw):

        # get h and J out of the given arguments
        h = args[1]
        J = args[2]

        # we want to know all of the variables used
        variables = reduce(set.union, ({n0, n1} for n0, n1 in J), set())
        variables.update(h)

        # if all of the variables are already index-labelled, then we don't need to do
        # anything
        if all(idx in variables for idx in range(len(variables))):
            return f(*args, **kw)

        # Let's make the mapping, we do this by sorting the current labels lexigraphically
        relabel = {var: idx for idx, var in enumerate(sorted(variables))}
        inv_relabel = {relabel[var]: var for var in relabel}

        # now apply this mapping to h and J
        h = {relabel[var]: h[var] for var in h}
        J = {(relabel[v0], relabel[v1]): J[(v0, v1)] for v0, v1 in J}

        # finally run the function with the new h, J
        newargs = [arg for arg in args]
        newargs[1] = h
        newargs[2] = J

        # run the solver
        response = f(*newargs, **kw)

        # finally unapply the relabelling
        return response.relabel_variables(inv_relabel)
    return _ising_index_labels
