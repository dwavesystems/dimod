"""
TODO
"""
import sys

from decorator import decorator

__all__ = ['solve_qubo_api', 'solve_ising_api',
           'qubo_index_labels', 'ising_index_labels']

# compatibility for python 2/3
if sys.version_info[0] == 2:
    range = xrange


def solve_qubo_api(Q_arg=1):
    """Provides input checking for QUBO methods.

    Returns:
        Function that checks the form of inputs to confirm that they
        match the form specified in TODO.

    Raises:
        TypeError: If `Q` is not a dict.

    """
    @decorator
    def _solve_qubo(f, *args, **kw):
        Q = args[Q_arg]
        if not isinstance(Q, dict):
            raise TypeError("expected first input 'Q' to be of type 'dict'")

        return f(*args, **kw)
    return _solve_qubo


def solve_ising_api(h_arg=1, J_arg=2):
    """Provides input checking for Ising methods.

    Returns:
        Function that checks the form of the Ising inputs to confirm
        TODO

    Raises:
        TypeError: If `J` is not a dict.

        ValueError: If the keys of `J` are not edges, that is tuples
        of form (var1, var2)

        ValueError: For each edge (v0, v1) in `J`, either expect v0 and v1
        to be keys of `h` if `h` is a dict, or v0 < len(h) and v1 < len(h).

        ValueError: Variables cannot have edges to themselves.

    """
    @decorator
    def _solve_ising(f, *args, **kw):

        h = args[h_arg]
        J = args[J_arg]

        if isinstance(h, (list, tuple)):
            h = {idx: bias for idx, bias in enumerate(h)}
            args[h_arg] = h

        if not isinstance(J, dict):
            raise TypeError("expected input 'J' to be a 'dict'")

        if any(len(edge) != 2 for edge in J):
            raise ValueError("each key in J should be an edge of form (var1, var2)")

        for edge in J:
            for v in edge:
                if v not in h:
                    raise ValueError("recieved unexpected variable '{}' in 'J'. ".format(v))

            v0, v1 = edge
            if v0 == v1:
                raise ValueError("'({}, {})' is not an allowed edge.".format(v0, v1))

        return f(*args, **kw)
    return _solve_ising


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
        nodes = reduce(set.union, ({n0, n1} for n0, n1 in Q))

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
