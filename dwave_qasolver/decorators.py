"""
TODO
"""

from decorator import decorator

__all__ = ['solve_qubo_api', 'qubo_index_labels']


def solve_qubo_api():
    """Provides input checking for qubo methods.

    Returns:
        Function that checks the form of inputs to confirm that they
        match the form specified in TODO.

    Raises:
        TypeError: If `Q` is not a dict.

    """
    @decorator
    def _solve_qubo(f, *args, **kw):

        Q = args[1]
        if not isinstance(Q, dict):
            raise TypeError("expected first input 'Q' to be of type 'dict'")

        return f(*args, **kw)
    return _solve_qubo


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
        detailed_solution = args[2]

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

        if detailed_solution:
            raise NotImplemented
        else:
            return [{inv_relabel[idx]: soln[idx] for idx in soln}
                    for soln in solutions]

    return _qubo_index_labels
