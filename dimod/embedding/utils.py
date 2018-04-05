from __future__ import division, absolute_import

from six import iteritems
import numpy as np

from dimod.response import Response


__all__ = ['target_to_source', 'chain_to_quadratic', 'chain_break_frequency']


def target_to_source(target_adjacency, embedding):
    """Derive the source adjacency from an embedding and target adjacency.

    Args:
        target_adjacency (dict/:class:`networkx.Graph`):
            A dict of the form {v: Nv, ...} where v is a node in the target graph and Nv is the
            neighbors of v as an iterable. This can also be a networkx graph.

        embedding (dict):
            A mapping from a source graph to a target graph.

    Returns:
        dict: The adjacency of the source graph.

    Raises:
        ValueError: If any node in the target_adjacency is assigned more
            than  one node in the source graph by embedding.

    Examples:

        >>> target_adjacency = {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {0, 2}}  # a square graph
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> source_adjacency = dimod.embedding.target_to_source(target_adjacency, embedding)
        >>> source_adjacency  # triangle
        {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}

        This function also works with networkx graphs.

        >>> import networkx as nx
        >>> target_graph = nx.complete_graph(5)
        >>> embedding = {'a': {0, 1, 2}, 'b': {3, 4}}
        >>> dimod.embedding.target_to_source(target_graph, embedding)

    """
    # the nodes in the source adjacency are just the keys of the embedding
    source_adjacency = {v: set() for v in embedding}

    # we need the mapping from each node in the target to its source node
    reverse_embedding = {}
    for v, chain in iteritems(embedding):
        for u in chain:
            if u in reverse_embedding:
                raise ValueError("target node {} assigned to more than one source node".format(u))
            reverse_embedding[u] = v

    # v is node in target, n node in source
    for v, n in iteritems(reverse_embedding):
        neighbors = target_adjacency[v]

        # u is node in target
        for u in neighbors:

            # some nodes might not be assigned to chains
            if u not in reverse_embedding:
                continue

            # m is node in source
            m = reverse_embedding[u]

            if m == n:
                continue

            source_adjacency[n].add(m)
            source_adjacency[m].add(n)

    return source_adjacency


def chain_to_quadratic(chain, target_adjacency, chain_strength):
    """Determine the quadratic biases that induce the given chain.

    Args:
        chain (iterable):
            The variables that make up a chain.

        target_adjacency (dict/:class:`networkx.Graph`):
            Should be a dict of the form {s: Ns, ...} where s is a variable
            in the target graph and Ns is the set of neighbours of s.

        chain_strength (float):
            The magnitude of the quadratic bias that should be used to create chains.

    Returns:
        dict[edge, float]: The quadratic biases that induce the given chain.

    Raises:
        ValueError: If the variables in chain do not form a connected subgraph of target.

    Examples:
        >>> chain = {1, 2}
        >>> target_adjacency = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        >>> dimod.embedding.chain_to_quadratic(chain, target_adjacency, 1)
        {(1, 2): -1}

    """
    quadratic = {}  # we will be adding the edges that make the chain here

    # do a breadth first search
    seen = set()
    try:
        next_level = {next(iter(chain))}
    except StopIteration:
        raise ValueError("chain must have at least one variable")
    while next_level:
        this_level = next_level
        next_level = set()
        for v in this_level:
            if v not in seen:
                seen.add(v)

                for u in target_adjacency[v]:
                    if u not in chain:
                        continue
                    next_level.add(u)
                    if u != v and (u, v) not in quadratic:
                        quadratic[(v, u)] = -chain_strength

    if len(chain) != len(seen):
        raise ValueError('{} is not a connected chain'.format(chain))

    return quadratic


def chain_break_frequency(samples, embedding):
    """Determines the frequency of chain breaks in the given samples.

    Args:
        samples (array-like/:obj:`.Response`):
            A matrix of samples or a dimod response object.

        embedding (dict):
            The mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.

    Returns:
        dict: The frequency of chain breaks in the form {v: f, ...} where v
        is a variable in the source graph and frequency is the fraction
        of chains that were broken as a float.

    Examples:
        >>> samples = np.matrix([[-1, +1], [+1, +1]])
        >>> embedding = {'a': {0, 1}}
        >>> dimod.chain_break_frequency(samples, embedding)
        {'a': .5}

        From a dimod response:

        >>> response = dimod.Response.from_dicts([{'a': 1, 'b': 0}, {'a': 0, 'b': 0}], {'energy': [1, 0]})
        >>> embedding = {0: {'a', 'b'}}
        >>> dimod.chain_break_frequency(response, embedding)
        {0: .5}

    """
    if isinstance(samples, Response):
        if samples.variable_labels is not None:
            label_to_idx = samples.label_to_idx
            embedding = {v: {label_to_idx[u] for u in chain} for v, chain in iteritems(embedding)}
        samples = samples.samples_matrix
    else:
        samples = np.matrix(samples)

    f = {}
    for v, chain in iteritems(embedding):
        chain = list(chain)
        u = chain[0]
        f[v] = 1.0 - float(np.mean((samples[:, chain] == samples[:, u]).all(axis=1), dtype=float))

    return f


def edgelist_to_adjacency(edgelist):
    """Converts an iterator of edges to an adjacency dict.

    Args:
        edgelist (iterable):
            An iterator over 2-tuples where each 2-tuple is an edge.

    Returns:
        dict: The adjacency dict. A dict of the form {v: Nv, ...} where v is a node in a graph and
        Nv is the neighbors of v as an set.

    """
    adjacency = dict()
    for u, v in edgelist:
        if u in adjacency:
            adjacency[u].add(v)
        else:
            adjacency[u] = {v}
        if v in adjacency:
            adjacency[v].add(u)
        else:
            adjacency[v] = {u}
    return adjacency
