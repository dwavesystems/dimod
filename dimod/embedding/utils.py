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
# ================================================================================================

from __future__ import division, absolute_import

from six import iteritems
import numpy as np

from dimod.response import Response
from dimod.embedding.chain_breaks import broken_chains


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
    """Determine the frequency of chain breaks in the given samples.

    Args:
        samples (array-like/:obj:`.Response`):
            Matrix of samples or a dimod response object.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: {t, ...}, ...},
            where s is a source-model variable and t is a target-model variable.

    Returns:
        dict: Frequency of chain breaks as a dict in the form {s: f, ...},  where s
        is a variable in the source graph, and frequency, a float, is the fraction
        of broken chains.

    Examples:
        This example embeds a single source node, 'a', as a chain of two target nodes (0, 1)
        and uses :func:`.chain_break_frequency` to show that out of two synthetic samples,
        one ([-1, +1]) represents a broken chain.

        >>> import dimod
        >>> import numpy as np
        >>> samples = np.array([[-1, +1], [+1, +1]])
        >>> embedding = {'a': {0, 1}}
        >>> print(dimod.chain_break_frequency(samples, embedding)['a'])
        0.5


        This example embeds a single source node (0) as a chain of two target nodes (a, b)
        and uses :func:`.chain_break_frequency` to show that out of two samples in a
        dimod response, one ({'a': 1, 'b': 0}) represents a broken chain.

        >>> import dimod
        ...
        >>> response = dimod.Response.from_samples([{'a': 1, 'b': 0}, {'a': 0, 'b': 0}],
        ...                                        {'energy': [1, 0]}, {}, dimod.BINARY)
        >>> embedding = {0: {'a', 'b'}}
        >>> print(dimod.chain_break_frequency(response, embedding)[0])
        0.5

    """
    if isinstance(samples, Response):
        if samples.variable_labels is not None:
            label_to_idx = samples.label_to_idx
            embedding = {v: {label_to_idx[u] for u in chain} for v, chain in iteritems(embedding)}
        samples = samples.record.sample
    else:
        samples = np.asarray(samples, dtype=np.int8)

    if not embedding:
        return {}

    variables, chains = zip(*embedding.items())

    broken = broken_chains(samples, chains)

    freq = {v: float(broken[:, cidx].mean()) for cidx, v in enumerate(variables)}

    return freq


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
