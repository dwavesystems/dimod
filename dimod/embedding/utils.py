"""
D-Wave Embedding Utilities
==========================

This package provides functions that map samples between a source graph
and a target graph.

Terminology
-----------

**model** - A collection of variables with associated linear and
quadratic biases. Sometimes referred to in other projects as a **problem**.
In this project all models are expected to be spin-valued - that is the
variables in the model can be -1 or 1.

**graph** - A collection of nodes and edges. A graph can be derived
from a model; a node for each variable and an edge for each pair
of variables with a non-zero quadratic bias.

**source** - The model or induced graph that we wish to embed. Sometimes
referred to in other projects as the **logical** graph/model.

**target** - Embedding attempts to create a target model from a target
graph. The process of embedding takes a source model, derives the source
graph, maps the source graph to the target graph, then derives the target
model. Sometimes referred to in other projects at the **embedded** graph/model.

**chain** - A collection of nodes or variables in the target graph/model
that we want to act like a single node/variable.

**chain strength** - The magnitude of the negative quadratic bias applied
between variables within a chain.

Examples
--------

Imagine that we have a sampler which is structured as a 4-cycle graph.

.. code-block:: python

    import networkx as nx
    target_graph = nx.cycle_graph(4)
    # target_graph = {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {0, 2}}  # equivalent

We have a model on a 3-cycle that we wish to embed.

.. code-block:: python

    source_linear = {'a': 0., 'b': 0., 'c': 0.}
    source_quadratic = {('a', 'b'): 1., ('b', 'c'): 1., ('a', 'c'): 1.}

Finally, we have an embedding that maps a 3-cycle to a 4-cycle. In this
case we want variables 1, 2 in the target to behave as a single variable.

.. code-block:: python

    embedding = {'a': {0}, 'b': {1, 2}, 'c': {3}}

To get the target model, use the :func:`embed_ising` function.

.. code-block:: python

    target_linear, target_quadratic, chain_quadratic = embed_ising(
        source_linear, source_quadratic, embedding, target_graph)

Say that we sample from the target model using some sampler, we can then
umembed the samples using :func:`unembed_samples`.

.. code-block:: python

    samples = {0: -1, 1: -1, 2: 1, 3: 1}
    source_samples = unembed_samples(samples, embedding)

"""
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
        >>> source_adjacency = dimod.target_to_source(target_adjacency, embedding)
        >>> source_adjacency  # triangle
        {'a': {'b', 'c'}, 'b': {'a', 'c'}, 'c': {'a', 'b'}}

        This function also works with networkx graphs.

        >>> import networkx as nx
        >>> target_graph = nx.complete_graph(5)
        >>> embedding = {'a': {0, 1, 2}, 'b': {3, 4}}
        >>> dimod.target_to_source(target_graph, embedding)

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
        >>> chain_to_quadratic(chain, target_adjacency, 1)
        {(1, 2): -1}

    """
    quadratic = {}  # we will be adding the edges that make the chain here

    # do a breadth first search
    seen = set()
    next_level = {next(iter(chain))}
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
