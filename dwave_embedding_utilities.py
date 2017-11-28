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

import itertools
import random
from collections import Counter
import sys

_PY2 = sys.version_info[0] == 2
if _PY2:
    range = xrange
    zip = itertools.izip

    def iteritems(d):
        return d.iteritems()

    def itervalues(d):
        return d.itervalues()
else:
    def iteritems(d):
        return d.items()

    def itervalues(d):
        return d.values()

__all__ = ['target_to_source', 'chain_break_frequency', 'embed_ising',
           'edgelist_to_adjacency',
           'unembed_samples',
           'discard', 'majority_vote', 'weighted_random', 'minimize_energy']

__version__ = '0.1.0'
__author__ = 'D-Wave Systems Inc.'
__description__ = 'Utilities to manage embedding for the D-Wave System'
__authoremail__ = 'acondello@dwavesys.com'


def target_to_source(target_adjacency, embedding):
    """Derive the source adjacency from an embedding and target adjacency.

    Args:
        target_adjacency (dict/:class:`networkx.Graph`): A dict of the form
            {v: Nv, ...} where v is a node in the target graph and Nv is the
            neighbors of v as an iterable. This can also be a networkx graph.
        embedding (dict): A mapping from a source graph to a target graph.


    Returns:
        dict: The adjacency of the source graph.

    Raises:
        ValueError: If any node in the target_adjacency is assigned more
            than  one node in the source graph by embedding.

    """
    # the nodes in the source adjacency are just the keys of the embedding
    adj = {v: set() for v in embedding}

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

            adj[n].add(m)
            adj[m].add(n)

    return adj


def embed_ising(linear, quadratic, embedding, adjacency, chain_strength=1.0):
    """Embeds a logical Ising model onto another graph via an embedding.

    Args:
        linear (dict): The linear biases to be embedded. Should be a dict of
            the form {v: bias, ...} where v is a variable in the source model
            and bias is the linear bias associated with v.
        quadratic (dict): The quadratic biases to be embedded. Should be a dict
            of the form {(u, v): bias, ...} where u, v are variables in the
            source model and bias is the quadratic bias associated with (u, v).
        embedding (dict): The mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.
        adjacency (dict/:class:`networkx.Graph`): The adjacency dict of the target
            graph. Should be a dict of the form {s: Ns, ...} where s is a variable
            in the target graph and Ns is the set of neighbours of s.
        chain_strength (float, optional): The quadratic bias that should be used
            to create chains.

    Returns:
        (dict, dict, dict): A 3-tuple containing:

            dict: The linear biases of the target problem. In the form {s: bias, ...}
            where s is a node in the target graph and bias is the associated linear bias.

            dict: The quadratic biases of the target problem. A dict of the form
            {(s, t): bias, ...} where (s, t) is an edge in the target graph and bias is
            the associated quadratic bias.

            dict: The quadratic biases that induce the variables in the target problem to
            act as one. A dict of the form {(s, t): -chain_strength, ...} which
            is the quadratic biases associated with the chains.

    Examples:
        >>> source_linear = {'a': 1, 'b': 1}
        >>> source_quadratic = {('a', 'b'): -1}
        >>> embedding = {'a': [0, 1], 'b': [2]}
        >>> target_adjacency = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        >>> target_linear, target_quadratic, chain_quadratic = embed_ising(
        ...     source_linear, source_quadratic, embedding, target_adjacency)
        >>> target_linear
        {0: 0.5, 1: 0.5, 2: 1.0}
        >>> target_quadratic
        {(0, 2): -0.5, (1, 2): -0.5}
        >>> chain_quadratic
        {(0, 1): -1.0}

    """

    # ok, let's begin with the linear biases.
    # we spread the value of h evenly over the chain
    emb_h = {v: 0. for v in adjacency}
    for v, bias in iteritems(linear):
        try:
            chain_variables = embedding[v]
        except KeyError:
            raise ValueError('no embedding provided for source variable {}'.format(v))

        b = bias / len(chain_variables)

        for s in chain_variables:
            try:
                emb_h[s] += b
            except KeyError:
                raise ValueError('chain variable {} not in adjacency'.format(s))

    # next up the quadratic biases.
    # We spread the quadratic biases evenly over the edges
    emb_J = {}
    for (u, v), bias in iteritems(quadratic):
        edges = set()

        if u not in embedding:
            raise ValueError('no embedding provided for source variable {}'.format(u))
        if v not in embedding:
            raise ValueError('no embedding provided for source variable {}'.format(v))

        for s in embedding[u]:
            for t in embedding[v]:
                try:
                    if s in adjacency[t] and (t, s) not in edges:
                        edges.add((s, t))
                except KeyError:
                    raise ValueError('chain variable {} not in adjacency'.format(s))

        if not edges:
            raise ValueError("no edges in target graph between source variables {}, {}".format(u, v))

        b = bias / len(edges)

        # in some cases the logical J can have (u, v) and (v, u) as inputs, so make
        # sure we are not doubling them up with our choice of ordering
        for s, t in edges:
            if (s, t) in emb_J:
                emb_J[(s, t)] += b
            elif (t, s) in emb_J:
                emb_J[(t, s)] += b
            else:
                emb_J[(s, t)] = b

    # finally we need to connect the nodes in the chains
    chain_J = {}
    for chain_variables in itervalues(embedding):
        chain_J.update(_embedding_to_chain(chain_variables, adjacency, chain_strength))

    return emb_h, emb_J, chain_J


def _embedding_to_chain(chain_variables, adjacency, chain_strength):
    """Converts an embedding into a chain while checking connected.
    chain_variables is an iterable of nodes that define the chain
    adjacency is a dict of sets
    chain_strength is numeric

    returned chain is an edge dict {(u, v): -chain_strength, ...}
    """
    chain = {}  # we will be adding the edges that make the chain here

    # do a breadth first search
    seen = set()
    next_level = {next(iter(chain_variables))}
    while next_level:
        this_level = next_level
        next_level = set()
        for v in this_level:
            if v not in seen:
                seen.add(v)

                for u in adjacency[v]:
                    if u not in chain_variables:
                        continue
                    next_level.add(u)
                    if u != v and (u, v) not in chain:
                        chain[(v, u)] = -chain_strength

    if len(chain_variables) != len(seen):
        raise ValueError('{} does not form a connected chain'.format(chain_variables))

    return chain


def chain_break_frequency(samples, embedding):
    """Determines the frequency of chain breaks in the given samples.

    Args:
        samples (iterable): An iterable of samples where each sample
            is a dict of the form {v: val, ...} where v is a variable
            in the target graph and val is the associated value as
            determined by a binary quadratic model sampler.
        embedding (dict): The mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.

    Returns:
        dict: The frequency of chain breaks in the form {v: f, ...} where v
        is a variable in the source graph and frequency is the fraction
        of chains that were broken as a float.

    """
    counts = {v: 0 for v in embedding}
    total = 0
    for sample in samples:
        for v, chain in iteritems(embedding):
            vals = [sample[u] for u in chain]

            if not _all_equal(vals):
                counts[v] += 1
        total += 1

    return {v: counts[v] / total for v in embedding}


def unembed_samples(samples, embedding, chain_break_method=None, **method_args):
    """Return samples over the variables in the source graph.

    Args:
        samples (iterable): An iterable of samples where each sample
            is a dict of the form {v: val, ...} where v is a variable
            in the target model and val is the associated value as
            determined by a binary quadratic model sampler.
        embedding (dict): The mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a node in the
            source graph and s is a node in the target graph.
        chain_break_method (function, optional): The method used to resolve chain
            breaks. Default is :method:`majority_vote`.
        method_args**: Additional keyword args are passed to the chain_break_method.

    Returns:
        list: A list of unembedded samples. Each sample is a dict of the form
        {v: val, ...} where v is a variable in the source graph and val
        is the value associated with the variable.

    """
    if chain_break_method is None:
        if 'linear' in method_args or 'quadratic' in method_args:
            chain_break_method = minimize_energy
        else:
            chain_break_method = majority_vote
    return list(itertools.chain(*(chain_break_method(sample, embedding, **method_args) for sample in samples)))


def discard(sample, embedding):
    """Discards the sample if broken.

    Args:
        sample (dict): A sample of the form {v: val, ...} where v is
            a variable in the target graph and val is the associated value as
            determined by a binary quadratic model sampler.
        embedding (dict): The mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a node in the
            source graph and s is a node in the target graph.

    Yields:
        dict: The unembedded sample is no chains were broken.

    """
    unembeded = {}

    for v, chain in iteritems(embedding):
        vals = [sample[u] for u in chain]

        if _all_equal(vals):
            unembeded[v] = vals.pop()
        else:
            return

    yield unembeded


def majority_vote(sample, embedding):
    """Determines the sample values by majority vote.

    Args:
        sample (dict): A sample of the form {v: val, ...} where v is
            a variable in the target graph and val is the associated value as
            determined by a binary quadratic model sampler.
        embedding (dict): The mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a node in the
            source graph and s is a node in the target graph.

    Yields:
        dict: The unembedded sample. When there is a chain break, the value
        is chosen to match the most common value in the chain.

    """
    unembeded = {}

    for v, chain in iteritems(embedding):
        vals = [sample[u] for u in chain]

        if _all_equal(vals):
            unembeded[v] = vals.pop()
        else:
            unembeded[v] = _most_common(vals)

    yield unembeded


def weighted_random(sample, embedding):
    """Determines the sample values by weighed random choice.

    Args:
        sample (dict): A sample of the form {v: val, ...} where v is
            a variable in the target graph and val is the associated value as
            determined by a binary quadratic model sampler.
        embedding (dict): The mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a node in the
            source graph and s is a node in the target graph.

    Yields:
        dict: The unembedded sample. When there is a chain break, the value
        is chosen randomly, weighted by the frequency of the values
        within the chain.

    """
    unembeded = {}

    for v, chain in iteritems(embedding):
        vals = [sample[u] for u in chain]

        # pick a random element uniformly from all vals, this weights them by
        # the proportion of each
        unembeded[v] = random.choice(vals)

    yield unembeded


def minimize_energy(sample, embedding, linear=None, quadratic=None):
    """Determines the sample values by minimizing the local energy.

    Args:
        sample (dict): A sample of the form {v: val, ...} where v is
            a variable in the target graph and val is the associated value as
            determined by a binary quadratic model sampler.
        embedding (dict): The mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a node in the
            source graph and s is a node in the target graph.
        linear (dict): The linear biases of the source model. Should be a dict of
            the form {v: bias, ...} where v is a variable in the source model
            and bias is the linear bias associated with v.
        quadratic (dict): The quadratic biases of the source model. Should be a dict
            of the form {(u, v): bias, ...} where u, v are variables in the
            source model and bias is the quadratic bias associated with (u, v).

    Yields:
        dict: The unembedded sample. When there is a chain break, the value
        is chosen to minimize the energy relative to its neighbors.

    """
    if linear is None and quadratic is None:
        raise TypeError("the minimize_energy method requires `linear` and `quadratic` keyword arguments")
    elif linear is None:
        linear = {v: 0. for v in embedding}
    elif quadratic is None:
        quadratic = {}

    unembeded = {}
    broken = {}  # keys are the broken source variables, values are the energy contributions

    vartype = set(itervalues(sample))
    if len(vartype) > 2:
        raise ValueError("sample has more than two different values")

    # first establish the values of all of the unbroken chains
    for v, chain in iteritems(embedding):
        vals = [sample[u] for u in chain]

        if _all_equal(vals):
            unembeded[v] = vals.pop()
        else:
            broken[v] = linear[v]  # broken tracks the linear energy

    # now, we want to determine the energy for each of the broken variable
    # as much as we can
    for (u, v), bias in iteritems(quadratic):
        if u in unembeded and v in broken:
            broken[v] += unembeded[u] * bias
        elif v in unembeded and u in broken:
            broken[u] += unembeded[v] * bias

    # in order of energy contribution, pick spins for the broken variables
    while broken:
        v = max(broken, key=lambda u: abs(broken[u]))  # biggest energy contribution

        # get the value from vartypes that minimizes the energy
        val = min(vartype, key=lambda b: broken[v] * b)

        # set that value and remove it from broken
        unembeded[v] = val
        del broken[v]

        # add v's energy contribution to all of the nodes it is connected to
        for u in broken:
            if (u, v) in quadratic:
                broken[u] += val * quadratic[(u, v)]
            if (v, u) in quadratic:
                broken[u] += val * quadratic[(v, u)]

    yield unembeded


def _all_equal(iterable):
    """True if all values in `iterable` are equal, else False."""
    iterator = iter(iterable)
    first = next(iterator)
    return all(first == rest for rest in iterator)


def _most_common(iterable):
    """Returns the most common element in `iterable`."""
    data = Counter(iterable)
    return max(data, key=data.__getitem__)


def edgelist_to_adjacency(edgelist):
    """Converts an iterator of edges to an adjacency dict.

    Args:
        edgelist (iterable): An iterator over 2-tuples where
            each 2-tuple is an edge.

    Returns:
        dict: The adjacency dict. A dict of the form {v: Nv, ...} where
        v is a node in a graph and Nv is the neighbors of v as an set.

    """
    adj = dict()
    for u, v in edgelist:
        if u in adj:
            adj[u].add(v)
        else:
            adj[u] = {v}
        if v in adj:
            adj[v].add(u)
        else:
            adj[v] = {u}
    return adj
