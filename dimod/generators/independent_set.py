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

from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = ["maximum_weighted_independent_set",
           "maximum_weighted_independent_set_qubo",
           "maximum_independent_set",
           "is_independent_set",
           ]


@binary_quadratic_model_sampler(2)
def maximum_weighted_independent_set(G, weight=None, sampler=None, lagrange=2.0, **sampler_args):
    """Returns an approximate maximum weighted independent set.

    Defines a QUBO with ground states corresponding to a
    maximum weighted independent set and uses the sampler to sample
    from it.

    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges. A maximum
    independent set is an independent set of maximum total node weight.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a maximum cut weighted independent set.

    weight : string, optional (default None)
        If None, every node has equal weight. If a string, use this node
        attribute as the node weight. A node without this attribute is
        assumed to have max weight.

    sampler
        A binary quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.
        
    lagrange : optional (default 2)
        Lagrange parameter to weight constraints (no edges within set) 
        versus objective (largest set possible).

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    indep_nodes : list
       List of nodes that form a maximum weighted independent set, as
       determined by the given sampler.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    References
    ----------

    `Independent Set on Wikipedia <https://en.wikipedia.org/wiki/Independent_set_(graph_theory)>`_

    `QUBO on Wikipedia <https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization>`_

    Lucas, A. (2014). Ising formulations of many NP problems.
    Frontiers in Physics, Volume 2, Article 5.

    """
    # Get a QUBO representation of the problem
    Q = maximum_weighted_independent_set_qubo(G, weight, lagrange)

    # use the sampler to find low energy states
    response = sampler.sample_qubo(Q, **sampler_args)

    # we want the lowest energy sample
    sample = next(iter(response))

    # nodes that are spin up or true are exactly the ones in S.
    return [node for node in sample if sample[node] > 0]


@binary_quadratic_model_sampler(1)
def maximum_independent_set(G, sampler=None, lagrange=2.0, **sampler_args):
    """Returns an approximate maximum independent set.

    Defines a QUBO with ground states corresponding to a
    maximum independent set and uses the sampler to sample from
    it.

    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges. A maximum
    independent set is an independent set of largest possible size.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a maximum cut independent set.

    sampler
        A binary quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.
        
    lagrange : optional (default 2)
        Lagrange parameter to weight constraints (no edges within set) 
        versus objective (largest set possible).

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    indep_nodes : list
       List of nodes that form a maximum independent set, as
       determined by the given sampler.

    Example
    -------
    This example uses a sampler from
    `dimod <https://github.com/dwavesystems/dimod>`_ to find a maximum
    independent set for a graph of a Chimera unit cell created using the
    `chimera_graph()` function.

    >>> import dimod
    >>> sampler = dimod.SimulatedAnnealingSampler()
    >>> G = dnx.chimera_graph(1, 1, 4)
    >>> indep_nodes = dnx.maximum_independent_set(G, sampler)

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    References
    ----------

    `Independent Set on Wikipedia <https://en.wikipedia.org/wiki/Independent_set_(graph_theory)>`_

    `QUBO on Wikipedia <https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization>`_

    Lucas, A. (2014). Ising formulations of many NP problems.
    Frontiers in Physics, Volume 2, Article 5.

    """
    return maximum_weighted_independent_set(G, None, sampler, lagrange, **sampler_args)


def is_independent_set(G, indep_nodes):
    """Determines whether the given nodes form an independent set.

    An independent set is a set of nodes such that the subgraph
    of G induced by these nodes contains no edges.

    Parameters
    ----------
    G : NetworkX graph
       The graph on which to check the independent set.

    indep_nodes : list
       List of nodes that form a maximum independent set, as
       determined by the given sampler.

    Returns
    -------
    is_independent : bool
        True if indep_nodes form an independent set.

    Example
    -------
    This example checks two sets of nodes, both derived from a
    single Chimera unit cell, for an independent set. The first set is
    the horizontal tile's nodes; the second has nodes from the horizontal and
    verical tiles.

    >>> import dwave_networkx as dnx
    >>> G = dnx.chimera_graph(1, 1, 4)
    >>> dnx.is_independent_set(G, [0, 1, 2, 3])
    True
    >>> dnx.is_independent_set(G, [0, 4])
    False

    """
    return len(G.subgraph(indep_nodes).edges) == 0


def maximum_weighted_independent_set_qubo(G, weight=None, lagrange=2.0):
    """Return the QUBO with ground states corresponding to a maximum weighted independent set.

    Parameters
    ----------
    G : NetworkX graph

    weight : string, optional (default None)
        If None, every node has equal weight. If a string, use this node
        attribute as the node weight. A node without this attribute is
        assumed to have max weight.
        
    lagrange : optional (default 2)
        Lagrange parameter to weight constraints (no edges within set) 
        versus objective (largest set possible).

    Returns
    -------
    QUBO : dict
       The QUBO with ground states corresponding to a maximum weighted independent set.

    Examples
    --------

    >>> from dwave_networkx.algorithms.independent_set import maximum_weighted_independent_set_qubo
    ...
    >>> G = nx.path_graph(3)
    >>> Q = maximum_weighted_independent_set_qubo(G, weight='weight', lagrange=2.0)
    >>> Q[(0, 0)]
    -1.0
    >>> Q[(1, 1)]
    -1.0
    >>> Q[(0, 1)]
    2.0

    """

    # empty QUBO for an empty graph
    if not G:
        return {}

    # We assume that the sampler can handle an unstructured QUBO problem, so let's set one up.
    # Let us define the largest independent set to be S.
    # For each node n in the graph, we assign a boolean variable v_n, where v_n = 1 when n
    # is in S and v_n = 0 otherwise.
    # We call the matrix defining our QUBO problem Q.
    # On the diagnonal, we assign the linear bias for each node to be the negative of its weight.
    # This means that each node is biased towards being in S. Weights are scaled to a maximum of 1.
    # Negative weights are considered 0.
    # On the off diagnonal, we assign the off-diagonal terms of Q to be 2. Thus, if both
    # nodes are in S, the overall energy is increased by 2.
    cost = dict(G.nodes(data=weight, default=1))
    scale = max(cost.values())
    Q = {(node, node): min(-cost[node] / scale, 0.0) for node in G}
    Q.update({edge: lagrange for edge in G.edges})

    return Q
