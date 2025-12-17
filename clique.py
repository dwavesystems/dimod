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

import networkx as nx
import dwave_networkx as dnx

from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = ["maximum_clique", "clique_number", "is_clique"]

@binary_quadratic_model_sampler(1)
def maximum_clique(G, sampler=None, lagrange=2.0, **sampler_args):
    r"""Returns an approximate maximum clique.

    A clique in an undirected graph, G = (V, E), is a subset of the vertex set
    :math:`C \subseteq V` such that for every two vertices in C there exists an edge
    connecting the two. This is equivalent to saying that the subgraph
    induced by C is complete (in some cases, the term clique may also refer
    to the subgraph). A maximum clique is a clique of the largest
    possible size in a given graph.

    This function works by finding the maximum independent set of the compliment
    graph of the given graph G which is equivalent to finding maximum clique.
    It defines a QUBO with ground states corresponding
    to a maximum weighted independent set and uses the sampler to sample from it.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a maximum clique.

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
    clique_nodes : list
       List of nodes that form a maximum clique, as
       determined by the given sampler.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    References
    ----------

    `Maximum Clique on Wikipedia <https://en.wikipedia.org/wiki/Clique_(graph_theory)>`_

    `Independent Set on Wikipedia <https://en.wikipedia.org/wiki/Independent_set_(graph_theory)>`_

    `QUBO on Wikipedia <https://en.wikipedia.org/wiki/Quadratic_unconstrained_binary_optimization>`_

    Lucas, A. (2014). Ising formulations of many NP problems. 
    Frontiers in Physics, Volume 2, Article 5.
    """
    if G is None:
        raise ValueError("Expected NetworkX graph!")

    # finding the maximum clique in a graph is equivalent to finding
    # the independent set in the complementary graph
    complement_G = nx.complement(G)
    return dnx.maximum_independent_set(complement_G, sampler, lagrange, **sampler_args)


@binary_quadratic_model_sampler(1)
def clique_number(G, sampler=None, lagrange=2.0, **sampler_args):
    r"""Returns the number of vertices in the maximum clique of a graph.

    A maximum clique is a clique of the largest possible size in a given graph.
    The clique number math:`\omega(G)` of a graph G is the number of
    vertices in a maximum clique in G. The intersection number of
    G is the smallest number of cliques that together cover all edges of G.

    This function works by finding the maximum independent set of the compliment
    graph of the given graph G which is equivalent to finding maximum clique.
    It defines a QUBO with ground states corresponding
    to a maximum weighted independent set and uses the sampler to sample from it.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a maximum clique.

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
    clique_nodes : list
       List of nodes that form a maximum clique, as
       determined by the given sampler.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    References
    ----------

    `Maximum Clique on Wikipedia <https://en.wikipedia.org/wiki/Clique_(graph_theory)>`_
    """
    return len(maximum_clique(G, sampler, lagrange, **sampler_args))

def is_clique(G, clique_nodes):
    """Determines whether the given nodes form a clique.

    A clique is a subset of nodes of an undirected graph such that every two
    distinct nodes in the clique are adjacent.

    Parameters
    ----------
    G : NetworkX graph
       The graph on which to check the clique nodes.

    clique_nodes : list
       List of nodes that form a clique, as
       determined by the given sampler.

    Returns
    -------
    is_clique : bool
        True if clique_nodes forms a clique.

    Example
    -------
    This example checks two sets of nodes, both derived from a
    single Chimera unit cell, for an independent set. The first set is
    the horizontal tile's nodes; the second has nodes from the horizontal and
    verical tiles.

    >>> import dwave_networkx as dnx
    >>> G = dnx.chimera_graph(1, 1, 4)
    >>> dnx.is_clique(G, [0, 1, 2, 3])
    False
    >>> dnx.is_clique(G, [0, 4])
    True
    """
    for x in clique_nodes:
        for y in clique_nodes:
            if x != y:
                if not(G.has_edge(x,y)):
                    return False
    return True
