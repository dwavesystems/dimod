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

import itertools
import numbers
import warnings

import dimod
import networkx as nx

from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = ['is_matching',
           'is_maximal_matching',
           'matching_bqm',
           'maximal_matching_bqm',
           'min_maximal_matching',
           'min_maximal_matching_bqm',
           ]


def matching_bqm(G):
    """Find a binary quadratic model for the graph's matchings.

    A matching is a subset of edges in which no node occurs more than
    once. This function returns a binary quadratic model (BQM) with ground
    states corresponding to the possible matchings of G.

    Finding valid matchings can be done in polynomial time, so finding matching
    with BQMs is generally inefficient.
    This BQM may be useful when combined with other constraints and objectives.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a matching.

    Returns
    -------
    bqm : :class:`dimod.BinaryQuadraticModel`
        A binary quadratic model with ground states corresponding to a
        matching. The variables of the BQM are the edges of `G` as frozensets.
        The BQM's ground state energy is 0 by construction.
        The energy of the first excited state is 1.

    """
    bqm = dimod.BinaryQuadraticModel.empty('BINARY')

    # add the edges of G as variables
    for edge in G.edges:
        bqm.add_variable(frozenset(edge), 0)

    for node in G:
        for edge0, edge1 in itertools.combinations(G.edges(node), 2):
            u = frozenset(edge0)
            v = frozenset(edge1)
            bqm.add_interaction(u, v, 1)

    return bqm


def maximal_matching_bqm(G, lagrange=None):
    """Find a binary quadratic model for the graph's maximal matchings.

    A matching is a subset of edges in which no node occurs more than
    once. A maximal matching is one in which no edges from G can be
    added without violating the matching rule.
    This function returns a binary quadratic model (BQM) with ground
    states corresponding to the possible maximal matchings of G.

    Finding maximal matchings can be done in polynomial time, so finding
    maximal matching with BQMs is generally inefficient.
    This BQM may be useful when combined with other constraints and objectives.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a maximal matching.

    lagrange : float (optional)
        The Lagrange multiplier for the matching constraint. Should be positive
        and greater than `max_degree - 2`.
        Defaults to `1.25 * (max_degree - 2)`.

    Returns
    -------
    bqm : :class:`dimod.BinaryQuadraticModel`
        A binary quadratic model with ground states corresponding to a maximal
        matching. The variables of the BQM are the edges of `G` as frozensets.
        The BQM's ground state energy is 0 by construction.

    """
    bqm = matching_bqm(G)

    if lagrange is None:
        delta = max((G.degree[v] for v in G), default=0)
        lagrange = max(1.25 * (delta - 2), 1)

    bqm.scale(lagrange)

    for node0, node1 in G.edges:
        # (1 - y_v - y_u + y_v*y_u) <- see paper

        bqm.offset += 1

        for edge in G.edges(node0):
            bqm.linear[frozenset(edge)] -= 1

        for edge in G.edges(node1):
            bqm.linear[frozenset(edge)] -= 1

        for edge0 in G.edges(node0):
            u = frozenset(edge0)
            for edge1 in G.edges(node1):
                v = frozenset(edge1)
                if u == v:
                    bqm.linear[u] += 1
                else:
                    bqm.add_interaction(u, v, 1)

    return bqm


def min_maximal_matching_bqm(G, maximal_lagrange=2, matching_lagrange=None):
    """Find a binary quadratic model for the graph's minimum maximal matchings.

    A matching is a subset of edges in which no node occurs more than
    once. A maximal matching is one in which no edges from G can be
    added without violating the matching rule. A minimum maximal matching
    is a maximal matching that contains the smallest possible number of edges.
    This function returns a binary quadratic model (BQM) with ground
    states corresponding to the possible maximal matchings of G.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum maximal matching.

    maximal_lagrange : float (optional, default=2)
        The Lagrange multiplier for the maximal constraint. Should be greater
        than 1.

    matching_lagrange : float (optional)
        The Lagrange multiplier for the matching constraint. Should be positive
        and greater than `maximal_lagrange * max_degree - 2`.
        Defaults to `1.25 * (maximal_lagrange * max_degree - 2)`.

    Returns
    -------
    bqm : :class:`dimod.BinaryQuadraticModel`
        A binary quadratic model with ground states corresponding to a
        minimum maximal matching. The variables of the BQM are the edges
        of `G` as frozensets.

    """

    if matching_lagrange is not None:
        # we're going to scale the bqm by maximal_matching so undo that
        # for maximal_lagrange
        matching_lagrange /= maximal_lagrange
    bqm = maximal_matching_bqm(G, lagrange=matching_lagrange)
    bqm.scale(maximal_lagrange)

    for v in bqm.variables:
        bqm.linear[v] += 1

    return bqm


@binary_quadratic_model_sampler(1)
def maximal_matching(G, sampler=None, **sampler_args):
    """Finds an approximate maximal matching.

    Defines a QUBO with ground states corresponding to a maximal
    matching and uses the sampler to sample from it.

    A matching is a subset of edges in which no node occurs more than
    once. A maximal matching is one in which no edges from G can be
    added without violating the matching rule.

    Finding maximal matchings can be done is polynomial time, so this method
    is only useful pedagogically.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a maximal matching.

    sampler
        A binary quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    matching : set
        A maximal matching of the graph.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    References
    ----------

    `Matching on Wikipedia <https://w.wiki/r9s>`_

    `QUBO on Wikipedia <https://w.wiki/r9t>`_

    Based on the formulation presented in [Luc2014]_.

    """
    if not G.edges:
        return set()

    bqm = maximal_matching_bqm(G)
    sampleset = sampler.sample(bqm, **sampler_args)
    sample = sampleset.first.sample

    # the matching are the edges that are 1 in the sample
    return set(tuple(edge) for edge, val in sample.items() if val > 0)


@binary_quadratic_model_sampler(1)
def min_maximal_matching(G, sampler=None, **sampler_args):
    """Returns an approximate minimum maximal matching.

    Defines a QUBO with ground states corresponding to a minimum
    maximal matching and uses the sampler to sample from it.

    A matching is a subset of edges in which no node occurs more than
    once. A maximal matching is one in which no edges from G can be
    added without violating the matching rule. A minimum maximal
    matching is the smallest maximal matching for G.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum maximal matching.

    sampler
        A binary quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    matching : set
        A minimum maximal matching of the graph.

    Example
    -------
    This example uses a sampler from
    `dimod <https://github.com/dwavesystems/dimod>`_ to find a minimum maximal
    matching for a Chimera unit cell.

    >>> import dimod
    >>> sampler = dimod.ExactSolver()
    >>> G = dnx.chimera_graph(1, 1, 4)
    >>> matching = dnx.min_maximal_matching(G, sampler)

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    References
    ----------

    `Matching on Wikipedia <https://w.wiki/r9s>`_

    `QUBO on Wikipedia <https://w.wiki/r9t>`_

    Lucas, A. (2014). Ising formulations of many NP problems.
    Frontiers in Physics, Volume 2, Article 5.

    """
    if not G.edges:
        return set()

    bqm = min_maximal_matching_bqm(G)
    sampleset = sampler.sample(bqm, **sampler_args)
    sample = sampleset.first.sample

    # the matching are the edges that are 1 in the sample
    return set(tuple(edge) for edge, val in sample.items() if val > 0)


def is_matching(edges):
    """Determine whether the given set of edges is a matching.

    Deprecated in favour of :func:`networkx.is_matching`.
    """
    warnings.warn("This method is deprecated, please use NetworkX's"
                  "nx.is_matching(G, edges) rather than dwave-networkx's "
                  "dnx.is_matching(edges)", DeprecationWarning, stacklevel=2)
    return len(set().union(*edges)) == len(edges) * 2


def is_maximal_matching(G, matching):
    """Determine whether the given set of edges is a maximal matching.

    Deprecated in favour of :func:`networkx.is_matching`.
    """
    warnings.warn("This method is deprecated, please use NetworkX's"
                  "nx.is_maximal_matching(G, edges) rather than "
                  "dwave-networkx's dnx.is_maximal_matching(G, edges)",
                  DeprecationWarning, stacklevel=2)
    touched_nodes = set().union(*matching)

    # first check if a matching
    if len(touched_nodes) != len(matching) * 2:
        return False

    # now for each edge, check that at least one of its variables is
    # already in the matching
    for (u, v) in G.edges:
        if u not in touched_nodes and v not in touched_nodes:
            return False

    return True
