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

import math
import itertools

import networkx as nx

from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = ["is_vertex_coloring",
           "is_cycle",
           "min_vertex_color",
           "min_vertex_coloring",  # alias for min_vertex_color
           "min_vertex_color_qubo",
           "vertex_color",
           "vertex_color_qubo",
           ]


@nx.utils.decorators.nodes_or_number(1)
def vertex_color_qubo(G, colors):
    """Return the QUBO with ground states corresponding to a vertex coloring.

    If `V` is the set of nodes, `E` is the set of edges and `C` is the set of
    colors the resulting qubo will have:

    * :math:`|V|*|C|` variables/nodes
    * :math:`|V|*|C|*(|C| - 1) / 2 + |E|*|C|` interactions/edges

    The QUBO has ground energy :math:`-|V|` and an infeasible gap of 1.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum vertex coloring.

    colors : int/sequence
        The colors. If an int, the colors are labelled `[0, n)`. The number of
        colors must be greater or equal to the chromatic number of the graph.

    Returns
    -------
    QUBO : dict
        The QUBO with ground states corresponding to valid colorings of the
        graph. The QUBO variables are labelled `(v, c)` where `v` is a node
        in `G` and `c` is a color. In the ground state of the QUBO, a variable
        `(v, c)` has value 1 if `v` should be colored `c` in a valid coloring.


    """
    _, colors = colors

    Q = {}

    # enforce that each variable in G has at most one color
    for v in G.nodes:
        # 1 in k constraint
        for c in colors:
            Q[(v, c), (v, c)] = -1

        for c0, c1 in itertools.combinations(colors, 2):
            Q[(v, c0), (v, c1)] = 2

    # enforce that adjacent nodes do not have the same color
    for u, v in G.edges:
        # NAND constraint
        for c in colors:
            Q[(u, c), (v, c)] = 1

    return Q


@binary_quadratic_model_sampler(2)
def vertex_color(G, colors, sampler=None, **sampler_args):
    """Returns an approximate vertex coloring.

    Vertex coloring is the problem of assigning a color to the
    vertices of a graph in a way that no adjacent vertices have the
    same color.

    Defines a QUBO [Dah2013]_ with ground states corresponding to valid
    vertex colorings and uses the sampler to sample from it.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum vertex coloring.

    colors : int/sequence
        The colors. If an int, the colors are labelled `[0, n)`. The number of
        colors must be greater or equal to the chromatic number of the graph.

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
    coloring : dict
        A coloring for each vertex in G such that no adjacent nodes
        share the same color. A dict of the form {node: color, ...}

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """
    Q = vertex_color_qubo(G, colors)

    # get the lowest energy sample
    sample = sampler.sample_qubo(Q, **sampler_args).first.sample

    return {v: c for (v, c), val in sample.items() if val}


def _chromatic_number_upper_bound(G):
    # tries to determine an upper bound on the chromatic number of G
    # Assumes G is not complete

    if not nx.is_connected(G):
        return max((_chromatic_number_upper_bound(G.subgraph(c))
                    for c in nx.connected_components(G)))

    n_nodes = len(G.nodes)
    n_edges = len(G.edges)

    # chi * (chi - 1) <= 2 * |E|
    quad_bound = math.ceil((1 + math.sqrt(1 + 8 * n_edges)) / 2)

    if n_nodes % 2 == 1 and is_cycle(G):
        # odd cycle graphs need three colors
        bound = 3
    elif n_nodes > 2:
        try:
            import numpy as np
        except ImportError:
            # chi <= max degree, unless it is complete or a cycle graph of odd length,
            # in which case chi <= max degree + 1 (Brook's Theorem)
            bound = max(G.degree(node) for node in G)
        else:
            # Let A be the adj matrix of G (symmetric, 0 on diag). Let theta_1
            # be the largest eigenvalue of A. Then chi <= theta_1 + 1 with
            # equality iff G is complete or an odd cycle.
            # this is strictly better than brooks theorem
            # G is real symmetric, use eigvalsh for real valued output.
            bound = math.ceil(max(np.linalg.eigvalsh(nx.to_numpy_array(G))))
    else:
        # we know it's connected
        bound = n_nodes

    return min(quad_bound, bound)


def _chromatic_number_lower_bound(G):
    # find a random maximal clique and use that to determine a lower bound
    v = max(G, key=G.degree)

    clique = {v}
    for u in G[v]:
        if all(w in G[u] for w in clique):
            clique.add(u)

    return len(clique)


def min_vertex_color_qubo(G, chromatic_lb=None, chromatic_ub=None):
    """Return a QUBO with ground states corresponding to a minimum vertex
    coloring.

    Vertex coloring is the problem of assigning a color to the
    vertices of a graph in a way that no adjacent vertices have the
    same color. A minimum vertex coloring is the problem of solving
    the vertex coloring problem using the smallest number of colors.

    Defines a QUBO [Dah2013]_ with ground states corresponding to minimum
    vertex colorings and uses the sampler to sample from it.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum vertex coloring.

    chromatic_lb : int, optional
         A lower bound on the chromatic number. If one is not provided, a
         bound is calulcated.

    chromatic_ub : int, optional
        An upper bound on the chromatic number. If one is not provided, a bound
        is calculated.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    QUBO : dict
        The QUBO with ground states corresponding to minimum colorings of the
        graph. The QUBO variables are labelled `(v, c)` where `v` is a node
        in `G` and `c` is a color. In the ground state of the QUBO, a variable
        `(v, c)` has value 1 if `v` should be colored `c` in a valid coloring.

    """

    chi_ub = _chromatic_number_upper_bound(G)
    chromatic_ub = chi_ub if chromatic_ub is None else min(chi_ub, chromatic_ub)

    chib_lb = _chromatic_number_lower_bound(G)
    chromatic_lb = chib_lb if chromatic_lb is None else max(chib_lb, chromatic_lb)

    if chromatic_lb > chromatic_ub:
        raise RuntimeError("something went wrong when calculating the "
                           "chromatic number bounds")

    # our base QUBO is one with as many colors as we might need, so we use the
    # upper bound
    Q = vertex_color_qubo(G, int(chromatic_ub))

    if chromatic_lb != chromatic_ub:
        # we want to penalize the colors that we aren't sure that we need
        # we might need to use some of the colors, so we want to penalize
        # them in increasing amounts, linearly.

        num_penalized = chromatic_ub - chromatic_lb

        # we want evenly spaced penalties in (0, 1) without the endpoints
        weights = [p / (num_penalized + 1) for p in range(1, num_penalized + 1)]

        for p, c in zip(weights, range(chromatic_lb, chromatic_ub)):
            for v in G:
                Q[(v, c), (v, c)] += p

    return Q


@binary_quadratic_model_sampler(1)
def min_vertex_color(G, sampler=None, chromatic_lb=None, chromatic_ub=None,
                     **sampler_args):
    """Returns an approximate minimum vertex coloring.

    Vertex coloring is the problem of assigning a color to the
    vertices of a graph in a way that no adjacent vertices have the
    same color. A minimum vertex coloring is the problem of solving
    the vertex coloring problem using the smallest number of colors.

    Defines a QUBO [Dah2013]_ with ground states corresponding to minimum
    vertex colorings and uses the sampler to sample from it.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum vertex coloring.

    sampler
        A binary quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.

    chromatic_lb : int, optional
         A lower bound on the chromatic number. If one is not provided, a
         bound is calulcated.

    chromatic_ub : int, optional
        An upper bound on the chromatic number. If one is not provided, a bound
        is calculated.

    sampler_args
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    coloring : dict
        A coloring for each vertex in G such that no adjacent nodes
        share the same color. A dict of the form {node: color, ...}

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """

    Q = min_vertex_color_qubo(G, chromatic_lb=chromatic_lb,
                              chromatic_ub=chromatic_ub)

    # get the lowest energy sample
    sample = sampler.sample_qubo(Q, **sampler_args).first.sample

    return {v: c for (v, c), val in sample.items() if val}


# legacy name, alias
min_vertex_coloring = min_vertex_color


def is_cycle(G):
    """Determines whether the given graph is a cycle or circle graph.

    A cycle graph or circular graph is a graph that consists of a single cycle.

    https://en.wikipedia.org/wiki/Cycle_graph

    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    is_cycle : bool
        True if the graph consists of a single cycle.

    """
    if len(G) <= 2:
        return False

    trailing, leading = next(iter(G.edges))
    start_node = trailing

    # travel around the graph, checking that each node has degree exactly two
    # also track how many nodes were visited
    n_visited = 1
    while leading != start_node:
        neighbors = G[leading]

        if len(neighbors) != 2:
            return False

        node1, node2 = neighbors

        if node1 == trailing:
            trailing, leading = leading, node2
        else:
            trailing, leading = leading, node1

        n_visited += 1

    # if we haven't visited all of the nodes, then it is not a connected cycle
    return n_visited == len(G)


def is_vertex_coloring(G, coloring):
    """Determines whether the given coloring is a vertex coloring of graph G.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which the vertex coloring is applied.

    coloring : dict
        A coloring of the nodes of G. Should be a dict of the form
        {node: color, ...}.

    Returns
    -------
    is_vertex_coloring : bool
        True if the given coloring defines a vertex coloring; that is, no
        two adjacent vertices share a color.

    Example
    -------
    This example colors checks two colorings for a graph, G, of a single Chimera
    unit cell. The first uses one color (0) for the four horizontal qubits
    and another (1) for the four vertical qubits, in which case there are
    no adjacencies; the second coloring swaps the color of one node.

    >>> G = dnx.chimera_graph(1,1,4)
    >>> colors = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
    >>> dnx.is_vertex_coloring(G, colors)
    True
    >>> colors[4]=0
    >>> dnx.is_vertex_coloring(G, colors)
    False

   """
    return all(coloring[u] != coloring[v] for u, v in G.edges)
