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

from random import random

import networkx as nx

from dwave.graphs.generators.pegasus import pegasus_coordinates
from dwave.graphs.generators.zephyr import zephyr_coordinates
from dwave.graphs.generators.chimera import chimera_coordinates

__all__ = ['is_almost_simplicial',
           'is_simplicial',
           'chimera_elimination_order',
           'pegasus_elimination_order',
           'zephyr_elimination_order',
           'max_cardinality_heuristic',
           'min_fill_heuristic',
           'min_width_heuristic',
           'treewidth_branch_and_bound',
           'minor_min_width',
           'elimination_order_width',
           ]


def is_simplicial(G, n):
    """Determines whether a node n in G is simplicial.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to check whether node n is simplicial.
    n : node
        A node in graph G.

    Returns
    -------
    is_simplicial : bool
        True if its neighbors form a clique.

    Examples
    --------
    This example checks whether node 0 is simplicial for two graphs: G, a
    single Chimera unit cell, which is bipartite, and K_5, the :math:`K_5`
    complete graph.

    >>> G = dwave.graphs.chimera_graph(1, 1, 4)
    >>> K_5 = nx.complete_graph(5)
    >>> dwave.graphs.is_simplicial(G, 0)
    False
    >>> dwave.graphs.is_simplicial(K_5, 0)
    True

    """
    return all(u in G[v] for u, v in itertools.combinations(G[n], 2))


def is_almost_simplicial(G, n):
    """Determines whether a node n in G is almost simplicial.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to check whether node n is almost simplicial.
    n : node
        A node in graph G.

    Returns
    -------
    is_almost_simplicial : bool
        True if all but one of its neighbors induce a clique

    Examples
    --------
    This example checks whether node 0 is simplicial or almost simplicial for
    a :math:`K_5` complete graph with one edge removed.

    >>> K_5 = nx.complete_graph(5)
    >>> K_5.remove_edge(1,3)
    >>> dwave.graphs.is_simplicial(K_5, 0)
    False
    >>> dwave.graphs.is_almost_simplicial(K_5, 0)
    True

    """
    for w in G[n]:
        if all(u in G[v] for u, v in itertools.combinations(G[n], 2) if u != w and v != w):
            return True
    return False


def minor_min_width(G):
    """Computes a lower bound for the treewidth of graph G.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to compute a lower bound on the treewidth.

    Returns
    -------
    lb : int
        A lower bound on the treewidth.

    Examples
    --------
    This example computes a lower bound for the treewidth of the :math:`K_7`
    complete graph.

    >>> K_7 = nx.complete_graph(7)
    >>> dwave.graphs.minor_min_width(K_7)
    6

    References
    ----------
    Based on the algorithm presented in [Gog2004]_.

    """
    # we need only deal with the adjacency structure of G. We will also
    # be manipulating it directly so let's go ahead and make a new one
    adj = {v: set(u for u in G[v] if u != v) for v in G}

    lb = 0  # lower bound on treewidth
    while len(adj) > 1:

        # get the node with the smallest degree
        v = min(adj, key=lambda v: len(adj[v]))

        # find the vertex u such that the degree of u is minimal in the neighborhood of v
        neighbors = adj[v]

        if not neighbors:
            # if v is a singleton, then we can just delete it
            del adj[v]
            continue

        def neighborhood_degree(u):
            Gu = adj[u]
            return sum(w in Gu for w in neighbors)

        u = min(neighbors, key=neighborhood_degree)

        # update the lower bound
        new_lb = len(adj[v])
        if new_lb > lb:
            lb = new_lb

        # contract the edge between u, v
        adj[v] = adj[v].union(n for n in adj[u] if n != v)
        for n in adj[v]:
            adj[n].add(v)
        for n in adj[u]:
            adj[n].discard(u)
        del adj[u]

    return lb


def min_fill_heuristic(G):
    """Computes an upper bound on the treewidth of graph G based on
    the min-fill heuristic for the elimination ordering.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to compute an upper bound for the treewidth.

    Returns
    -------
    treewidth_upper_bound : int
        An upper bound on the treewidth of the graph G.

    order : list
        An elimination order that induces the treewidth.

    Examples
    --------
    This example computes an upper bound for the treewidth of the :math:`K_4`
    complete graph.

    >>> K_4 = nx.complete_graph(4)
    >>> tw, order = dwave.graphs.min_fill_heuristic(K_4)

    References
    ----------
    Based on the algorithm presented in [Gog2004]_.

    """
    # we need only deal with the adjacency structure of G. We will also
    # be manipulating it directly so let's go ahead and make a new one
    adj = {v: set(u for u in G[v] if u != v) for v in G}

    num_nodes = len(adj)

    # preallocate the return values
    order = [0] * num_nodes
    upper_bound = 0

    for i in range(num_nodes):
        # get the node that adds the fewest number of edges when eliminated from the graph
        v = min(adj, key=lambda x: _min_fill_needed_edges(adj, x))

        # if the number of neighbours of v is higher than upper_bound, update
        dv = len(adj[v])
        if dv > upper_bound:
            upper_bound = dv

        # make v simplicial by making its neighborhood a clique then remove the
        # node
        _elim_adj(adj, v)
        order[i] = v

    return upper_bound, order


def _min_fill_needed_edges(adj, n):
    # determines how many edges would needed to be added to G in order
    # to make node n simplicial.
    e = 0  # number of edges needed
    for u, v in itertools.combinations(adj[n], 2):
        if u not in adj[v]:
            e += 1
    # We add random() which picks a value in the range [0., 1.). This is ok because the
    # e are all integers. By adding a small random value, we randomize which node is
    # chosen without affecting correctness.
    return e + random()


def min_width_heuristic(G):
    """Computes an upper bound on the treewidth of graph G based on
    the min-width heuristic for the elimination ordering.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to compute an upper bound for the treewidth.

    Returns
    -------
    treewidth_upper_bound : int
        An upper bound on the treewidth of the graph G.

    order : list
        An elimination order that induces the treewidth.

    Examples
    --------
    This example computes an upper bound for the treewidth of the :math:`K_4`
    complete graph.

    >>> K_4 = nx.complete_graph(4)
    >>> tw, order = dwave.graphs.min_width_heuristic(K_4)

    References
    ----------
    Based on the algorithm presented in [Gog2004]_.

    """
    # we need only deal with the adjacency structure of G. We will also
    # be manipulating it directly so let's go ahead and make a new one
    adj = {v: set(u for u in G[v] if u != v) for v in G}

    num_nodes = len(adj)

    # preallocate the return values
    order = [0] * num_nodes
    upper_bound = 0

    for i in range(num_nodes):
        # get the node with the smallest degree. We add random() which picks a value
        # in the range [0., 1.). This is ok because the lens are all integers. By
        # adding a small random value, we randomize which node is chosen without affecting
        # correctness.
        v = min(adj, key=lambda u: len(adj[u]) + random())

        # if the number of neighbours of v is higher than upper_bound, update
        dv = len(adj[v])
        if dv > upper_bound:
            upper_bound = dv

        # make v simplicial by making its neighborhood a clique then remove the
        # node
        _elim_adj(adj, v)
        order[i] = v

    return upper_bound, order


def max_cardinality_heuristic(G):
    """Computes an upper bound on the treewidth of graph G based on
    the max-cardinality heuristic for the elimination ordering.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to compute an upper bound for the treewidth.

    Returns
    -------
    treewidth_upper_bound : int
        An upper bound on the treewidth of the graph G.

    order : list
        An elimination order that induces the treewidth.

    Examples
    --------
    This example computes an upper bound for the treewidth of the :math:`K_4`
    complete graph.

    >>> K_4 = nx.complete_graph(4)
    >>> tw, order = dwave.graphs.max_cardinality_heuristic(K_4)

    References
    ----------
    Based on the algorithm presented in [Gog2004]_.

    """
    # we need only deal with the adjacency structure of G. We will also
    # be manipulating it directly so let's go ahead and make a new one
    adj = {v: set(u for u in G[v] if u != v) for v in G}

    num_nodes = len(adj)

    # preallocate the return values
    order = [0] * num_nodes
    upper_bound = 0

    # we will need to track the nodes and how many labelled neighbors
    # each node has
    labelled_neighbors = {v: 0 for v in adj}

    # working backwards
    for i in range(num_nodes):
        # pick the node with the most labelled neighbors
        v = max(labelled_neighbors, key=lambda u: labelled_neighbors[u] + random())
        del labelled_neighbors[v]

        # increment all of its neighbors
        for u in adj[v]:
            if u in labelled_neighbors:
                labelled_neighbors[u] += 1

        order[-(i + 1)] = v

    for v in order:
        # if the number of neighbours of v is higher than upper_bound, update
        dv = len(adj[v])
        if dv > upper_bound:
            upper_bound = dv

        # make v simplicial by making its neighborhood a clique then remove the node
        # add v to order
        _elim_adj(adj, v)

    return upper_bound, order


def _elim_adj(adj, n):
    """eliminates a variable, acting on the adj matrix of G,
    returning set of edges that were added.

    Parameters
    ----------
    adj: dict
        A dict of the form {v: neighbors, ...} where v are
        vertices in a graph and neighbors is a set.

    Returns
    ----------
    new_edges: set of edges that were added by eliminating v.

    """
    neighbors = adj[n]
    new_edges = set()
    for u, v in itertools.combinations(neighbors, 2):
        if v not in adj[u]:
            adj[u].add(v)
            adj[v].add(u)
            new_edges.add((u, v))
            new_edges.add((v, u))
    for v in neighbors:
        adj[v].discard(n)
    del adj[n]
    return new_edges


def elimination_order_width(G, order):
    """Calculates the width of the tree decomposition induced by a
    variable elimination order.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to compute the width of the tree decomposition.

    order : list
        The elimination order. Must be a list of all of the variables
        in G.

    Returns
    -------
    treewidth : int
        The width of the tree decomposition induced by  order.

    Examples
    --------
    This example computes the width of the tree decomposition for the :math:`K_4`
    complete graph induced by an elimination order found through the min-width
    heuristic.

    >>> K_4 = nx.complete_graph(4)
    >>> tw, order = dwave.graphs.min_width_heuristic(K_4)
    >>> print(tw)
    3
    >>> dwave.graphs.elimination_order_width(K_4, order)
    3


    """
    # we need only deal with the adjacency structure of G. We will also
    # be manipulating it directly so let's go ahead and make a new one
    adj = {v: set(u for u in G[v] if u != v) for v in G}

    treewidth = 0

    for v in order:

        # get the degree of the eliminated variable
        try:
            dv = len(adj[v])
        except KeyError:
            raise ValueError('{} is in order but not in G'.format(v))

        # the treewidth is the max of the current treewidth and the degree
        if dv > treewidth:
            treewidth = dv

        # eliminate v by making it simplicial (acts on adj in place)
        _elim_adj(adj, v)

    # if adj is not empty, then order did not include all of the nodes in G.
    if adj:
        raise ValueError('not all nodes in G were in order')

    return treewidth


def treewidth_branch_and_bound(G, elimination_order=None, treewidth_upperbound=None):
    """Computes the treewidth of graph G and a corresponding perfect elimination ordering.

    Algorithm based on [Gog2004]_.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to compute the treewidth and perfect elimination ordering.

    elimination_order: list (optional, Default None)
        An elimination order used as an initial best-known order. If a good
        order is provided, it may speed up computation. If not provided, the
        initial order is generated using the min-fill heuristic.

    treewidth_upperbound : int (optional, Default None)
        An upper bound on the treewidth. Note that using
        this parameter can result in no returned order.

    Returns
    -------
    treewidth : int
        The treewidth of graph G.
    order : list
        An elimination order that induces the treewidth.

    Examples
    --------
    This example computes the treewidth for the :math:`K_7`
    complete graph using an optionally provided elimination order (a sequential
    ordering of the nodes, arbitrally chosen).

    >>> K_7 = nx.complete_graph(7)
    >>> dwave.graphs.treewidth_branch_and_bound(K_7, [0, 1, 2, 3, 4, 5, 6])
    (6, [0, 1, 2, 3, 4, 5, 6])

    References
    ----------
    Based on the algorithm presented in [Gog2004]_.
    """
    # empty graphs have treewidth 0 and the nodes can be eliminated in
    # any order
    if not any(G[v] for v in G):
        return 0, list(G)

    # variable names are chosen to match the paper

    # our order will be stored in vector x, named to be consistent with
    # the paper
    x = []  # the partial order

    f = minor_min_width(G)  # our current lower bound guess, f(s) in the paper
    g = 0  # g(s) in the paper

    # we need the best current update we can find.
    ub, order = min_fill_heuristic(G)

    # if the user has provided an upperbound or an elimination order, check those against
    # our current best guess
    if elimination_order is not None:
        upperbound = elimination_order_width(G, elimination_order)
        if upperbound <= ub:
            ub, order = upperbound, elimination_order

    if treewidth_upperbound is not None and treewidth_upperbound < ub:
        # in this case the order might never be found
        ub, order = treewidth_upperbound, []

    # best found encodes the ub and the order
    best_found = ub, order

    # if our upper bound is the same as f, then we are done! Otherwise begin the
    # algorithm.
    if f < ub:
        # we need only deal with the adjacency structure of G. We will also
        # be manipulating it directly so let's go ahead and make a new one
        adj = {v: set(u for u in G[v] if u != v) for v in G}

        best_found = _branch_and_bound(adj, x, g, f, best_found)
    elif f > ub and treewidth_upperbound is None:
        raise RuntimeError("logic error")

    return best_found


def _branch_and_bound(adj, x, g, f, best_found, skipable=set(), theorem6p2=None):
    """ Recursive branch and bound for computing treewidth of a subgraph.
    adj: adjacency list
    x: partial elimination order
    g: width of x so far
    f: lower bound on width of any elimination order starting with x
    best_found = ub,order: best upper bound on the treewidth found so far, and its elimination order
    skipable: vertices that can be skipped according to Lemma 5.3
    theorem6p2: terms that have been explored/can be pruned according to Theorem 6.2
    """

    # theorem6p2 checks for branches that can be pruned using Theorem 6.2
    if theorem6p2 is None:
        theorem6p2 = _theorem6p2()
    prune6p2, explored6p2, finished6p2 = theorem6p2
    # current6p2 is the list of prunable terms created during this instantiation of _branch_and_bound.
    # These terms will only be use during this call and its successors,
    # so they are removed before the function terminates.
    current6p2 = list()

    # theorem6p4 checks for branches that can be pruned using Theorem 6.4.
    # These terms do not need to be passed to successive calls to _branch_and_bound,
    # so they are simply created and deleted during this call.
    prune6p4, explored6p4 = _theorem6p4()

    # Note: theorem6p1 and theorem6p3 are a pruning strategies that are currently disabled
    # # as they does not appear to be invoked regularly,
    # and invoking it can require large memory allocations.
    # This can be fixed in the future if there is evidence that it's useful.
    # To add them in, define _branch_and_bound as follows:
    # def _branch_and_bound(adj, x, g, f, best_found, skipable=set(), theorem6p1=None,
    #                       theorem6p2=None, theorem6p3=None):

    # if theorem6p1 is None:
    #     theorem6p1 = _theorem6p1()
    # prune6p1, explored6p1 = theorem6p1

    # if theorem6p3 is None:
    #     theorem6p3 = _theorem6p3()
    # prune6p3, explored6p3 = theorem6p3

    # we'll need to know our current upper bound in several places
    ub, order = best_found

    # ok, take care of the base case first
    if len(adj) < 2:
        # check if our current branch is better than the best we've already
        # found and if so update our best solution accordingly.
        if f < ub:
            return (f, x + list(adj))
        elif f == ub and not order:
            return (f, x + list(adj))
        else:
            return best_found

    # so we have not yet reached the base case
    # Note: theorem 6.4 gives a heuristic for choosing order of n in adj.
    # Quick_bb suggests using a min-fill or random order.
    # We don't need to consider the neighbors of the last vertex eliminated
    sorted_adj = sorted((n for n in adj if n not in skipable), key=lambda x: _min_fill_needed_edges(adj, x))
    for n in sorted_adj:

        g_s = max(g, len(adj[n]))

        # according to Lemma 5.3, we can skip all of the neighbors of the last
        # variable eliniated when choosing the next variable
        # this does not get altered so we don't need a copy
        next_skipable = adj[n]

        if prune6p2(x, n, next_skipable):
            continue

        # update the state by eliminating n and adding it to the partial ordering
        adj_s = {v: adj[v].copy() for v in adj}  # create a new object
        edges_n = _elim_adj(adj_s, n)
        x_s = x + [n]  # new partial ordering

        # pruning (disabled):
        # if prune6p1(x_s):
        #     continue

        if prune6p4(edges_n):
            continue

        # By Theorem 5.4, if any two vertices have ub + 1 common neighbors then
        # we can add an edge between them
        _theorem5p4(adj_s, ub)

        # ok, let's update our values
        f_s = max(g_s, minor_min_width(adj_s))

        g_s, f_s, as_list = _graph_reduction(adj_s, x_s, g_s, f_s)

        # pruning (disabled):
        # if prune6p3(x, as_list, n):
        #     continue

        if f_s < ub:
            best_found = _branch_and_bound(adj_s, x_s, g_s, f_s, best_found,
                                           next_skipable, theorem6p2=theorem6p2)
            # if theorem6p1, theorem6p3 are enabled, this should be called as:
            # best_found = _branch_and_bound(adj_s, x_s, g_s, f_s, best_found,
            #                                next_skipable, theorem6p1=theorem6p1,
            #                                theorem6p2=theorem6p2,theorem6p3=theorem6p3)
            ub, __ = best_found

        # store some information for pruning (disabled):
        # explored6p3(x, n, as_list)

        prunable = explored6p2(x, n, next_skipable)
        current6p2.append(prunable)

        explored6p4(edges_n)

    # store some information for pruning (disabled):
    # explored6p1(x)

    for prunable in current6p2:
        finished6p2(prunable)

    return best_found


def _graph_reduction(adj, x, g, f):
    """we can go ahead and remove any simplicial or almost-simplicial vertices from adj.
    """
    as_list = set()
    as_nodes = {v for v in adj if len(adj[v]) <= f and is_almost_simplicial(adj, v)}
    while as_nodes:
        as_list.union(as_nodes)
        for n in as_nodes:

            # update g and f
            dv = len(adj[n])
            if dv > g:
                g = dv
            if g > f:
                f = g

            # eliminate v
            x.append(n)
            _elim_adj(adj, n)

        # see if we have any more simplicial nodes
        as_nodes = {v for v in adj if len(adj[v]) <= f and is_almost_simplicial(adj, v)}

    return g, f, as_list


def _theorem5p4(adj, ub):
    """By Theorem 5.4, if any two vertices have ub + 1 common neighbors
    then we can add an edge between them.
    """
    new_edges = set()
    for u, v in itertools.combinations(adj, 2):
        if u in adj[v]:
            # already an edge
            continue

        if len(adj[u].intersection(adj[v])) > ub:
            new_edges.add((u, v))

    while new_edges:
        for u, v in new_edges:
            adj[u].add(v)
            adj[v].add(u)

        new_edges = set()
        for u, v in itertools.combinations(adj, 2):
            if u in adj[v]:
                continue

            if len(adj[u].intersection(adj[v])) > ub:
                new_edges.add((u, v))


def _theorem6p1():
    """See Theorem 6.1 in paper."""

    pruning_set = set()

    def _prune(x):
        if len(x) <= 2:
            return False
        # this is faster than tuple(x[-3:])
        key = (tuple(x[:-2]), x[-2], x[-1])
        return key in pruning_set

    def _explored(x):
        if len(x) >= 3:
            prunable = (tuple(x[:-2]), x[-1], x[-2])
            pruning_set.add(prunable)

    return _prune, _explored


def _theorem6p2():
    """See Theorem 6.2 in paper.
    Prunes (x,...,a) when (x,a) is explored and a has the same neighbour set in both graphs.
    """
    pruning_set2 = set()

    def _prune2(x, a, nbrs_a):
        frozen_nbrs_a = frozenset(nbrs_a)
        for i in range(len(x)):
            key = (tuple(x[0:i]), a, frozen_nbrs_a)
            if key in pruning_set2:
                return True
        return False

    def _explored2(x, a, nbrs_a):
        prunable = (tuple(x), a, frozenset(nbrs_a))  # (s,a,N(a))
        pruning_set2.add(prunable)
        return prunable

    def _finished2(prunable):
        pruning_set2.remove(prunable)

    return _prune2, _explored2, _finished2


def _theorem6p3():
    """See Theorem 6.3 in paper.
    Prunes (s,b) when (s,a) is explored, b (almost) simplicial in (s,a), and a (almost) simplicial in (s,b)
    """
    pruning_set3 = set()

    def _prune3(x, as_list, b):
        for a in as_list:
            key = (tuple(x), a, b)  # (s,a,b) with (s,a) explored
            if key in pruning_set3:
                return True
        return False

    def _explored3(x, a, as_list):
        for b in as_list:
            prunable = (tuple(x), a, b)  # (s,a,b) with (s,a) explored
            pruning_set3.add(prunable)

    return _prune3, _explored3


def _theorem6p4():
    """See Theorem 6.4 in paper.
    Let E(x) denote the edges added when eliminating x. (edges_x below).
    Prunes (s,b) when (s,a) is explored and E(a) is a subset of E(b).
    For this theorem we only record E(a) rather than (s,E(a))
    because we only need to check for pruning in the same s context
    (i.e the same level of recursion).
    """
    pruning_set4 = list()

    def _prune4(edges_b):
        for edges_a in pruning_set4:
            if edges_a.issubset(edges_b):
                return True
        return False

    def _explored4(edges_a):
        pruning_set4.append(edges_a)  # (s,E_a) with (s,a) explored

    return _prune4, _explored4


def chimera_elimination_order(m, n=None, t=4, coordinates=False):
    """Provides a variable elimination order for a Chimera graph.

    A graph defined by ``chimera_graph(m,n,t)`` has treewidth :math:`max(m,n)*t`.
    This function outputs a variable elimination order inducing a tree
    decomposition of that width.

    Parameters
    ----------
    m : int
        Number of rows in the Chimera lattice.
    n : int (optional, default m)
        Number of columns in the Chimera lattice.
    t : int (optional, default 4)
        Size of the shore within each Chimera tile.
    coordinates bool (optional, default False):
        If True, the elimination order is given in terms of 4-term Chimera
        coordinates, otherwise given in linear indices.
        
    Returns
    -------
    order : list
        An elimination order that induces the treewidth of chimera_graph(m,n,t).

    Examples
    --------

    >>> G = dwave.graphs.chimera_elimination_order(1, 1, 4)  # a single Chimera tile

    """
    if n is None:
        n = m

    index_flip = m > n
    if index_flip:
        m, n = n, m

    def chimeraI(m0, n0, k0, l0):
        if index_flip:
            return m*2*t*n0 + 2*t*m0 + t*(1-k0) + l0
        else:
            return n*2*t*m0 + 2*t*n0 + t*k0 + l0

    order = []

    for n_i in range(n):
        for t_i in range(t):
            for m_i in range(m):
                order.append(chimeraI(m_i, n_i, 0, t_i))

    for n_i in range(n):
        for m_i in range(m):
            for t_i in range(t):
                order.append(chimeraI(m_i, n_i, 1, t_i))

    if coordinates:
        return list(chimera_coordinates(m,n,t).iter_linear_to_chimera(order))
    else:
        return order


def pegasus_elimination_order(n, coordinates=False):
    """Provides a variable elimination order for the Pegasus graph.

    The treewidth of a Pegasus graph ``pegasus_graph(n)`` is lower-bounded by
    :math:`12n-11` and upper bounded by :math:`12n-4` [Boo2019]_.

    Simple pegasus variable elimination order rules:

       - eliminate vertical qubits, one column at a time
       - eliminate horizontal qubits in each column once their adjacent vertical
         qubits have been eliminated

    Args
    ----
    n : int
        The size parameter for the Pegasus lattice.

    coordinates : bool, optional (default False)
        If True, the elimination order is given in terms of 4-term Pegasus
        coordinates, otherwise given in linear indices.

    Returns
    -------
    order : list
        An elimination order that provides an upper bound on the treewidth.

    """
    m = n
    l = 12

    # ordering for horizontal qubits in each tile, from east to west:
    h_order = [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11]
    order = []
    for n_i in range(n):  # for each tile offset
        # eliminate vertical qubits:
        for l_i in range(0, l, 2):
            for l_v in range(l_i, l_i + 2):
                for m_i in range(m - 1):  # for each column
                    order.append((0, n_i, l_v, m_i))
            # eliminate horizontal qubits:
            if n_i > 0 and not(l_i % 4):
                # a new set of horizontal qubits have had all their neighbouring vertical qubits eliminated.
                for m_i in range(m):
                    for l_h in range(h_order[l_i], h_order[l_i] + 4):
                        order.append((1, m_i, l_h, n_i - 1))

    if coordinates:
        return order
    else:
        return list(pegasus_coordinates(n).iter_pegasus_to_linear(order))

    
def zephyr_elimination_order(m, t=4, coordinates=False):
    """Provides a variable elimination order for the zephyr graph.

    The treewidth of a Zephyr graph ``zephyr_graph(m,t)`` is upper-bounded by
    :math:`4tm+2t` and lower-bounded by :math:`4tm` [Boo2021]_.

    Simple zephyr variable elimination rules:
       - eliminate vertical qubits, one column at a time
       - eliminate horizontal qubits in each column from top to bottom

    Args
    ----
    m : int
        Grid parameter for the Zephyr lattice.
    t : int
        Tile parameter for the Zephyr lattice.
    coordinates : bool, optional (default False)
        If True, the elimination order is given in terms of 4-term Zephyr
        coordinates, otherwise given in linear indices.

    Returns
    -------
    order : list
        An elimination order that achieves an upper bound on the treewidth.

    """
    order = ([(0,w,k,j,z) for w in range(2*m+1) for k in range(t) for z in range(m) for j in range(2)]
             + [(1,w,k,j,z) for z in range(m) for j in range(2)  for w in range(2*m+1) for k in range(t)])
    
    if coordinates:
        return order
    else:
        return list(zephyr_coordinates(m).iter_zephyr_to_linear(order))

