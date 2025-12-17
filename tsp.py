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

from collections import defaultdict

from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = ["traveling_salesperson",
           "traveling_salesperson_qubo",
           "traveling_salesman",
           "traveling_salesman_qubo",
           "is_hamiltonian_path",
           ]


@binary_quadratic_model_sampler(1)
def traveling_salesperson(G, sampler=None, lagrange=None, weight='weight',
                          start=None, **sampler_args):
    """Returns an approximate minimum traveling salesperson route.

    Defines a QUBO with ground states corresponding to the
    minimum routes and uses the sampler to sample
    from it.

    A route is a cycle in the graph that reaches each node exactly once.
    A minimum route is a route with the smallest total edge weight.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a minimum traveling salesperson route.
        This should be a complete graph with non-zero weights on every edge.

    sampler :
        A binary quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Unconstrained Binary Optimization
        Problem (QUBO). A sampler is expected to have a 'sample_qubo'
        and 'sample_ising' method. A sampler is expected to return an
        iterable of samples, in order of increasing energy. If no
        sampler is provided, one must be provided using the
        `set_default_sampler` function.

    lagrange : number, optional (default None)
        Lagrange parameter to weight constraints (visit every city once)
        versus objective (shortest distance route).

    weight : optional (default 'weight')
        The name of the edge attribute containing the weight.

    start : node, optional
        If provided, the route will begin at `start`.

    sampler_args :
        Additional keyword parameters are passed to the sampler.

    Returns
    -------
    route : list
       List of nodes in order to be visited on a route

    Examples
    --------

    >>> import dimod
    ...
    >>> G = nx.Graph()
    >>> G.add_weighted_edges_from({(0, 1, .1), (0, 2, .5), (0, 3, .1), (1, 2, .1),
    ...                            (1, 3, .5), (2, 3, .1)})
    >>> dnx.traveling_salesperson(G, dimod.ExactSolver(), start=0) # doctest: +SKIP
    [0, 1, 2, 3]

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """
    # Get a QUBO representation of the problem
    Q = traveling_salesperson_qubo(G, lagrange, weight)

    # use the sampler to find low energy states
    response = sampler.sample_qubo(Q, **sampler_args)

    sample = response.first.sample

    route = [None]*len(G)
    for (city, time), val in sample.items():
        if val:
            route[time] = city

    if start is not None and route[0] != start:
        # rotate to put the start in front
        idx = route.index(start)
        route = route[idx:] + route[:idx]

    return route


traveling_salesman = traveling_salesperson


def traveling_salesperson_qubo(G, lagrange=None, weight='weight', missing_edge_weight=None):
    """Return the QUBO with ground states corresponding to a minimum TSP route.

    If :math:`|G|` is the number of nodes in the graph, the resulting qubo will have:

    * :math:`|G|^2` variables/nodes
    * :math:`2 |G|^2 (|G| - 1)` interactions/edges

    Parameters
    ----------
    G : NetworkX graph
        A complete graph in which each edge has a attribute giving its weight.

    lagrange : number, optional (default None)
        Lagrange parameter to weight constraints (no edges within set)
        versus objective (largest set possible).

    weight : optional (default 'weight')
        The name of the edge attribute containing the weight.
    
    missing_edge_weight : number, optional (default None)
        For bi-directional graphs, the weight given to missing edges.
        If None is given (the default), missing edges will be set to
        the sum of all weights.

    Returns
    -------
    QUBO : dict
       The QUBO with ground states corresponding to a minimum travelling
       salesperson route. The QUBO variables are labelled `(c, t)` where `c`
       is a node in `G` and `t` is the time index. For instance, if `('a', 0)`
       is 1 in the ground state, that means the node 'a' is visted first.

    """
    N = G.number_of_nodes()

    if lagrange is None:
        # If no lagrange parameter provided, set to 'average' tour length.
        # Usually a good estimate for a lagrange parameter is between 75-150%
        # of the objective function value, so we come up with an estimate for 
        # tour length and use that.
        if G.number_of_edges()>0:
            lagrange = G.size(weight=weight)*G.number_of_nodes()/G.number_of_edges()
        else:
            lagrange = 2
    
    # calculate default missing_edge_weight if required
    if missing_edge_weight is None:
        # networkx method to calculate sum of all weights
        missing_edge_weight = G.size(weight=weight)

    # some input checking
    if N in (1, 2):
        msg = "graph must have at least 3 nodes or be empty"
        raise ValueError(msg)

    # Creating the QUBO
    Q = defaultdict(float)

    # Constraint that each row has exactly one 1
    for node in G:
        for pos_1 in range(N):
            Q[((node, pos_1), (node, pos_1))] -= lagrange
            for pos_2 in range(pos_1+1, N):
                Q[((node, pos_1), (node, pos_2))] += 2.0*lagrange

    # Constraint that each col has exactly one 1
    for pos in range(N):
        for node_1 in G:
            Q[((node_1, pos), (node_1, pos))] -= lagrange
            for node_2 in set(G)-{node_1}:
                # QUBO coefficient is 2*lagrange, but we are placing this value 
                # above *and* below the diagonal, so we put half in each position.
                Q[((node_1, pos), (node_2, pos))] += lagrange

    # Objective that minimizes distance
    for u, v in itertools.combinations(G.nodes, 2):
        for pos in range(N):
            nextpos = (pos + 1) % N

            # going from u -> v
            try:
                value = G[u][v][weight]
            except KeyError:
                value = missing_edge_weight

            Q[((u, pos), (v, nextpos))] += value

            # going from v -> u
            try:
                value = G[v][u][weight]
            except KeyError:
                value = missing_edge_weight

            Q[((v, pos), (u, nextpos))] += value

    return Q


traveling_salesman_qubo = traveling_salesperson_qubo


def is_hamiltonian_path(G, route):
    """Determines whether the given list forms a valid TSP route.

    A travelling salesperson route must visit each city exactly once.

    Parameters
    ----------
    G : NetworkX graph

        The graph on which to check the route.

    route : list

        List of nodes in the order that they are visited.

    Returns
    -------
    is_valid : bool
        True if route forms a valid travelling salesperson route.

    """

    return (set(route) == set(G))
