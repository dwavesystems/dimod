# Copyright 2021 D-Wave Systems Inc.
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
import math

import dimod

__all__ = ["partition",
           "graph_partition_cqm",
           ]


def partition(G, num_partitions=2, sampler=None, **sampler_args):
    """Returns an approximate k-partition of G.
    
    Defines an CQM with ground states corresponding to a
    balanced k-partition of G and uses the sampler to sample from it.
    A k-partition is a collection of k subsets of the vertices
    of G such that each vertex is in exactly one subset, and
    the number of edges between vertices in different subsets
    is as small as possible. If G is a weighted graph, the sum
    of weights over those edges are minimized.
    
    Parameters
    ----------
    G : NetworkX graph
        The graph to partition.
    num_partitions : int, optional (default 2)
        The number of subsets in the desired partition.
    sampler : 
        A constrained quadratic model sampler. A sampler is a process that
        samples from low energy states in models defined by an Ising
        equation or a Quadratic Model, with or without constraints. The sampler 
        is expected to have a 'sample_cqm' method. A sampler is expected to
        return an iterable of samples, in order of increasing energy.
    sampler_args
        Additional keyword parameters are passed to the sampler.
    
    Returns
    -------
    node_partition : dict
        The partition as a dictionary mapping each node to subsets labelled
        as integers 0, 1, 2, ... num_partitions.
    
    Example
    -------
    This example uses a sampler from
    `dimod <https://github.com/dwavesystems/dimod>`_ to find a 2-partition
    for a graph of a Chimera unit cell created using the `chimera_graph()`
    function.
    
    >>> import dimod
    >>> sampler = dimod.ExactCQMSolver()
    >>> G = dnx.chimera_graph(1, 1, 4)
    >>> partitions = dnx.partition(G, sampler=sampler)
    
    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.
    """
    if not len(G.nodes):
        return {}
        
    cqm = graph_partition_cqm(G, num_partitions)
    
    # Solve the problem using the CQM solver
    response = sampler.sample_cqm(cqm, **sampler_args)

    # Consider only results satisfying all constraints
    possible_partitions = response.filter(lambda d: d.is_feasible)
    
    if not possible_partitions: 
        raise RuntimeError("No feasible solution could be found for this problem instance.")

    # Reinterpret result as partition assignment over nodes
    indicators = (key for key, value in possible_partitions.first.sample.items() if math.isclose(value, 1.))
    node_partition = {key[0]: key[1] for key in indicators}
    
    return node_partition
    
    
def graph_partition_cqm(G, num_partitions):
    """Find a constrained quadratic model for the graph's partitions.

    Defines an CQM with ground states corresponding to a
    balanced k-partition of G and uses the sampler to sample from it.
    A k-partition is a collection of k subsets of the vertices
    of G such that each vertex is in exactly one subset, and
    the number of edges between vertices in different subsets
    is as small as possible. If G is a weighted graph, the sum
    of weights over those edges are minimized.
    
    Parameters
    ----------
    G : NetworkX graph
        The graph to partition.
    num_partitions : int
        The number of subsets in the desired partition.

    Returns
    -------
    cqm : :class:`dimod.ConstrainedQuadraticModel`
        A constrained quadratic model with ground states corresponding to a
        partition problem. The nodes of `G` are discrete logical variables
        of the CQM, where the cases are the different partitions the node
        can be assigned to. The objective is given as the number of edges
        connecting nodes in different partitions.

    """
    partition_size = G.number_of_nodes()/num_partitions
    partitions = range(num_partitions)
    cqm = dimod.ConstrainedQuadraticModel()

    # Variables will be added using the discrete method in CQM
    x = {vk: dimod.Binary(vk) for vk in itertools.product(G.nodes, partitions)}
    
    for v in G.nodes:
        cqm.add_discrete(((v, k) for k in partitions), label=v)
        
        
    if not math.isclose(partition_size, int(partition_size)):
        # if number of nodes don't divide into num_partitions,
        # accept partitions of size ceil() or floor()
        floor, ceil = int(partition_size), int(partition_size+1)
        for k in partitions:
            cqm.add_constraint(dimod.quicksum((x[u, k] for u in G.nodes)) >= floor, label='equal_partition_low_%s' %k)
            cqm.add_constraint(dimod.quicksum((x[u, k] for u in G.nodes)) <= ceil, label='equal_partition_high_%s' %k)
    else:
        # each partition must have partition_size elements
        for k in partitions:
            cqm.add_constraint(dimod.quicksum((x[u, k] for u in G.nodes)) == int(partition_size), label='equal_partition_%s' %k)

    cuts = 0
    for (u, v, d) in G.edges(data=True):
        for k in partitions:
            w = d.get('weight',1)
            cuts += w * x[u,k] * x[v,k]
            
    if cuts:
        cqm.set_objective(-cuts)

    return cqm