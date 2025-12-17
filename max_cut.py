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

from dwave_networkx.exceptions import DWaveNetworkXException
from dwave_networkx.utils import binary_quadratic_model_sampler

__all__ = ["maximum_cut", "weighted_maximum_cut"]


@binary_quadratic_model_sampler(1)
def maximum_cut(G, sampler=None, **sampler_args):
    """Returns an approximate maximum cut.

    Defines an Ising problem with ground states corresponding to
    a maximum cut and uses the sampler to sample from it.

    A maximum cut is a subset S of the vertices of G such that
    the number of edges between S and the complementary subset
    is as large as possible.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a maximum cut.

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
    S : set
        A maximum cut of G.

    Example
    -------
    This example uses a sampler from
    `dimod <https://github.com/dwavesystems/dimod>`_ to find a maximum cut
    for a graph of a Chimera unit cell created using the `chimera_graph()`
    function.

    >>> import dimod
    ...
    >>> sampler = dimod.SimulatedAnnealingSampler()
    >>> G = dnx.chimera_graph(1, 1, 4)
    >>> cut = dnx.maximum_cut(G, sampler)

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """
    # In order to form the Ising problem, we want to increase the
    # energy by 1 for each edge between two nodes of the same color.
    # The linear biases can all be 0.
    h = {v: 0. for v in G}
    J = {(u, v): 1 for u, v in G.edges}

    # draw the lowest energy sample from the sampler
    response = sampler.sample_ising(h, J, **sampler_args)
    sample = next(iter(response))

    return set(v for v in G if sample[v] >= 0)


@binary_quadratic_model_sampler(1)
def weighted_maximum_cut(G, sampler=None, **sampler_args):
    """Returns an approximate weighted maximum cut.

    Defines an Ising problem with ground states corresponding to
    a weighted maximum cut and uses the sampler to sample from it.

    A weighted maximum cut is a subset S of the vertices of G that
    maximizes the sum of the edge weights between S and its
    complementary subset.

    Parameters
    ----------
    G : NetworkX graph
        The graph on which to find a weighted maximum cut. Each edge in G should
        have a numeric `weight` attribute.

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
    S : set
        A maximum cut of G.

    Notes
    -----
    Samplers by their nature may not return the optimal solution. This
    function does not attempt to confirm the quality of the returned
    sample.

    """
    # In order to form the Ising problem, we want to increase the
    # energy by 1 for each edge between two nodes of the same color.
    # The linear biases can all be 0.
    h = {v: 0. for v in G}
    try:
        J = {(u, v): G[u][v]['weight'] for u, v in G.edges}
    except KeyError:
        raise DWaveNetworkXException("edges must have 'weight' attribute")

    # draw the lowest energy sample from the sampler
    response = sampler.sample_ising(h, J, **sampler_args)
    sample = next(iter(response))

    return set(v for v in G if sample[v] >= 0)
