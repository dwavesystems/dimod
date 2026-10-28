# Copyright 2026 D-Wave
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

import collections
import itertools
from typing import Any

import dimod
from dimod.decorators import graph_argument
from dimod.typing import GraphLike

__all__ = ['matching',
           'maximal_matching',
           'min_maximal_matching',
           ]


@graph_argument('graph')
def matching(graph: GraphLike) -> dimod.BinaryQuadraticModel:
    """Find a binary quadratic model for the graph's matchings.

    A matching is a subset of edges in which no node occurs more than
    once. This function returns a binary quadratic model (BQM) with ground
    states corresponding to the possible matchings of ``graph``.

    Finding valid matchings can be done in polynomial time, so finding matching
    with BQMs is generally inefficient.
    This BQM may be useful when combined with other constraints and objectives.

    Args:
        graph:
            The graph on which to find a matching. Either an integer
            ``n``, interpreted as a complete graph of size ``n``, a nodes/edges
            pair, a list of edges or a NetworkX graph.

    Returns:
        A binary quadratic model with ground states corresponding to a
        matching. The variables of the BQM are the edges of ``graph`` as frozensets.
        The BQM's ground state energy is 0 by construction.
        The energy of the first excited state is 1.

    """
    nodes, edges = graph
    hood = _neighborhood(edges)

    bqm = dimod.BinaryQuadraticModel.empty('BINARY')

    # add the edges of `graph` as variables
    for edge in edges:
        bqm.add_variable(frozenset(edge), 0)

    for node in nodes:
        for edge0, edge1 in itertools.combinations(hood[node], 2):
            u = frozenset(edge0)
            v = frozenset(edge1)
            bqm.add_interaction(u, v, 1)

    return bqm


def _neighborhood(edges: list[tuple[Any, Any]]) -> dict[Any, list[tuple[Any, Any]]]:
    # return neighborhood dict from the list of edges
    hood = collections.defaultdict(list)
    for edge in edges:
        for node in edge:
            hood[node].append(edge)
    return hood


@graph_argument('graph')
def maximal_matching(graph: GraphLike,
                     lagrange: float | None = None,
                     ) -> dimod.BinaryQuadraticModel:
    """Find a binary quadratic model for the graph's maximal matchings.

    A matching is a subset of edges in which no node occurs more than
    once. A maximal matching is one in which no edges from ``graph`` can be
    added without violating the matching rule.
    This function returns a binary quadratic model (BQM) with ground
    states corresponding to the possible maximal matchings of ``graph``.

    Finding maximal matchings can be done in polynomial time, so finding
    maximal matching with BQMs is generally inefficient.
    This BQM may be useful when combined with other constraints and objectives.

    Args:
        graph:
            The graph on which to find a maximal matching. Either an integer
            ``n``, interpreted as a complete graph of size ``n``, a nodes/edges
            pair, a list of edges or a NetworkX graph.

        lagrange:
            The Lagrange multiplier for the matching constraint. Should be
            positive and greater than ``max_degree - 2``.
            Defaults to :math:`1.25 * (max_degree - 2)`.

    Returns:
        A binary quadratic model with ground states corresponding to a maximal
        matching. The variables of the BQM are the edges of ``graph`` as
        frozensets. The BQM's ground state energy is 0 by construction.

    """
    nodes, edges = graph
    hood = _neighborhood(edges)

    bqm = matching(graph)

    if lagrange is None:
        max_degree = len(max(hood.values(), key=len, default=[]))
        lagrange = max(1.25 * (max_degree - 2), 1)

    bqm.scale(lagrange)

    for node0, node1 in edges:
        # (1 - y_v - y_u + y_v*y_u) <- see paper

        bqm.offset += 1

        for edge in hood[node0]:
            bqm.linear[frozenset(edge)] -= 1

        for edge in hood[node1]:
            bqm.linear[frozenset(edge)] -= 1

        for edge0 in hood[node0]:
            u = frozenset(edge0)
            for edge1 in hood[node1]:
                v = frozenset(edge1)
                if u == v:
                    bqm.linear[u] += 1
                else:
                    bqm.add_interaction(u, v, 1)

    return bqm


@graph_argument('graph')
def min_maximal_matching(graph: GraphLike,
                         maximal_lagrange: float = 2,
                         matching_lagrange: float | None = None
                         ) -> dimod.BinaryQuadraticModel:
    """Find a binary quadratic model for the graph's minimum maximal matchings.

    A matching is a subset of edges in which no node occurs more than
    once. A maximal matching is one in which no edges from ``graph`` can be
    added without violating the matching rule. A minimum maximal matching
    is a maximal matching that contains the smallest possible number of edges.
    This function returns a binary quadratic model (BQM) with ground
    states corresponding to the possible maximal matchings of ``graph``.

    Args:
        graph:
            The graph on which to find a minimum maximal matching. Either an
            integer ``n``, interpreted as a complete graph of size ``n``, a
            nodes/edges pair, a list of edges or a NetworkX graph.

        maximal_lagrange:
            The Lagrange multiplier for the maximal constraint. Should be
            greater than 1. Defaults to 2.

        matching_lagrange:
            The Lagrange multiplier for the matching constraint. Should be
            positive and greater than ``maximal_lagrange * max_degree - 2``.
            Defaults to ``1.25 * (maximal_lagrange * max_degree - 2)``.

    Returns:
        A binary quadratic model with ground states corresponding to a
        minimum maximal matching. The variables of the BQM are the edges
        of ``graph`` as frozensets.

    """

    if matching_lagrange is not None:
        # we're going to scale the bqm by maximal_matching so undo that
        # for maximal_lagrange
        matching_lagrange /= maximal_lagrange

    bqm = maximal_matching(graph, lagrange=matching_lagrange)
    bqm.scale(maximal_lagrange)

    for v in bqm.variables:
        bqm.linear[v] += 1

    return bqm
