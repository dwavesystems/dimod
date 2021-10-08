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

from typing import Iterable, Optional, Tuple

from dimod.binary.binary_quadratic_model import BinaryQuadraticModel
from dimod.typing import Variable
from dimod.vartypes import Vartype

__all__ = [
    'independent_set',
    'maximum_independent_set',
    'maximum_weight_independent_set',
    ]


def independent_set(edges: Iterable[Tuple[Variable, Variable]],
                    nodes: Optional[Iterable[Variable]] = None,
                    ) -> BinaryQuadraticModel:
    """Return a binary quadratic model encoding an independent set problem.

    Given a graph `G`, an independent set is a set of nodes such that the
    subgraph of `G` induced by these nodes contains no edges.

    Args:
        edges: The edges of the graph as an iterable of two-tuples.
        nodes: The nodes of the graph as an iterable.

    Returns:
        A binary quadratic model. The binary quadratic model will have
        variables and interactions corresponding to ``nodes`` and ``edges``.
        Each interaction will have a quadratic bias of exactly ``1`` and
        each node will have a linear bias of ``0``.

    Examples:

        >>> from dimod.generators import independent_set

        Get an independent set binary quadratic model from a list of edges.

        >>> independent_set([(0, 1), (1, 2)])
        BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0}, {(1, 0): 1.0, (2, 1): 1.0}, 0.0, 'BINARY')

        Get an independent set binary quadratic model from a list of edges and
        nodes.

        >>> independent_set([(0, 1)], [0, 1, 2])
        BinaryQuadraticModel({0: 0.0, 1: 0.0, 2: 0.0}, {(1, 0): 1.0}, 0.0, 'BINARY')

        Get an independent set binary quadratic model from a
        :class:`networkx.Graph`.

        >>> import networkx as nx
        >>> G = nx.complete_graph(2)
        >>> independent_set(G.edges, G.nodes)
        BinaryQuadraticModel({0: 0.0, 1: 0.0}, {(1, 0): 1.0}, 0.0, 'BINARY')

    """
    bqm = BinaryQuadraticModel(vartype=Vartype.BINARY)

    bqm.add_quadratic_from((u, v, 1) for u, v in edges)

    if nodes is not None:
        bqm.add_linear_from((v, 0) for v in nodes)

    return bqm


def maximum_independent_set(edges: Iterable[Tuple[Variable, Variable]],
                            nodes: Optional[Iterable[Variable]] = None,
                            *,
                            strength: float = 2,
                            ) -> BinaryQuadraticModel:
    """Return a binary quadratic model encoding a maximum independent set problem.

    Given a graph `G`, an independent set is a set of nodes such that the
    subgraph of `G` induced by these nodes contains no edges.
    A maximum independent set is the independent set with the most nodes.


    Args:
        edges: The edges of the graph as an iterable of two-tuples.
        nodes: The nodes of the graph as an iterable.
        strength: The strength of the quadratic biases. Must be strictly
            greater than ``1`` in order to enforce the independent set
            constraint.

    Returns:
        A binary quadratic model. The binary quadratic model will have
        variables and interactions corresponding to ``nodes`` and ``edges``.

    Examples:

        >>> from dimod.generators import maximum_independent_set

        Get a maximum independent set binary quadratic model froma  list of
        edges.

        >>> maximum_independent_set([(0, 1), (1, 2)])
        BinaryQuadraticModel({0: -1.0, 1: -1.0, 2: -1.0}, {(1, 0): 2.0, (2, 1): 2.0}, 0.0, 'BINARY')

        Get a maximum independent set binary quadratic model from a list of
        edges and nodes.

        >>> maximum_independent_set([(0, 1)], [0, 1, 2])
        BinaryQuadraticModel({0: -1.0, 1: -1.0, 2: -1.0}, {(1, 0): 2.0}, 0.0, 'BINARY')

        Get a maximum independent set binary quadratic model from a
        :class:`networkx.Graph`.

        >>> import networkx as nx
        >>> G = nx.complete_graph(2)
        >>> maximum_independent_set(G.edges, G.nodes)
        BinaryQuadraticModel({0: -1.0, 1: -1.0}, {(1, 0): 2.0}, 0.0, 'BINARY')

    """
    return maximum_weight_independent_set(
        edges, None if nodes is None else ((v, 1) for v in nodes), strength=strength)


def maximum_weight_independent_set(edges: Iterable[Tuple[Variable, Variable]],
                                   nodes: Optional[Iterable[Tuple[Variable, float]]] = None,
                                   *,
                                   strength: Optional[float] = None,
                                   strength_multiplier: float = 2,
                                   ) -> BinaryQuadraticModel:
    """Return a binary quadratic model encoding a maximum-weight independent set problem.

    Given a graph `G`, an independent set is a set of nodes such that the
    subgraph of `G` induced by these nodes contains no edges.
    A maximum-weight independent set is the independent set with the highest
    total node weight.


    Args:
        edges: The edges of the graph as an iterable of two-tuples.
        nodes: The nodes of the graph as an iterable of two-tuples where the
            first element of the tuple is the node label and the second element
            is the node weight. Nodes not specified are given a weight of ``1``.
        strength: The strength of the quadratic biases. Must be strictly
            greater than ``1`` in order to enforce the independent set
            constraint. If not given, the strength is determined by the
            ``strength_multiplier``.
        strength_multiplier: The strength of the quadratic biases is given by
            the maximum node weight multiplied by ``strength_multiplier``.

    Returns:
        A binary quadratic model. The binary quadratic model will have
        variables and interactions corresponding to ``nodes`` and ``edges``.

    Examples:

        >>> from dimod.generators import maximum_weight_independent_set

        Get a maximum-weight independent set binary quadratic model from a list
        of edges and nodes.

        >>> maximum_weight_independent_set([(0, 1)], [(0, .25), (1, .5), (2, 1)])
        BinaryQuadraticModel({0: -0.25, 1: -0.5, 2: -1.0}, {(1, 0): 2.0}, 0.0, 'BINARY')

        Get a maximum-weight independent set binary quadratic model from a
        :class:`networkx.Graph`.

        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> G.add_edges_from([(0, 1), (1, 2)])
        >>> G.add_nodes_from([0, 2], weight=.25)
        >>> G.add_node(1, weight=.5)
        >>> maximum_weight_independent_set(G.edges, G.nodes('weight'))
        BinaryQuadraticModel({0: -0.25, 1: -0.5, 2: -0.25}, {(1, 0): 1.0, (2, 1): 1.0}, 0.0, 'BINARY')

    """
    bqm = independent_set(edges)

    objective = BinaryQuadraticModel(vartype=Vartype.BINARY)
    objective.add_linear_from((v, 1) for v in bqm.variables)
    if nodes is None:
        max_weight = 1.
    else:
        for v, weight in nodes:
            objective.set_linear(v, weight)
        max_weight = objective.linear.max(default=1)

    if strength is None:
        bqm *= max_weight*strength_multiplier
        bqm -= objective
    else:
        bqm *= strength
        bqm -= objective

    bqm.offset = 0  # otherwise subtracting the objective gives -0 offset

    return bqm
