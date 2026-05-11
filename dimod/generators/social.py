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

__all__ = ["structural_imbalance_ising",
           ]


def structural_imbalance_ising(S):
    """Construct the Ising problem to calculate the structural imbalance of a signed social network.

    A signed social network graph is a graph whose signed edges
    represent friendly/hostile interactions between nodes. A
    signed social network is considered balanced if it can be cleanly
    divided into two factions, where all relations within a faction are
    friendly, and all relations between factions are hostile. The measure
    of imbalance or frustration is the minimum number of edges that
    violate this rule.

    Parameters
    ----------
    S : NetworkX graph
        A social graph on which each edge has a 'sign' attribute with a numeric value.

    Returns
    -------
    h : dict
        The linear biases of the Ising problem. Each variable in the Ising problem represent
        a node in the signed social network. The solution that minimized the Ising problem
        will assign each variable a value, either -1 or 1. This bi-coloring defines the factions.

    J : dict
        The quadratic biases of the Ising problem.

    Raises
    ------
    ValueError
        If any edge does not have a 'sign' attribute.

    Examples
    --------
    >>> import dimod
    >>> from dwave_networkx.algorithms.social import structural_imbalance_ising
    ...
    >>> S = nx.Graph()
    >>> S.add_edge('Alice', 'Bob', sign=1)  # Alice and Bob are friendly
    >>> S.add_edge('Alice', 'Eve', sign=-1)  # Alice and Eve are hostile
    >>> S.add_edge('Bob', 'Eve', sign=-1)  # Bob and Eve are hostile
    ...
    >>> h, J = structural_imbalance_ising(S)
    >>> h  # doctest: +SKIP
    {'Alice': 0.0, 'Bob': 0.0, 'Eve': 0.0}
    >>> J  # doctest: +SKIP
    {('Alice', 'Bob'): -1.0, ('Alice', 'Eve'): 1.0, ('Bob', 'Eve'): 1.0}

    """
    h = {v: 0.0 for v in S}
    J = {}
    for u, v, data in S.edges(data=True):
        try:
            J[(u, v)] = -1. * data['sign']
        except KeyError:
            raise ValueError(("graph should be a signed social graph,"
                              "each edge should have a 'sign' attr"))

    return h, J
