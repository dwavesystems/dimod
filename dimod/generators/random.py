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
#
# ================================================================================================
from __future__ import absolute_import

import random

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.decorators import graph_argument

__all__ = ['uniform', 'randint']


@graph_argument('graph')
def uniform(graph, vartype, low=0.0, high=1.0, cls=BinaryQuadraticModel):
    """Generate a bqm with random biases and offset.

    Biases and offset are drawn from a uniform distribution range (low, high).

    Args:
        graph (int/tuple[nodes, edges]/:obj:`~networkx.Graph`):
            The graph to build the bqm loops on. Either an integer n, interpreted as a
            complete graph of size n, or a nodes/edges pair, or a NetworkX graph.

        vartype (:class:`.Vartype`/str/set):
            Variable type for the binary quadratic model. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        low (float, optional, default=0.0):
            The low end of the range for the random biases.

        high (float, optional, default=1.0):
            The high end of the range for the random biases.

        cls (:class:`.BinaryQuadraticModel`):
            Binary quadratic model class to build from.

    """
    nodes, edges = graph
    return cls(((n, random.uniform(low, high)) for n in nodes),
               ((u, v, random.uniform(low, high)) for u, v in edges),
               random.uniform(low, high),
               vartype)


@graph_argument('graph')
def randint(graph, vartype, low=0, high=1, cls=BinaryQuadraticModel):
    """Generate a bqm with random biases and offset.

    Biases and offset are integer-valued in range [low, high] inclusive.

    Args:
        graph (int/tuple[nodes, edges]/:obj:`~networkx.Graph`):
            The graph to build the bqm loops on. Either an integer n, interpreted as a
            complete graph of size n, or a nodes/edges pair, or a NetworkX graph.

        vartype (:class:`.Vartype`/str/set):
            Variable type for the binary quadratic model. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        low (float, optional, default=0):
            The low end of the range for the random biases.

        high (float, optional, default=1):
            The high end of the range for the random biases.

        cls (:class:`.BinaryQuadraticModel`):
            Binary quadratic model class to build from.

    """
    nodes, edges = graph
    return cls(((n, random.randint(low, high)) for n in nodes),
               ((u, v, random.randint(low, high)) for u, v in edges),
               0.0,
               vartype)
