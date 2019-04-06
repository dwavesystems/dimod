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
# =============================================================================
from __future__ import absolute_import

import numpy as np
import numpy.random

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.decorators import graph_argument

__all__ = ['uniform', 'randint']


@graph_argument('graph')
def uniform(graph, vartype, low=0.0, high=1.0, cls=BinaryQuadraticModel,
            seed=None):
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

        seed (int, optional, default=None):
            Random seed.

    Returns:
        :obj:`.BinaryQuadraticModel`

    """
    if seed is None:
        seed = numpy.random.randint(2**32, dtype=np.uint32)
    r = numpy.random.RandomState(seed)

    variables, edges = graph

    index = {v: idx for idx, v in enumerate(variables)}

    if edges:
        irow, icol = zip(*((index[u], index[v]) for u, v in edges))
    else:
        irow = icol = tuple()

    ldata = r.uniform(low, high, size=len(variables))
    qdata = r.uniform(low, high, size=len(irow))
    offset = r.uniform(low, high)

    return cls.from_numpy_vectors(ldata, (irow, icol, qdata), offset, vartype,
                                  variable_order=variables)


@graph_argument('graph')
def randint(graph, vartype, low=0, high=1, cls=BinaryQuadraticModel,
            seed=None):
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

        seed (int, optional, default=None):
            Random seed.

    Returns:
        :obj:`.BinaryQuadraticModel`

    """
    if seed is None:
        seed = numpy.random.randint(2**32, dtype=np.uint32)
    r = numpy.random.RandomState(seed)

    variables, edges = graph

    index = {v: idx for idx, v in enumerate(variables)}

    if edges:
        irow, icol = zip(*((index[u], index[v]) for u, v in edges))
    else:
        irow = icol = tuple()

    # high+1 for inclusive range
    ldata = r.randint(low, high+1, size=len(variables))
    qdata = r.randint(low, high+1, size=len(irow))
    offset = r.randint(low, high+1)

    return cls.from_numpy_vectors(ldata, (irow, icol, qdata), offset, vartype,
                                  variable_order=variables)
