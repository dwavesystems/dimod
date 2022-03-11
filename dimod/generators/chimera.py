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

import warnings

from typing import Optional, Union

import numpy as np
import numpy.random

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.decorators import graph_argument
from dimod.typing import GraphLike
from dimod.vartypes import SPIN

__all__ = ['chimera_anticluster']


@graph_argument('subgraph', allow_None=True)
def chimera_anticluster(m: int, n: Optional[int] = None, t: int = 4,
                        multiplier: float = 3,
                        cls: None = None,
                        subgraph: GraphLike = None,
                        seed: Union[None, int] = None,
                        ) -> BinaryQuadraticModel:
    """Generate an anticluster problem on a Chimera lattice.

    An anticluster problem has weak interactions within a tile and strong
    interactions between tiles.

    Args:
        m: Number of rows in the Chimera lattice.

        n: Number of columns in the Chimera lattice. Defaults to ``m``.

        t: Size of the shore within each Chimera tile.

        multiplier: Strength of the inter-tile edges.

        cls: Deprecated. Does nothing.

        subgraph: A subgraph of the specified Chimera(m, n, t) graph to build the
            anticluster problem on. Can be used to remove nodes and edges.

        seed: Random seed.

    Returns:
        A binary quadratic model.

    .. deprecated:: 0.10.6

        The ``cls`` keyword argument will be removed in 0.12.0.
        It currently does nothing.

    """

    if cls is not None:
        warnings.warn("cls keyword argument is deprecated after 0.10.6 and will "
                      "be removed in 0.11. Does nothing.", DeprecationWarning,
                      stacklevel=2)

    if seed is None:
        seed = numpy.random.randint(2**32, dtype=np.uint32)
    r = numpy.random.RandomState(seed)

    m = int(m)
    if n is None:
        n = m
    else:
        n = int(n)
    t = int(t)

    ldata = np.zeros(m*n*t*2)  # number of nodes

    if m and n and t:
        inrow, incol = zip(*_iter_chimera_tile_edges(m, n, t))

        if m > 1 or n > 1:
            outrow, outcol = zip(*_iter_chimera_intertile_edges(m, n, t))
        else:
            outrow = outcol = tuple()

        qdata = r.choice((-1., 1.), size=len(inrow)+len(outrow))

        qdata[len(inrow):] *= multiplier

        irow = inrow + outrow
        icol = incol + outcol

    else:
        irow = icol = qdata = tuple()

    bqm = BinaryQuadraticModel.from_numpy_vectors(ldata, (irow, icol, qdata), 0.0, SPIN)

    if subgraph is not None:
        nodes, edges = subgraph

        subbqm = BinaryQuadraticModel.empty(SPIN)

        try:
            subbqm.add_variables_from((v, bqm.linear[v]) for v in nodes)

        except KeyError:
            msg = "given 'subgraph' contains nodes not in Chimera({}, {}, {})".format(m, n, t)
            raise ValueError(msg)

        try:
            subbqm.add_interactions_from((u, v, bqm.adj[u][v]) for u, v in edges)
        except KeyError:
            msg = "given 'subgraph' contains edges not in Chimera({}, {}, {})".format(m, n, t)
            raise ValueError(msg)

        bqm = subbqm

    return bqm


def _iter_chimera_tile_edges(m, n, t):
    hoff = 2 * t
    voff = n * hoff
    mi = m * voff
    ni = n * hoff

    # tile edges
    for edge in ((k0, k1)
                 for i in range(0, ni, hoff)
                 for j in range(i, mi, voff)
                 for k0 in range(j, j + t)
                 for k1 in range(j + t, j + 2 * t)):
        yield edge


def _iter_chimera_intertile_edges(m, n, t):
    hoff = 2 * t
    voff = n * hoff
    mi = m * voff
    ni = n * hoff

    # horizontal edges
    for edge in ((k, k + hoff)
                 for i in range(t, 2 * t)
                 for j in range(i, ni - hoff, hoff)
                 for k in range(j, mi, voff)):
        yield edge

    # vertical edges
    for edge in ((k, k + voff)
                 for i in range(t)
                 for j in range(i, ni, hoff)
                 for k in range(j, mi - voff, voff)):
        yield edge
