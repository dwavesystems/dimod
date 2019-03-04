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

import random

from random import choice

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.decorators import graph_argument
from dimod.vartypes import SPIN

__all__ = ['chimera_anticluster']


@graph_argument('subgraph', allow_None=True)
def chimera_anticluster(m, n=None, t=4, multiplier=3.0,
                        cls=BinaryQuadraticModel, subgraph=None, seed=None):
    """Generate an anticluster problem on a Chimera lattice.

    An anticluster problem has weak interactions within a tile and strong
    interactions between tiles.

    Args:
        m (int):
            Number of rows in the Chimera lattice.

        n (int, optional, default=m):
            Number of columns in the Chimera lattice.

        t (int, optional, default=t):
            Size of the shore within each Chimera tile.

        multiplier (number, optional, default=3.0):
            Strength of the intertile edges.

        cls (class, optional, default=:class:`.BinaryQuadraticModel`):
            Binary quadratic model class to build from.

        subgraph (int/tuple[nodes, edges]/:obj:`~networkx.Graph`):
            A subgraph of a Chimera(m, n, t) graph to build the anticluster
            problem on.

        seed (int, optional, default=None):
            Random seed.

    Returns:
        :obj:`.BinaryQuadraticModel`: spin-valued binary quadratic model.

    """
    if seed is not None:
        random.seed(seed)

    m = int(m)
    if n is None:
        n = m
    else:
        n = int(n)
    t = int(t)

    # only defined for Ising problems
    bqm = cls.empty(SPIN)

    bqm.add_interactions_from((u, v, choice((-1., 1.)))
                              for u, v in _iter_chimera_tile_edges(m, n, t))
    bqm.add_interactions_from((u, v, choice((-multiplier, multiplier)))
                              for u, v in _iter_chimera_intertile_edges(m, n, t))

    if subgraph is not None:
        nodes, edges = subgraph

        subbqm = cls.empty(SPIN)

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
