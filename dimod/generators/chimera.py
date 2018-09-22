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
from dimod.vartypes import SPIN

__all__ = ['chimera_anticluster']


def chimera_anticluster(m, n=None, t=4, multiplier=3.0, cls=BinaryQuadraticModel):
    """Generate an anticluster problem on a Chimera lattice.

    An anticluster problem has weak interactions within a tile and strong interactions outside.

    Args:
        m (int):
            Number of rows in the Chimera lattice.

        n (int, optional, default=m):
            Number of columns in the Chimera lattice.

        t (int, optiona, default=t):
            Size of the shore within each Chimera tile.

        multiplier (number):
            Strength of the intertile edges.

        cls (:class:`.BinaryQuadraticModel`):
            Binary quadratic model class to build from.

    Returns:
        :obj:`.BinaryQuadraticModel`: spin-valued binary quadratic model.

    """
    m = int(m)
    if n is None:
        n = m
    else:
        n = int(n)
    t = int(t)

    # only defined for Ising problems
    linear = {}

    quadratic = {edge: random.choice((-1., 1.)) for edge in _iter_chimera_tile_edges(m, n, t)}
    quadratic.update({edge: multiplier*random.choice((-1., 1.)) for edge in _iter_chimera_intertile_edges(m, n, t)})

    return cls(linear, quadratic, 0.0, SPIN)


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
