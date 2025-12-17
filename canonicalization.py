# Copyright 2018 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

from dwave_networkx.generators.chimera import chimera_coordinates

__all__ = ['canonical_chimera_labeling']


def canonical_chimera_labeling(G, t=None):
    """Returns a mapping from the labels of G to chimera-indexed labeling.

    Parameters
    ----------
    G : NetworkX graph
        A Chimera-structured graph.
    t : int (optional, default 4)
        Size of the shore within each Chimera tile.

    Returns
    -------
    chimera_indices: dict
        A mapping from the current labels to a 4-tuple of Chimera indices.

    """
    adj = G.adj

    if t is None:
        if hasattr(G, 'edges'):
            num_edges = len(G.edges)
        else:
            num_edges = len(G.quadratic)
        t = _chimera_shore_size(adj, num_edges)

    chimera_indices = {}

    row = col = 0

    # need to find a node in a corner
    root_edge = min(((u, v) for u in adj for v in adj[u]),
                    key=lambda edge: len(adj[edge[0]]) + len(adj[edge[1]]))
    root, _ = root_edge

    horiz, verti = rooted_tile(adj, root, t)
    while len(chimera_indices) < len(adj):

        new_indices = {}

        if row == 0:
            # if we're in the 0th row, we can assign the horizontal randomly
            for si, v in enumerate(horiz):
                new_indices[v] = (row, col, 0, si)
        else:
            # we need to match the row above
            for v in horiz:
                north = [u for u in adj[v] if u in chimera_indices]
                assert len(north) == 1
                i, j, u, si = chimera_indices[north[0]]
                assert i == row - 1 and j == col and u == 0
                new_indices[v] = (row, col, 0, si)

        if col == 0:
            # if we're in the 0th col, we can assign the vertical randomly
            for si, v in enumerate(verti):
                new_indices[v] = (row, col, 1, si)
        else:
            # we need to match the column to the east
            for v in verti:
                east = [u for u in adj[v] if u in chimera_indices]
                assert len(east) == 1
                i, j, u, si = chimera_indices[east[0]]
                assert i == row and j == col - 1 and u == 1
                new_indices[v] = (row, col, 1, si)

        chimera_indices.update(new_indices)

        # get the next root
        root_neighbours = [v for v in adj[root] if v not in chimera_indices]
        if len(root_neighbours) == 1:
            # we can increment the row
            root = root_neighbours[0]
            horiz, verti = rooted_tile(adj, root, t)

            row += 1
        else:
            # need to go back to row 0, and increment the column
            assert not root_neighbours  # should be empty

            # we want (0, col, 1, 0), we could cache this, but for now let's just go look for it
            # the slow way
            vert_root = [v for v in chimera_indices if chimera_indices[v] == (0, col, 1, 0)][0]

            vert_root_neighbours = [v for v in adj[vert_root] if v not in chimera_indices]

            if vert_root_neighbours:

                verti, horiz = rooted_tile(adj, vert_root_neighbours[0], t)
                root = next(iter(horiz))

                row = 0
                col += 1

    return chimera_indices


def rooted_tile(adj, n, t):
    horiz = {n}
    vert = set()

    # get all of the nodes that are two steps away from n
    two_steps = {v for u in adj[n] for v in adj[u] if v != n}

    # find the subset of two_steps that share exactly t neighbours
    for v in two_steps:
        shared = set(adj[n]).intersection(adj[v])

        if len(shared) == t:
            assert v not in horiz
            horiz.add(v)
            vert |= shared

    assert len(vert) == t
    return horiz, vert


def _chimera_shore_size(adj, num_edges):
    # we know |E| = m*n*t*t + (2*m*n-m-n)*t

    num_nodes = len(adj)

    max_degree = max(len(adj[v]) for v in adj)

    if num_nodes == 2 * max_degree:
        return max_degree

    def a(t):
        return -2*t

    def b(t):
        return (t + 2) * num_nodes - 2 * num_edges

    def c(t):
        return -num_nodes

    t = max_degree - 1
    m = (-b(t) + math.sqrt(b(t)**2 - 4*a(t)*c(t))) / (2 * a(t))

    if m.is_integer():
        return t

    return max_degree - 2
