# Copyright 2019 D-Wave Systems Inc.
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

"""
Functions that convert binary quadratic models to and from other formats.
"""

import warnings

from dimod.binary.binary_quadratic_model import BQM

__all__ = ['to_networkx_graph', 'from_networkx_graph']


def to_networkx_graph(bqm,
                      node_attribute_name='bias',
                      edge_attribute_name='bias'):
    """Convert a binary quadratic model to NetworkX graph format.

    Note that NetworkX must be installed for this method to work.

    Args:
        node_attribute_name (hashable, optional, default='bias'):
            Attribute name for linear biases.

        edge_attribute_name (hashable, optional, default='bias'):
            Attribute name for quadratic biases.

    Returns:
        :class:`networkx.Graph`: A NetworkX graph with biases stored as
        node/edge attributes.

    """
    import networkx as nx

    BQM = nx.Graph()

    # add the linear biases
    BQM.add_nodes_from(((v, {node_attribute_name: bias, 'vartype': bqm.vartype})
                        for v, bias in bqm.linear.items()))

    # add the quadratic biases
    BQM.add_edges_from(((u, v, {edge_attribute_name: bias})
                        for (u, v), bias in bqm.quadratic.items()))

    # set the offset and vartype properties for the graph
    BQM.offset = bqm.offset
    BQM.vartype = bqm.vartype

    return BQM


def from_networkx_graph(G, vartype=None,
                        node_attribute_name='bias', edge_attribute_name='bias',
                        cls=None,
                        dtype=object):
    """Create a binary quadratic model from a NetworkX graph.

    Args:
        G (:obj:`networkx.Graph`):
            A NetworkX graph with biases stored as node/edge attributes.

        vartype (:class:`.Vartype`/str/set, optional):
            Variable type for the binary quadratic model. Accepted input
            values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

            If not provided, the `G` should have a vartype attribute. If
            `vartype` is provided and `G.vartype` exists then the argument
            overrides the property.

        node_attribute_name (hashable, optional, default='bias'):
            Attribute name for linear biases. If the node does not have a
            matching attribute then the bias defaults to 0.

        edge_attribute_name (hashable, optional, default='bias'):
            Attribute name for quadratic biases. If the edge does not have a
            matching attribute then the bias defaults to 0.

        cls: Deprecated keyword argument. Does nothing.

        dtype: The dtype of the binary quadratic model. Defaults to `object`.

    Returns:
        A binary quadratic model of type `cls`.

    Examples:
        This example creates a BQM from an illustrative NetworkX graph.

        >>> import networkx as nx
        >>> import random
        ...
        >>> G = nx.generators.complete_graph(4)
        >>> for node in G.nodes:
        ...     G.nodes[node]['bias'] = random.choice([1,-1])
        >>> for edge in G.edges:
        ...     G.edges[edge]['quadratic'] = random.choice([1,-1])
        ...
        >>> bqm = dimod.from_networkx_graph(G,
        ...                                 vartype='BINARY',
        ...                                 edge_attribute_name='quadratic')

    .. deprecated:: 0.10.0

        The ``cls`` keyword will be removed in dimod 0.12.0. It currently does
        nothing.

    """
    if cls is not None:
        warnings.warn(
            "'cls' keyword argument is deprecated since dimod 0.10.0 and will be removed in 0.12.0. "
            "It does nothing.",
            DeprecationWarning, stacklevel=2)
    if vartype is None:
        if not hasattr(G, 'vartype'):
            msg = ("either 'vartype' argument must be provided or "
                   "the given graph should have a vartype attribute.")
            raise ValueError(msg)
        vartype = G.vartype

    linear = {v: b for v, b in G.nodes(data=node_attribute_name, default=0)}
    quadratic = {(u, v): b
                 for u, v, b in G.edges(data=edge_attribute_name, default=0)}
    offset = getattr(G, 'offset', 0)

    return BQM(linear, quadratic, offset, vartype, dtype=dtype)
