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

import dimod

__all__ = ['markov_network',
           ]


###############################################################################
# The following code is partially based on https://github.com/tbabej/gibbs
#
# MIT License
# ===========
#
# Copyright 2017 Tomas Babej
# https://github.com/tbabej/gibbs
#
# This software is released under MIT licence.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#


def markov_network(graph: 'nx.Graph') -> dimod.BinaryQuadraticModel:
    """Construct a binary quadratic model for a Markov network.

    Args:
        graph:
            A Markov network (in a NetworkX graph) as returned by
            :func:`dwave.graphs.generators.markov.markov_network`.

    Returns:
        A binary quadratic model.
    """

    if not hasattr(graph, 'nodes') or not hasattr(graph, 'edges'):
        raise ValueError("A NetworkX graph with potentials data required")

    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # the variable potentials
    for v, ddict in graph.nodes(data=True, default=None):
        potential = ddict.get('potential', None)

        if potential is None:
            continue

        # for single nodes we don't need to worry about order

        phi0 = potential[(0,)]
        phi1 = potential[(1,)]

        bqm.add_variable(v, phi1 - phi0)
        bqm.offset += phi0

    # the interaction potentials
    for u, v, ddict in graph.edges(data=True, default=None):
        potential = ddict.get('potential', None)

        if potential is None:
            continue

        # in python<=3.5 the edge order might not be consistent so we use the
        # one that was stored
        order = ddict['order']
        u, v = order

        phi00 = potential[(0, 0)]
        phi01 = potential[(0, 1)]
        phi10 = potential[(1, 0)]
        phi11 = potential[(1, 1)]

        bqm.add_variable(u, phi10 - phi00)
        bqm.add_variable(v, phi01 - phi00)
        bqm.add_interaction(u, v, phi11 - phi10 - phi01 + phi00)
        bqm.offset += phi00

    return bqm
