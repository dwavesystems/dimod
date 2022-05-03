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

"""

A structured sampler can only sample from binary quadratic models with a specific graph.

For structured samplers you must implement the :attr:`~.Structured.nodelist` and
:attr:`~.Structured.edgelist` properties. The :class:`.Structured` abstract base
class provides access to the :attr:`~.Structured.structure` and
:attr:`~.Structured.adjacency` properties, the
:meth:`~.Structured.to_networkx_graph` method, as well as any method or
properties required by the :class:`.Sampler` abstract base class. The
:obj:`~.bqm_structured` decorator verifies that any given binary quadratic model
conforms to the supported structure.

Examples:
    This simple example shows a structured sampler that can only sample from a binary
    quadratic model with two variables and one interaction.

    .. testcode::

        class TwoVariablesSampler(dimod.Sampler, dimod.Structured):
            @property
            def nodelist(self):
                return [0, 1]

            @property
            def edgelist(self):
                return [(0, 1)]

            @property
            def properties(self):
                return dict()

            @property
            def parameters(self):
                return dict()

            @dimod.decorators.bqm_structured
            def sample(self, bqm):
                # All bqm's passed in will be a subgraph of the sampler's structure
                variable_list = list(bqm.linear)
                samples = []
                energies = []
                for values in itertools.product(bqm.vartype.value, repeat=len(bqm)):
                    sample = dict(zip(variable_list, values))
                    samples.append(sample)
                    energies.append(bqm.energy(sample))

                return dimod.SampleSet.from_samples(samples, bqm.vartype, energies)

                return response

    >>> import itertools
    >>> sampler = TwoVariablesSampler()
    >>> solutions = sampler.sample_ising({}, {(0, 1): -1})
    >>> solutions.first.energy
    -1.0

"""
from __future__ import annotations

from collections import namedtuple

import abc
import typing

from dimod.typing import Variable

if typing.TYPE_CHECKING:
    from dimod import BinaryQuadraticModel

__all__ = ['Structured']


class _Structure(typing.NamedTuple):
    nodelist: typing.List[Variable]
    edgelist: typing.List[typing.Tuple[Variable, Variable]]
    adjacency: typing.Dict[Variable, typing.Set]


class Structured(abc.ABC):
    """The abstract base class for dimod structured samplers.

    Provides the :attr:`.Structured.adjacency` and :attr:`.Structured.structure`
    properties, and the :meth:`.Structured.to_networkx_graph` method.

    Abstract properties :attr:`~.Structured.nodelist` and :attr:`~.Structured.edgelist`
    must be implemented.

    """
    @abc.abstractproperty
    def nodelist(self) -> typing.List[Variable]:
        """Nodes/variables allowed by the sampler."""
        pass

    @abc.abstractproperty
    def edgelist(self) -> typing.List[typing.Tuple[Variable, Variable]]:
        """Edges/interactions allowed by the sampler.
        """
        pass

    @property
    def adjacency(self) -> typing.Dict[Variable, typing.Set]:
        """Adjacency structure formatted as a dict, where keys are the nodes of
        the structured sampler and values are sets of all adjacent nodes for each
        key node.
        """
        if not hasattr(self, '_adjacency'):
            adjacency = {v: set() for v in self.nodelist}
            for u, v in self.edgelist:
                if v in adjacency[u]:
                    raise ValueError("Each edge in edgelist must be unique")
                adjacency[u].add(v)
                adjacency[v].add(u)
            self._adjacency = adjacency
            return adjacency
        return self._adjacency

    @property
    def structure(self) -> _Structure:
        """Structure of the structured sampler formatted as a
        :func:`~collections.namedtuple` where the 3-tuple values are the
        :attr:`.nodelist`, :attr:`.edgelist` and :attr:`.adjacency` attributes.
        """
        return _Structure(self.nodelist, self.edgelist, self.adjacency)

    def to_networkx_graph(self):
        """Convert structure to NetworkX graph format.

        Note that NetworkX must be installed for this method to work.

        Returns:
            :class:`networkx.Graph`: A NetworkX graph containing the nodes and
            edges from the sampler's structure.

        """
        import networkx as nx

        G = nx.Graph(self.edgelist)
        G.add_nodes_from(self.nodelist)

        return G

    def valid_bqm_graph(self, bqm: BinaryQuadraticModel) -> bool:
        """Validate that problem defined by :class:`dimod.BinaryQuadraticModel`
        matches the graph provided by the sampler.

        Args:
            bqm: :class:`dimod.BinaryQuadraticModel` object to validate.

        Returns:
            Boolean indicating validity of BQM graph compared to sampler graph.

        """
        return all(variable in self.adjacency for variable in bqm.variables) and \
               all(neighbor in self.adjacency[vertex] for neighbor, vertex in bqm.quadratic)
