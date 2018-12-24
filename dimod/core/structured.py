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
"""

A structured sampler can only sample from binary quadratic models with a specific graph.

For structured samplers you must implement the :attr:`~.Structured.nodelist`
and :attr:`~.Structured.edgelist` properties. The :class:`.Structured` abstract base class
provides access to the :attr:`~.Structured.structure` and :attr:`~.Structured.adjacency`
properties as well as any method or properties required by the :class:`.Sampler` abstract
base class. The :obj:`~.bqm_structured` decorator verifies that any given binary quadratic
model conforms to the supported structure.

Examples:
    This simple example shows a structured sampler that can only sample from a binary quadratic model
    with two variables and one interaction.

    .. code-block:: python

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

                return dimod.SampleSet.from_samples(samples, bqm.Vartype, energies)

                return response

"""
import abc

from collections import namedtuple

from six import add_metaclass

__all__ = ['Structured']

_Structure = namedtuple("Structure", ['nodelist', 'edgelist', 'adjacency'])


@add_metaclass(abc.ABCMeta)
class Structured:
    """The abstract base class for dimod structured samplers.

    Provides the :attr:`.Structured.adjacency` and :attr:`.Structured.structure` properties.

    Abstract properties :attr:`~.Structured.nodelist` and :attr:`~.Structured.edgelist`
    must be implemented.

    """
    @abc.abstractproperty
    def nodelist(self):
        """list: Nodes/variables allowed by the sampler formatted as a list.

        Examples:
            This example defines the `nodelist` for a structured sampler that can only sample
            from a binary quadratic model with two variables.

            .. code-block:: python

                class TwoVariablesSampler(dimod.Sampler, dimod.Structured):
                    @property
                    def nodelist(self):
                        return [0, 1]

                    # Remaining properties and code for the sampler

        """
        pass

    @abc.abstractproperty
    def edgelist(self):
        """list[(node, node)]: Edges/interactions allowed by the sampler, formatted as
        a list where each edge/interaction is a 2-tuple.

        Examples:
            This example defines the `edgelist` for a structured sampler that can only sample
            from a binary quadratic model with a single interaction between two variables.

            .. code-block:: python

                class TwoVariablesSampler(dimod.Sampler, dimod.Structured):
                    @property
                    def edgelist(self):
                        return [(0, 1)]

                    # Remaining properties and code for the sampler

        """
        pass

    @property
    def adjacency(self):
        """dict[variable, set]: Adjacency structure formatted as a dict, where keys are the
        nodes of the structured sampler and values are sets of all adjacent nodes for
        each key node.

        Examples:
            This example shows the adjacencies for a placeholder structured sampler that
            samples only from the K4 complete graph, where each of the four nodes connects
            to all the other nodes.

            >>> class K4StructuredClass(dimod.Structured):
            ...     @property
            ...     def nodelist(self):
            ...         return [1, 2, 3, 4]
            ...
            ...     @property
            ...     def edgelist(self):
            ...         return [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
            >>> K4sampler = K4StructuredClass()
            >>> K4sampler.adjacency.keys()
            [1, 2, 3, 4]

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
    def structure(self):
        """Structure of the structured sampler formatted as a :class:`~collections.namedtuple`
        :samp:`Structure(nodelist, edgelist, adjacency)`, where the 3-tuple values are
        the :attr:`~.Structured.nodelist` and :attr:`~.Structured.edgelist` properties
        and :meth:`~.Structured.adjacency` method.

        Examples:
            This example shows the structure of a placeholder structured sampler that
            samples only from the K3 complete graph, where each of the three nodes connects
            to all the other nodes.

            >>> class K3StructuredClass(dimod.Structured):
            ...     @property
            ...     def nodelist(self):
            ...         return [1, 2, 3]
            ...
            ...     @property
            ...     def edgelist(self):
            ...         return [(1, 2), (1, 3), (2, 3)]
            >>> K3sampler = K3StructuredClass()
            >>> K3sampler.structure.edgelist
            [(1, 2), (1, 3), (2, 3)]

        """
        return _Structure(self.nodelist, self.edgelist, self.adjacency)
