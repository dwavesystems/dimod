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
        """list: Nodes/variables allowed by the sampler."""
        pass

    @abc.abstractproperty
    def edgelist(self):
        """list: Edges/interactions allowed by the sampler in the form
        `[(u, v), ...]`.
        """
        pass

    @property
    def adjacency(self):
        """dict[variable, set]: Adjacency structure formatted as a dict, where
        keys are the nodes of the structured sampler and values are sets of all
        adjacent nodes for each key node.
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
        """Structure of the structured sampler formatted as a
        :class:`~collections.namedtuple`, `Structure(nodelist, edgelist, adjacency)`,
        where the 3-tuple values are the :attr:`.nodelist` and :attr:`.edgelist`
        properties and :meth:`.adjacency` method.
        """
        return _Structure(self.nodelist, self.edgelist, self.adjacency)
