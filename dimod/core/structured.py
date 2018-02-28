"""

A structured sampler can only sample from binary quadratic models with a specific graph.

A simple example of a structured sampler is one that can only sample from a binary quadratic model
with two variables and one interaction

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

            response = dimod.Response.from_dicts(samples, {'energy': energies}, vartype=bqm.vartype)

            return response

By consulting the table above, we know that we need to implement the :attr:`~.Structured.nodelist`
and :attr:`~.Structured.edgelist` properties. By doing so we can access the
:attr:`~.Structured.structure` and :attr:`~.Structured.adjacency` properties. As well as any method
or properties required by the :class:`.Sampler` abstract base class.

An additional benefit of the :class:`.Structured` abstract base class is the
:obj:`~.bqm_structured` decorator which will check the given binary quadratic model for the correct
structure.

"""
from collections import namedtuple

from six import add_metaclass

import dimod.abc as abc

__all__ = ['Structured']

_Structure = namedtuple("Structure", ['nodelist', 'edgelist', 'adjacency'])


@add_metaclass(abc.ABCMeta)
class Structured:
    """The abstract base class for dimod Structured samplers.

    Provides the :attr:`.Structured.adjacency` and :attr:`.Structured.structure` properties.

    """
    @abc.abstractproperty
    def nodelist(self):
        """list: Should be a list of the nodes/variables allowed by the sampler."""
        pass

    @abc.abstractproperty
    def edgelist(self):
        """list[(node, node)]: Should be a list of the edges/interactions allowed by the sampler.
        Each edge/interaction should be a 2-tuple.
        """
        pass

    @property
    def adjacency(self):
        """dict[variable, set]: The adjacency structure.

        Examples:

            >>> class StructuredObject(dimod.Structured):
            ...     @property
            ...      def nodelist(self):
            ...         return [0, 1, 2]
            ...
            ...     @property
            ...     def edgelist(self):
            ...         return [(0, 1), (1, 2)]
            >>> test_obj = StructuredObject()
            >>> for u, v in test_obj.edgelist:
            ...     assert u in test_obj.adjacency[v]
            ...     assert v in test_obj.adjacency[u]

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
        """A :class:`~collections.namedtuple` :samp:`Structure(nodelist, edgelist, adjacency)`"""
        return _Structure(self.nodelist, self.edgelist, self.adjacency)
