"""
Structured
==========

A structured sampler can only sample from binary quadratic models with a specific graph.

Creating a Structured Sampler
-----------------------------

A simple example of a structured sampler is one that only has two variables and one interaction.

>>> class TwoVariablesSampler(dimod.Sampler, dimod.Structured):
...     def __init__(self):
...         dimod.Sampler()
...         dimod.Structured(self, [0, 1], [(0, 1)])

This sampler as constructed won't work (because we have not overwritten any of the sample
methods), but it is sufficient to demonstrate accessing the properties.

First the structure is made a property of the sampler and it can be accessed appropriately.

>>> sampler = TwoVariablesSampler()
>>> sampler.structure
Structure([0, 1], [(0, 1)], {0: {1}, 1: {0}})  # namedtuple
>>> sampler.structure.nodelist
[0, 1]
>>> sampler.structure.edgelist
[(0, 1)]
>>> sampler.structure.adjacency
{0: {1}, 1: {0}}

Each of these components are also stored as properties of the sampler.

>>> sampler = TwoVariablesSampler()
>>> sampler.properties
{'structure': Structure([0, 1], [(0, 1)], {0: {1}, 1: {0}}),
 'nodelist': [0, 1],
 'edgelist': [(0, 1)],
 'adjacency': {0: {1}, 1: {0}}}}

 If you want the structure of the given problem to be checked against the sampler's structure, you
 can enforce this with a decorator.

>>> class TwoVariablesSampler(dimod.Sampler, dimod.Structured):
...     def __init__(self):
...         dimod.Sampler()
...         dimod.Structured(self, [0, 1], [(0, 1)])
...
...     @dimod.decorators.bqm_structured
...     def sample(self, bqm):
...         # All bqm's passed in will be a subgraph of the sampler's structure
...         variable_list = list(bqm.linear)
...         response = dimod.Response(bqm.vartype)
...         for values in itertools.product(bqm.vartype.value, len(bqm)):
...             sample = dict(zip(variable_list, values))
...             energy = bqm.energy(sample)
...             response.add_sample(sample, energy)
...         return response

Creating a Structured Composite
-------------------------------

todo

"""
from collections import namedtuple

Structure = namedtuple("Structure", ['nodelist', 'edgelist', 'adjacency'])


class Structured():
    """todo

    """
    def __init__(self, nodelist, edgelist, sort_nodes=True, sort_edges=True):

        if sort_nodes:
            nodelist = sorted(nodelist)
        elif not isinstance(nodelist, list):
            nodelist = list(nodelist)

        if sort_edges:
            edgelist = sorted(tuple(sorted(edge)) for edge in edgelist)
        elif not isinstance(edgelist, list):
            edgelist = list(edgelist)

        adjacency = {v: set() for v in nodelist}
        for u, v in edgelist:
            if v in adjacency[u]:
                raise ValueError("Each edge in edgelist must be unique")
            adjacency[u].add(v)
            adjacency[v].add(u)

        self.structure = Structure(nodelist, edgelist, adjacency)
        try:
            self.properties['structure'] = self.structure
            self.properties['nodelist'] = nodelist
            self.properties['edgelist'] = edgelist
            self.properties['adjacency'] = adjacency
        except AttributeError:
            raise
