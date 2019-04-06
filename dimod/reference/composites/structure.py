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
A composite that structures a sampler.
"""
from dimod.core.sampler import Sampler
from dimod.core.composite import Composite
from dimod.core.structured import Structured
from dimod.decorators import bqm_structured


class StructureComposite(Sampler, Composite, Structured):
    """Creates a structured composed sampler from an unstructured sampler.

    Args:
        Sampler (:obj:`~dimod.Sampler`):
            Unstructured sampler.
        nodelist (list):
            Nodes/variables allowed by the sampler formatted as a list.
        edgelist (list[(node, node)]):
            Edges/interactions allowed by the sampler, formatted as a list where each
            edge/interaction is a 2-tuple.

    Examples:
        This example creates a composed sampler from the unstructure dimod ExactSolver sampler.
        The target structure is a square graph.

        >>> import dimod
        ...
        >>> base_sampler = dimod.ExactSolver()
        >>> node_list = [0, 1, 2, 3]
        >>> edge_list = [(0, 1), (1, 2), (2, 3), (0, 3)]
        >>> structured_sampler = dimod.StructureComposite(base_sampler, node_list, edge_list)
        >>> linear = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        >>> quadratic = {(0, 1): 1.0, (1, 2): 1.0, (0, 3): 1.0, (2, 3): -1.0}
        >>> bqm = dimod.BinaryQuadraticModel(linear, quadratic, 1.0, dimod.Vartype.SPIN)
        >>> response = structured_sampler.sample(bqm)
        >>> print(next(response.data()))
        Sample(sample={0: 1, 1: -1, 2: -1, 3: -1}, energy=-1.0, num_occurrences=1)
        >>> # Try giving the composed sampler a non-square model
        >>> del quadratic[(0, 1)]
        >>> quadratic[(0, 2)] = 1.0
        >>> bqm = dimod.BinaryQuadraticModel(linear, quadratic, 1.0, dimod.Vartype.SPIN)
        >>> try: response = structured_sampler.sample(bqm)    # doctest: +SKIP
        ... except dimod.BinaryQuadraticModelStructureError as details:
        ...     print(details)
        ...
        given bqm does not match the sampler's structure

    """
    # we will override these in the __init__, but because they are abstract properties we need to
    # signal that we are overriding them
    edgelist = None
    nodelist = None
    children = None

    def __init__(self, sampler, nodelist, edgelist):
        self.children = [sampler]
        self.nodelist = nodelist
        self.edgelist = edgelist

    @property
    def parameters(self):
        return self.child.parameters

    @property
    def properties(self):
        return self.child.properties

    @bqm_structured
    def sample(self, bqm, **sample_kwargs):
        """Sample from the binary quadratic model.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

        Returns:
            :obj:`.SampleSet`

        Examples:
            This example submits an Ising problem to a composed sampler that uses
            the dimod ExactSampler only on problems structured for a K2 fully connected graph.

            >>> import dimod
            ...
            >>> response = dimod.StructureComposite(dimod.ExactSolver(),
            ...                  [0, 1], [(0, 1)]).sample_ising({0: 1, 1: 1}, {})
            >>> print(next(response.data()))
            Sample(sample={0: -1, 1: -1}, energy=-2.0, num_occurrences=1)
        """
        return self.child.sample(bqm, **sample_kwargs)
