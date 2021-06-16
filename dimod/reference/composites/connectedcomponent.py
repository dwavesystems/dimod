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
#
# =============================================================================
"""
A composite that breaks the problem into sub-problems corresponding to the
connected components of the binary
quadratic model graph before sending to its child sampler.
"""

from dimod.binary.binary_quadratic_model import as_bqm
from dimod.core.composite import ComposedSampler
from dimod.sampleset import SampleSet, append_variables
from dimod.traversal import connected_components

__all__ = ['ConnectedComponentsComposite']


class ConnectedComponentsComposite(ComposedSampler):
    """Composite to decompose a problem to the connected components
    and solve each.

    Connected components of a binary quadratic model (BQM) graph are computed
    (if not provided), and each subproblem is passed to the child sampler.
    Returned samples from each child sampler are merged. Only the best solution
    of each response is pick and merge with others
    (i.e. this composite returns a single solution).

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

    Examples:
       This example uses :class:`.ConnectedComponentsComposite` to solve a simple
       Ising problem that can be separated into two components. This small example
       uses :class:`dimod.ExactSolver` and is just illustrative.

       >>> h = {}
       >>> J1 = {(1, 2): -1.0, (2, 3): 2.0, (3, 4): 3.0}
       >>> J2 = {(12, 13): 6}
       >>> sampler = dimod.ExactSolver()
       >>> sampler_ccc = dimod.ConnectedComponentsComposite(sampler)
       >>> e1 = sampler.sample_ising(h, J1).first.energy
       >>> e2 = sampler.sample_ising(h, J2).first.energy
       >>> e_ccc = sampler_ccc.sample_ising(h, {**J1, **J2}).first.energy
       >>> e_ccc == e1 + e2
       True

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        params = self.child.parameters.copy()
        return params

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, components=None, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            components (list(set)):
                A list of disjoint set of variables that fully partition the variables

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        # make sure the BQM is shapeable
        bqm = as_bqm(bqm)

        # solve the problem on the child system
        child = self.child
        variables = bqm.variables
        if components is None:
            components = list(connected_components(bqm))
        if isinstance(components, set):
            components = [components]
        sampleset = None
        fixed_value = min(bqm.vartype.value)
        for component in components:
            bqm_copy = bqm.copy()
            bqm_copy.fix_variables({i: fixed_value for i in (variables - component)})
            if sampleset is None:
                # here .truncate(1) is used to pick the best solution only. The other options
                # for future development is to combine all sample with all.
                # This way you'd get the same behaviour as the ExactSolver
                sampleset = child.sample(bqm_copy, **parameters).truncate(1)
            else:
                sampleset = append_variables(sampleset.truncate(1), child.sample(bqm_copy, **parameters).truncate(1))

        if sampleset is None:
            return SampleSet.from_samples_bqm({}, bqm)
        else:
            return SampleSet.from_samples_bqm(sampleset, bqm)
