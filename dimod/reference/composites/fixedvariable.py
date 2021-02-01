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
# =============================================================================
"""
A composite that fixes the variables provided and removes them from the binary
quadratic model before sending to its child sampler.
"""
import numpy as np

from dimod.bqm import as_bqm, AdjVectorBQM, AdjDictBQM
from dimod.core.composite import ComposedSampler
from dimod.sampleset import SampleSet, append_variables


__all__ = ['FixedVariableComposite']


class FixedVariableComposite(ComposedSampler):
    """Composite to fix variables of a problem to provided.

    Fixes variables of a binary quadratic model (BQM) and modifies linear and
    quadratic terms accordingly. Returned samples include the fixed variable

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

    Examples:
       This example uses :class:`.FixedVariableComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler fixes a variable and modifies linear and quadratic
       biases according.

       >>> h = {1: -1.3, 4: -0.5}
       >>> J = {(1, 4): -0.6}
       >>> sampler = dimod.FixedVariableComposite(dimod.ExactSolver())
       >>> sampleset = sampler.sample_ising(h, J, fixed_variables={1: -1})

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        params = self.child.parameters.copy()
        params['fixed_variables'] = []
        return params

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, fixed_variables=None, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            fixed_variables (dict):
                A dictionary of variable assignments.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """

        if not fixed_variables:  # None is falsey
            return self.child.sample(bqm, **parameters)

        # make sure that we're shapeable and that we have a BQM we can mutate
        bqm_copy = as_bqm(bqm, cls=[AdjVectorBQM, AdjDictBQM],
                          copy=True)

        bqm_copy.fix_variables(fixed_variables)

        sampleset = self.child.sample(bqm_copy, **parameters)

        def _hook(sampleset):
            # make RoofDualityComposite non-blocking

            if sampleset.variables:
                if len(sampleset):
                    return append_variables(sampleset, fixed_variables)
                else:
                    return sampleset.from_samples_bqm((np.empty((0, len(bqm))),
                                                       bqm.variables), bqm=bqm)

            # there are only fixed variables, make sure that the correct number
            # of samples are returned
            samples = [fixed_variables]*max(len(sampleset), 1)

            return sampleset.from_samples_bqm(samples, bqm=bqm)

        return SampleSet.from_future(sampleset, _hook)
