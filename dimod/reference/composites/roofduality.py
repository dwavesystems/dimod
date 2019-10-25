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
"""A composite that uses the roof duality algorithm [#bht]_ [#bh]_ to fix some
variables in the binary quadratic model before passing it on to its child
sampler.

.. [#bht] Boros, E., P.L. Hammer, G. Tavares. Preprocessing of Unconstrained
    Quadratic Binary Optimization. Rutcor Research Report 10-2006, April, 2006.

.. [#bh] Boros, E., P.L. Hammer. Pseudo-Boolean optimization. Discrete Applied
    Mathematics 123, (2002), pp. 155-225

"""

from dimod.reference.composites.fixedvariable import FixedVariableComposite
from dimod.roof_duality import fix_variables
from dimod.roof_duality.extended_fix_variables import find_and_contract_all_variables_roof_duality, uncontract_sampleset
from dimod.core.composite import ComposedSampler

__all__ = ['RoofDualityComposite', 'ExtendedRoofDualityComposite']


class RoofDualityComposite(FixedVariableComposite):
    """Uses roof duality to assign some variables before invoking child sampler.

    Uses the :func:`~dimod.roof_duality.fix_variables` function to determine
    variable assignments, then fixes them before calling the child sampler.
    Returned samples include the fixed variables.

    Args:
       child (:obj:`dimod.Sampler`):
            A dimod sampler. Used to sample the bqm after variables have been
            fixed.

    """

    @property
    def parameters(self):
        params = self.child.parameters.copy()
        params['sampling_mode'] = []
        return params

    def sample(self, bqm, sampling_mode=True, **parameters):
        """Sample from the provided binary quadratic model.

        Uses the :func:`~dimod.roof_duality.fix_variables` function to determine
        which variables to fix.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            sampling_mode (bool, optional, default=True):
                In sampling mode, only roof-duality is used. When
                `sampling_mode` is false, strongly connected components are used
                to fix more variables, but in some optimal solutions these
                variables may take different values.

            **parameters:
                Parameters for the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        # use roof-duality to decide which variables to fix
        parameters['fixed_variables'] = fix_variables(bqm, sampling_mode=sampling_mode)
        return super(RoofDualityComposite, self).sample(bqm, **parameters)


class ExtendedRoofDualityComposite(ComposedSampler):
    """Uses extended roof duality to assign some variables before invoking child sampler.

    Uses the :func:`~dimod.roof_duality.find_and_contract_all_variables_roof_duality` function to determine
    variable assignments, then fixes them before calling the child sampler. Variables may be fixed to values or to
    other variables.
    Returned samples include all original variables.
    Extended roof duality can potentially fix more variables than roof duality, but it can also be significantly slower.

    Args:
       child (:obj:`dimod.Sampler`):
            A dimod sampler. Used to sample the bqm after variables have been
            fixed.

    """


    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        params = self.child.parameters.copy()
        params['sampling_mode'] = []
        return params

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}


    def sample(self, bqm, sampling_mode=True, **parameters):
        """Sample from the provided binary quadratic model.

        Uses the :func:`~dimod.roof_duality.find_and_contract_all_variables_roof_duality` function to determine
        which variables to fix.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            sampling_mode (bool, optional, default=True):
                In sampling mode, only roof-duality is used. When
                `sampling_mode` is false, strongly connected components are used
                to fix more variables, but in some optimal solutions these
                variables may take different values.

            **parameters:
                Parameters for the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """

        # use extended roof duality to decide which variables to fix and contract
        bqm_small, variable_map, fixed_variables = find_and_contract_all_variables_roof_duality(bqm,
                                                                                            sampling_mode=sampling_mode)

        # solve the problem on the child system
        sampleset_small = self.child.sample(bqm_small, **parameters)

        # add fixed variables back in
        if len(sampleset_small):
            sampleset = sampleset_small.append_variables(fixed_variables)
        elif fixed_variables:
            sampleset = type(sampleset_small).from_samples_bqm(fixed_variables, bqm=bqm)
        else:
            # no fixed variables and sampleset is empty
            sampleset = sampleset_small

        # uncontract variables.
        return uncontract_sampleset(sampleset, variable_map)



