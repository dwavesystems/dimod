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
A composite that fixes the variables provided and removes them from the binary
polynomial model before sending to its child sampler.
"""

from dimod.core.polysampler import ComposedPolySampler
from dimod.higherorder.polynomial import BinaryPolynomial
from collections import defaultdict
import dimod


__all__ = ['FixedPolyVariableComposite']


class FixedPolyVariableComposite(ComposedPolySampler):
    """Composite to fix variables of a problem to provided.

    Fixes variables of a BinaryPolynomial and modifies linear and k-local terms
    accordingly. Returned samples include the fixed variable

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

    Examples:
       This example uses :class:`.FixedPolyVariableComposite` to instantiate a
       composed sampler that submits a simple high order Ising problem to a sampler.
       The composed sampler fixes a variable and modifies linear and k-local terms
       biases according.

       >>> h = {1: -1.3, 2: 1.2, 3: -3.4, 4: -0.5}
       >>> J = {(1, 4): -0.6, (1, 2, 3): 0.2, (1, 2, 3, 4): -0.1}
       >>> poly = dimod.BinaryPolynomial.from_hising(h, J, offset=0)
       >>> sampler = dimod.FixedPolyVariableComposite(dimod.ExactPolySolver())
       >>> sampleset = sampler.sample_poly(poly, fixed_variables={3: -1, 4: 1})

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

    def sample_poly(self, poly, fixed_variables=None, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            poly (:obj:`dimod.BinaryPolynomial`):
                Binary polynomial model to be sampled from.

            fixed_variables (dict):
                A dictionary of variable assignments.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """

        child = self.child
        if fixed_variables is None:
            sampleset = child.sample_poly(poly, **parameters)
            return sampleset
        else:
            poly_copy = fix_variables(poly, fixed_variables)
            sampleset = child.sample_poly(poly_copy, **parameters)
            if len(sampleset):
                return sampleset.append_variables(fixed_variables)
            elif fixed_variables:
                return type(sampleset).from_samples_bqm(fixed_variables, bqm=poly)
            else:
                return sampleset


def fix_variables(poly, fixed_variables):
    if () in poly.keys():
        offset = poly[()]
    else:
        offset = 0.0
    poly_copy = defaultdict(float)
    for k, v in poly.items():
        k = set(k)
        for var, value in fixed_variables.items():
            if var in k:
                k -= {var}
                v *= value
        k = frozenset(k)
        if len(k) > 0:
            poly_copy[k] += v
        else:
            offset += v
    poly_copy[()] = offset
    return BinaryPolynomial(poly_copy, poly.vartype)
