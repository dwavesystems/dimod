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
A composite that fixes the variables provided and removes them from the
bqm object before sending to its child sampler.

See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations
of technical terms in descriptions of Ocean tools.
"""
import numpy as np

from dimod.core.composite import ComposedSampler
from dimod.response import SampleSet

__all__ = ['FixedVariableComposite']


class FixedVariableComposite(ComposedSampler):
    """Composite to fix variables of a problem to provided, assigned values

    Inherits from :class:`dimod.ComposedSampler`.

    Fixes variables of a bqm and modifies linear and quadratic terms
    accordingly. Returned samples include the fixed variable

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

    Examples:
       This example uses :class:`.FixedVariableComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler fixes a variable and modifies linear and quadratic
       biases according.

       >>> import dimod
       >>> sampler = dimod.FixedVariableComposite(dimod.ExactSolver())
       >>> linear = {1: -1.3, 4: -0.5}
       >>> quadratic = {(1, 4): -0.6}
       >>> response = sampler.sample_ising(linear,quadratic,fixed_variables={1: -1})
       >>> print(response.first)  # doctest: +SKIP
       Sample(sample={1: -1, 4: -1}, energy=1.2000000000000002, num_occurrences=1)

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        return self.child.parameters.copy()

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

        # solve the problem on the child system
        child = self.child
        bqm_copy = bqm.copy()
        if fixed_variables is None:
            fixed_variables = {}

        bqm_copy.fix_variables(fixed_variables)
        response = child.sample(bqm_copy, **parameters)
        return _release_response(response, fixed_variables)


def _release_response(response, fixed_variables):
    """will add the fixed variables and their assigned values to the samples
       of the response object. Energies of the response do not change since
       in fixing step the offset is populated by the constant energy shift
       associated with fixing the variables.

    Args:
        response (:obj:`.SampleSet`):
            Samples from the bqm with fixed variables.

        fixed_variables (dict):
            The dict of fixed variables and their assigned values.
            These are the variables that will be added back to the samples
            of the response object.

    Returns:
        :obj:`dimod.SampleSet`:
            Samples for the source binary quadratic model.

    Examples:
       This example uses :class:`.FixedVariableComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler fixes a variable and modifies linear and quadratic
       biases according.

       >>> import dimod
       >>> sampler = dimod.FixedVariableComposite(dimod.ExactSolver())
       >>> h = {'d': -4}
       >>> J = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1, ('c', 'd'): -.1}
       >>> bqm = dimod.BinaryQuadraticModel.from_ising(h, J)
       >>> fixed_variables = dimod.roof_duality.fix_variables(bqm)
       >>> response = sampler.sample(bqm, fixed_variables=fixed_variables)
       >>> print(response.first)
       Sample(sample={'a': -1, 'b': 1, 'c': 1, 'd': 1}, energy=-5.1, num_occurrences=1)

    """

    record = response.record
    original_variables = list(response.variables)
    samples = np.asarray(record.sample)
    energy = np.asarray(record.energy)

    num_samples, num_variables = np.shape(samples)
    num_variables += len(fixed_variables)

    if len(fixed_variables) > 0:
        b = []
        for v, val in fixed_variables.items():
            original_variables.append(v)
            b.append([val] * num_samples)
        samples = np.concatenate((samples, np.transpose(b)), axis=1)

    datatypes = [('sample', np.dtype(np.int8), (num_variables,)),
                 ('energy', energy.dtype)]

    datatypes.extend((name, record[name].dtype, record[name].shape[1:])
                     for name in record.dtype.names if
                     name not in {'sample',
                                  'energy'})

    data = np.rec.array(np.empty(num_samples, dtype=datatypes))

    data.sample = samples
    data.energy = energy
    for name in record.dtype.names:
        if name not in {'sample', 'energy'}:
            data[name] = record[name]

    return SampleSet(data, original_variables, response.info,
                     response.vartype)
