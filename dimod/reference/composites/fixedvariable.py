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
"""
import numpy as np
from dimod.response import SampleSet

import dimod

__all__ = ['FixedVariableComposite']


class FixedVariableComposite(dimod.ComposedSampler):
    """Composite to fix variables of a problem to provided, assigned values

    Inherits from :class:`dimod.ComposedSampler`.

    Fixes variables of a bqm and modifies linear and quadratic terms
    accordingly. Returned samples include the fixed variable

    Args:
       sampler (:class:`dimod.Sampler`):
            A dimod sampler

    Examples:
       This example uses :class:`.FixedVariableComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler fixes a variable and modifies linear and quadratic
       biases according.

       >>> from dwave.system.samplers import DWaveSampler
       >>> from dimod.binary_quadratic_model import BinaryQuadraticModel
       >>> from dimod.reference.composites.fixedvariable import FixedVariableComposite
       >>> import dimod
       >>> sampler = FixedVariableComposite(DWaveSampler())
       >>> h = {1: -1.1, 4: -0.5}
       >>> J = {(1,4):-0.5}
       >>> bqm = BinaryQuadraticModel(linear= h,quadratic = J ,offset=0,
       >>> vartype=dimod.SPIN)
       >>> response = sampler.sample(bqm,fixed_variables={1:-1})
       >>> for sample in response.samples():    # doctest: +SKIP
       ...     print(sample)
       ...
       fixed variables =  {1: -1}
       {4: -1, 1: -1}

    See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations
    of technical terms in descriptions of Ocean tools.

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        """list: Children property inherited from :class:`dimod.Composite` class.

        For an instantiated composed sampler, contains the single wrapped structured sampler.

        Examples:
            This example instantiates a composed sampler using a D-Wave solver selected by
            the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`
            and views the solver's parameters.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dimod.reference.composites.fixedvariable import FixedVariableComposite
            >>> sampler = FixedVariableComposite(DWaveSampler())
            >>> print(sampler.children)   # doctest: +SKIP
            [<dwave.system.samplers.dwave_sampler.DWaveSampler object at 0x7f88a8aa9080>]

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations of technical terms in descriptions of Ocean tools.

        """

        return self._children

    @property
    def parameters(self):
        """dict[str, list]: Parameters in the form of a dict.

        For an instantiated composed sampler, keys are the keyword parameters accepted by the child sampler.

        Examples:
            This example instantiates a composed sampler using a D-Wave solver selected by
            the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`
            and views the solver's parameters.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dimod.reference.composites.fixedvariable import FixedVariableComposite
            >>> sampler = FixedVariableComposite(DWaveSampler())
            >>> sampler.parameters   # doctest: +SKIP
            {'anneal_offsets': ['parameters'],
             'anneal_schedule': ['parameters'],
             'annealing_time': ['parameters'],
             'answer_mode': ['parameters'],
             'auto_scale': ['parameters'],
            >>> # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations of technical terms in descriptions of Ocean tools.

        """
        # does not add or remove any parameters
        return self.child.parameters.copy()

    @property
    def properties(self):
        """dict: Properties in the form of a dict.

        For an instantiated composed sampler, contains one key :code:`'child_properties'` that
        has a copy of the child sampler's properties.

        Examples:
            This example instantiates a composed sampler using a D-Wave solver selected by
            the user's default
            :std:doc:`D-Wave Cloud Client configuration file <cloud-client:reference/intro>`
            and views the solver's properties.

            >>> from dwave.system.samplers import DWaveSampler
            >>> from dimod.reference.composites.fixedvariable import FixedVariableComposite
            >>> sampler = FixedVariableComposite(DWaveSampler())
            >>> sampler.properties   # doctest: +SKIP
            {'child_properties': {u'anneal_offset_ranges': [[-0.2197463755538704,
                0.03821687759418928],
               [-0.2242514597680286, 0.01718456460967399],
               [-0.20860153999435985, 0.05511969218508182],
            >>> # Snipped above response for brevity

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations of technical terms in descriptions of Ocean tools.

        """

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
            :class:`dimod.SampleSet`

        See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_
        for explanations of technical terms in descriptions of Ocean tools.
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
        response (:obj:`.Response`):
            Response from the bqm with fixed variables.

        fixed_variables (dict):
            The dict of fixed variables and their assigned values.
            These are the variables that will be added back to the samples
            of the response object.

    Returns:
        :obj:`.SampleSet`:
            Response for the source binary quadratic model.

    """

    record = response.record
    original_variables = list(response.variables)
    samples = np.asarray(record.sample)
    energy = np.asarray(record.energy)

    num_samples, num_variables = np.shape(samples)
    num_variables += len(fixed_variables)

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
