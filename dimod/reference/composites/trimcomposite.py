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
# ================================================================================================
"""
A composite that trims the response based on options provided by the user.

See `Ocean Glossary <https://docs.ocean.dwavesys.com/en/latest/glossary.html>`_ for explanations
of technical terms in descriptions of Ocean tools.
"""

import numpy as np

from dimod.core.composite import ComposedSampler

__all__ = ['TrimComposite']


class TrimComposite(ComposedSampler):
    """Composite to trim the returned samples

    Inherits from :class:`dimod.ComposedSampler`.

    Post-processing is expensive and sometimes one might want to only
    treat the lowest energy samples. This composite layer allows one to
    pre-select the samples within a multi-composite pipeline

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler

    Note:
        :attr:if aggregate is True `.SampleSet.record.num_occurrences` are
        accumulated but no other fields are.

    Examples:
       This example uses :class:`.ScaleComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler returns the n lowest energy samples and aggregates
       them if aggregate=True.

       >>> linear = {'a': -4.0, 'b': -4.0}
       >>> quadratic = {('a', 'b'): 3.2}
       >>> sampler = dimod.TrimComposite(dimod.ExactSolver())
       >>> response = sampler.sample_ising(linear, quadratic, n=1) # doctest: +SKIP
            a   b  energy  num_occ.
        0  +1  +1    -4.8         1

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        param = self.child.parameters.copy()
        param.update({'n': [],
                      'aggregate': []})
        return param

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, n=None, aggregate=False, **parameters):
        """ Sample from the problem provided by bqm and trim output

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            n (int): number of lowest energy samples to return. if None, will
                     return all samples

            aggregate (bool): Toggle for sample aggregation

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        if n is None:
            n = 0
        if not (isinstance(n, int) and n >= 0):
            raise ValueError('number of samples requested must be >0. '
                             'receieved {}'.format(n))

        child = self.child
        response = child.sample(bqm, **parameters)

        return _trim(response, n, aggregate)

    def sample_ising(self, h, J, n=None, aggregate=False,
                     **parameters):
        """ Sample from the problem provided by h, J, offset and trim output

        Args:
            h (dict): linear biases

            J (dict): quadratic or higher order biases

            n (int): number of lowest energy samples to return. if None, will
                     return all samples

            aggregate (bool): Toggle for sample aggregation

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        if n is None:
            n = 0
        if not (isinstance(n, int) and n >= 0):
            raise ValueError('number of samples requested must be >0. '
                             'receieved {}'.format(n))

        child = self.child
        response = child.sample_ising(h, J, **parameters)

        return _trim(response, n, aggregate)


def _trim(sampleset, n, aggregate):
    """Create a new SampleSet

       Samples are aggregated and trimmed to n lowest energy samples

    Returns:
        :obj:`.SampleSet`

    Note:
        :attr:`.SampleSet.record.num_occurrences` are accumulated but no
        other fields are.

    """

    if aggregate:
        sampleset = sampleset.aggregate()

    if n > 0:
        record = sampleset.record
        sort_indices = np.argsort(record['energy'])
        record = record[sort_indices[:n]]

        sampleset = sampleset.__class__(record, sampleset.variables,
                                        sampleset.info,
                                        sampleset.vartype)

    return sampleset
