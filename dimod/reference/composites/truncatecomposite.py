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
A composite that truncates the response based on options provided by the user.
"""

import numpy as np

from dimod.core.composite import ComposedSampler

__all__ = 'TruncateComposite',


class TruncateComposite(ComposedSampler):
    """Composite to truncate the returned samples

    Inherits from :class:`dimod.ComposedSampler`.

    Post-processing is expensive and sometimes one might want to only
    treat the lowest energy samples. This composite layer allows one to
    pre-select the samples within a multi-composite pipeline

    Args:
        child_sampler (:obj:`dimod.Sampler`):
            A dimod sampler

        n (int):
            Maximum number of rows in the returned sample set.

        sorted_by (str/None, optional, default='energy'):
            Selects the record field used to sort the samples before
            truncating. Note that sample order is maintained in the
            underlying array.

        aggregate (bool, optional, default=False):
            If True, aggregate the samples before truncating.

    Note:
        If aggregate is True :attr:`.SampleSet.record.num_occurrences` are
        accumulated but no other fields are.

    """

    def __init__(self, child_sampler, n, sorted_by='energy', aggregate=False):

        if n < 1:
            raise ValueError('n should be a positive integer, recived {}'.format(n))

        self._children = [child_sampler]
        self._truncate_kwargs = dict(n=n, sorted_by=sorted_by)
        self._aggregate = aggregate

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        return self.child.parameters.copy()

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def sample(self, bqm, **kwargs):
        """Sample from the problem provided by bqm and truncate output.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            **kwargs:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        tkw = self._truncate_kwargs
        if self._aggregate:
            return self.child.sample(bqm, **kwargs).aggregate().truncate(**tkw)
        else:
            return self.child.sample(bqm, **kwargs).truncate(**tkw)
