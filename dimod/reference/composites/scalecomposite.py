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
"""
A composite that scales problem variables as directed. If a scaling value is
not specified, calculates it based on quadratic and bias ranges.

"""

from dimod.core.composite import ComposedSampler
from dimod.decorators import nonblocking_sample_method
from dimod.sampleset import SampleSet

__all__ = ['ScaleComposite']


class ScaleComposite(ComposedSampler):
    """Composite that scales variables of a problem.

    Scales the variables of a binary quadratic model (BQM) and modifies linear
    and quadratic terms accordingly.

    Args:
       sampler (:obj:`dimod.Sampler`):
            A dimod sampler.

    Examples:
       This example uses :class:`.ScaleComposite` to instantiate a
       composed sampler that submits a simple Ising problem to a sampler.
       The composed sampler scales linear biases, quadratic biases, and
       offset as indicated by options.

       >>> h = {'a': -4.0, 'b': -4.0}
       >>> J = {('a', 'b'): 3.2}
       >>> sampler = dimod.ScaleComposite(dimod.ExactSolver())
       >>> response = sampler.sample_ising(h, J, scalar=0.5,
       ...                ignored_interactions=[('a','b')])

    """

    def __init__(self, child_sampler):
        self._children = [child_sampler]

    @property
    def children(self):
        return self._children

    @property
    def parameters(self):
        param = self.child.parameters.copy()
        param.update({'scalar': [],
                      'bias_range': [],
                      'quadratic_range': [],
                      'ignored_variables': [],
                      'ignored_interactions': [],
                      'ignore_offset': []})
        return param

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    @nonblocking_sample_method
    def sample(self, bqm, scalar=None, bias_range=1, quadratic_range=None,
               ignored_variables=None, ignored_interactions=None,
               ignore_offset=False, **parameters):
        """ Scale and sample from the provided binary quadratic model.

        if scalar is not given, problem is scaled based on bias and quadratic
        ranges. See :meth:`.BinaryQuadraticModel.scale` and
        :meth:`.BinaryQuadraticModel.normalize`

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            scalar (number):
                Value by which to scale the energy range of the binary
                quadratic model. Overrides `bias_range` and `quadratic_range`.

            bias_range (number/pair, default=1):
                Value/range by which to normalize the all the biases, or if
                `quadratic_range` is provided, just the linear biases.
                Overridden by `scalar`.

            quadratic_range (number/pair):
                Value/range by which to normalize the quadratic biases.
                Overridden by `scalar`.

            ignored_variables (iterable, optional):
                Biases associated with these variables are not scaled.

            ignored_interactions (iterable[tuple], optional):
                As an iterable of 2-tuples. Biases associated with these
                interactions are not scaled.

            ignore_offset (bool, default=False):
                If True, the offset is not scaled.

            **parameters:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        original_bqm = bqm
        bqm = bqm.copy()  # we're going to be scaling

        if scalar is not None:
            bqm.scale(scalar,
                      ignored_variables=ignored_variables,
                      ignored_interactions=ignored_interactions,
                      ignore_offset=ignore_offset)
        else:
            scalar = bqm.normalize(bias_range, quadratic_range,
                                   ignored_variables=ignored_variables,
                                   ignored_interactions=ignored_interactions,
                                   ignore_offset=ignore_offset)

        if scalar == 0:
            raise ValueError('scalar must be non-zero')

        sampleset = self.child.sample(bqm, **parameters)

        yield sampleset  # so that SampleSet.done() works

        if not (ignored_variables or ignored_interactions or ignore_offset):
            # we just need to scale back and don't need to worry about
            # the stuff we ignored
            sampleset.record.energy *= 1 / scalar
        else:
            sampleset.record.energy = original_bqm.energies(sampleset)

        sampleset.info.update(scalar=scalar)

        yield sampleset
