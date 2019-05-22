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
"""A composite that tracks inputs and outputs."""
from collections import OrderedDict
from copy import deepcopy
from functools import wraps

try:
    from inspect import getfullargspec
except ImportError:
    # python 2.7, we only use .arg so it's ok
    from inspect import getargspec as getfullargspec


from dimod.core.composite import ComposedSampler

__all__ = ['TrackingComposite']


def tracking(f):
    @wraps(f)
    def _tracking(sampler, *args, **kwargs):

        inpt = OrderedDict(zip(getfullargspec(f).args[1:], args))  # skip self
        inpt.update(kwargs)

        # we need to do this before in case they get mutated
        if sampler._copy:
            inpt = deepcopy(inpt)

        sampleset = f(sampler, *args, **kwargs)

        output = sampleset
        if sampler._copy:
            output = deepcopy(output)

        sampler.inputs.append(inpt)
        sampler.outputs.append(output)

        return sampleset

    return _tracking


class TrackingComposite(ComposedSampler):
    """Composite that tracks inputs and outputs for debugging and testing.

    Args:
        child (:obj:`dimod.Sampler`):
            A dimod sampler.

        copy (bool, optional, default=False):
            If True, the inputs/outputs are copied (with :func:`copy.deepcopy`)
            before they are stored. This is useful if the child sampler mutates
            the values.

    Examples:

        >>> sampler = dimod.TrackingComposite(dimod.RandomSampler())
        >>> sampleset = sampler.sample_ising({'a': -1}, {('a', 'b'): 1},
        ...                                  num_reads=5)
        >>> sampler.input
        OrderedDict([('h', {'a': -1}), ('J', {('a', 'b'): 1}), ('num_reads', 5)])
        >>> sampleset == sampler.output
        True

        If we make additional calls to the sampler, the most recent input/output
        are stored in :attr:`.input` and :attr:`.output` respectively. However,
        all are tracked in :attr:`.inputs` and :attr:`.outputs`.

        >>> sampleset = sampler.sample_qubo({('a', 'b'): 1})
        >>> sampler.input
        OrderedDict([('Q', {('a', 'b'): 1})])
        >>> sampler.inputs # doctest: +SKIP
        [OrderedDict([('h', {'a': -1}), ('J', {('a', 'b'): 1}), ('num_reads', 5)]),
         OrderedDict([('Q', {('a', 'b'): 1})])]

        In the case that you want to nest the tracking composite, there are two
        patterns for retrieving the data

        >>> from dimod import ScaleComposite, TrackingComposite, ExactSolver
        ...
        >>> sampler = ScaleComposite(TrackingComposite(ExactSolver()))
        >>> sampler.child.inputs  # empty because we haven't called sample
        []

        >>> intermediate_sampler = TrackingComposite(ExactSolver())
        >>> sampler = ScaleComposite(intermediate_sampler)
        >>> intermediate_sampler.inputs
        []

    """

    children = None

    def __init__(self, child, copy=False):
        self.children = [child]
        self._inputs = []
        self._outputs = []
        self._copy = copy

    @property
    def input(self):
        """The most recent input to any sampling method."""
        try:
            return self.inputs[-1]
        except IndexError:
            pass
        raise ValueError("The sample method has not been called")

    @property
    def inputs(self):
        """All of the inputs to any sampling methods."""
        return self._inputs

    @property
    def output(self):
        """The most recent output of any sampling method."""
        try:
            return self.outputs[-1]
        except IndexError:
            pass
        raise ValueError("The sample method has not been called")

    @property
    def outputs(self):
        """All of the outputs from any sampling methods."""
        return self._outputs

    @property
    def parameters(self):
        return self.child.parameters.copy()

    @property
    def properties(self):
        return {'child_properties': self.child.properties.copy()}

    def clear(self):
        """Clear all the inputs/outputs."""
        # we want to use self.inputs.clear() but it's not in python2
        del self.inputs[:]
        del self.outputs[:]

    @tracking
    def sample(self, bqm, **parameters):
        """Sample from the child sampler and store the given inputs/outputs.

        The binary quadratic model and any parameters are stored in
        :attr:`.inputs`. The returned sample set is stored in :attr:`.outputs`.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            **kwargs:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        return self.child.sample(bqm, **parameters)

    @tracking
    def sample_ising(self, h, J, **parameters):
        """Sample from the child sampler and store the given inputs/outputs.

        The binary quadratic model and any parameters are stored in
        :attr:`.inputs`. The returned sample set is stored in :attr:`.outputs`.

        Args:
            h (dict/list):
                Linear biases of the Ising problem. If a dict, should be of the
                form `{v: bias, ...}` where is a spin-valued variable and `bias`
                is its associated bias. If a list, it is treated as a list of
                biases where the indices are the variable labels.

            J (dict[(variable, variable), bias]):
                Quadratic biases of the Ising problem.

            **kwargs:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        return self.child.sample_ising(h, J, **parameters)

    @tracking
    def sample_qubo(self, Q, **parameters):
        """Sample from the child sampler and store the given inputs/outputs.

        The binary quadratic model and any parameters are stored in
        :attr:`.inputs`. The returned sample set is stored in :attr:`.outputs`.

        Args:
            Q (dict):
                Coefficients of a quadratic unconstrained binary optimization
                (QUBO) problem. Should be a dict of the form `{(u, v): bias, ...}`
                where `u`, `v`, are binary-valued variables and `bias` is their
                associated coefficient.

            **kwargs:
                Parameters for the sampling method, specified by the child
                sampler.

        Returns:
            :obj:`dimod.SampleSet`

        """
        return self.child.sample_qubo(Q, **parameters)
