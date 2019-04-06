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
try:
    import collections.abc as abc
except ImportError:
    import collections as abc

import dimod


def assert_sampler_api(sampler):
    """Assert that an instantiated sampler has the correct properties and methods exposed.

    Args:
        sampler (:obj:`.Sampler`):
            A user-made dimod sampler.

    Raises:
        AssertionError: If the given sampler does not match the sampler API.

    See also:
        :class:`.Sampler` The abstract base class that defines the sampler API.

    """

    # abstract base class

    assert isinstance(sampler, dimod.Sampler), "instantiated sampler must be a dimod Sampler object"

    # sample methods

    assert hasattr(sampler, 'sample'), "instantiated sampler must have a 'sample' method"
    assert callable(sampler.sample), "instantiated sampler must have a 'sample' method"

    assert hasattr(sampler, 'sample_ising'), "instantiated sampler must have a 'sample_ising' method"
    assert callable(sampler.sample_ising), "instantiated sampler must have a 'sample_ising' method"

    assert hasattr(sampler, 'sample_qubo'), "instantiated sampler must have a 'sample_qubo' method"
    assert callable(sampler.sample_qubo), "instantiated sampler must have a 'sample_qubo' method"

    # properties

    msg = "instantiated sampler must have a 'parameters' property, set to a Mapping"
    assert hasattr(sampler, 'parameters'), msg
    assert not callable(sampler.parameters), msg
    assert isinstance(sampler.parameters, abc.Mapping), msg

    msg = "instantiated sampler must have a 'properties' property, set to a Mapping"
    assert hasattr(sampler, 'properties'), msg
    assert not callable(sampler.properties), msg
    assert isinstance(sampler.properties, abc.Mapping), msg


def assert_composite_api(composed_sampler):
    """Assert that an instantiated composed sampler has the correct composite properties and methods
    exposed.

    Args:
        sampler (:obj:`.Composite`):
            A user-made dimod composed sampler.

    Raises:
        AssertionError: If the given sampler does not match the composite API.

    See also:
        :class:`.Composite` The abstract base class that defines the composite API.

        :obj:`~.assert_sampler_api` Asserts that the composed sampler matches the sampler API.

    """

    assert isinstance(composed_sampler, dimod.Composite)

    # properties

    msg = "instantiated composed sampler must have a 'children' property, set to a list (or Sequence)"
    assert hasattr(composed_sampler, 'children'), msg
    assert isinstance(composed_sampler.children, abc.Sequence), msg

    msg = "instantiated composed sampler must have a 'child' property, set to one of sampler.children"
    assert hasattr(composed_sampler, 'child'), msg
    assert composed_sampler.child in composed_sampler.children, msg


def assert_structured_api(sampler):
    """Assert that an instantiated structured sampler has the correct composite properties and methods
    exposed.

    Args:
        sampler (:obj:`.Structured`):
            A user-made dimod structured sampler.

    Raises:
        AssertionError: If the given sampler does not match the structured API.

    See also:
        :class:`.Structured` The abstract base class that defines the structured API.

        :obj:`~.assert_sampler_api` Asserts that the structured sampler matches the sampler API.

    """

    assert isinstance(sampler, dimod.Structured), "must be a Structured sampler"

    # properties

    msg = ("instantiated structured sampler must have an 'adjacency' property formatted as a dict "
           "where the keys are the nodes and the values are sets of all node adjacency to the key")
    assert hasattr(sampler, 'adjacency'), msg
    assert isinstance(sampler.adjacency, abc.Mapping), msg
    for u, neighborhood in sampler.adjacency.items():
        assert isinstance(neighborhood, abc.Set), msg
        for v in neighborhood:
            assert v in sampler.adjacency, msg
            assert u in sampler.adjacency[v], msg

    msg = "instantiated structured sampler must have a 'nodelist' property, set to a list"
    assert hasattr(sampler, 'nodelist'), msg
    assert isinstance(sampler.nodelist, abc.Sequence), msg
    for v in sampler.nodelist:
        assert v in sampler.adjacency, msg

    msg = "instantiated structured sampler must have a 'edge' property, set to a list of 2-lists/tuples"
    assert hasattr(sampler, 'edgelist'), msg
    assert isinstance(sampler.edgelist, abc.Sequence), msg
    for edge in sampler.edgelist:
        assert isinstance(edge, abc.Sequence), msg
        assert len(edge) == 2, msg

        u, v = edge
        assert v in sampler.adjacency, msg
        assert u in sampler.adjacency, msg
        assert u != v, msg


def assert_response_energies(response, bqm, precision=7):
    """Assert that each sample in the given response has the correct energy.

    Args:
        response (:obj:`.SampleSet`):
            The response as returned by a dimod sampler.

        bqm (:obj:`.BinaryQuadraticModel`):
            The binary quadratic model used to generate the samples.

        precision (int, optional, default=7):
            Equality of energy is tested by calculating the difference between
            the `response`'s sample energy and that returned by `bqm`'s
            :meth:`~.BinaryQuadraticModel.energy`, rounding to the closest
            multiple of 10 to the power minus `precision`.

    Raises:
        AssertionError: If any of the samples in the response do not match their
        associated energy.

    See also:
        :func:`.assert_sampleset_energies`

    """
    return assert_sampleset_energies(response, bqm, precision)


def assert_sampleset_energies(sampleset, bqm, precision=7):
    """Assert that each sample in the given sampleset has the correct energy.

    Args:
        sampleset (:obj:`.SampleSet`):
            The sample set as returned by a dimod sampler.

        bqm (:obj:`.BinaryQuadraticModel`):
            The binary quadratic model used to generate the samples.

        precision (int, optional, default=7):
            Equality of energy is tested by calculating the difference between
            the `response`'s sample energy and that returned by `bqm`'s
            :meth:`~.BinaryQuadraticModel.energy`, rounding to the closest
            multiple of 10 to the power minus `precision`.

    Raises:
        AssertionError: If any of the samples in the sample set do not match
        their associated energy.

    Examples:

        >>> import dimod.testing
        ...
        >>> sampler = dimod.ExactSolver()
        >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {(0, 1): -1})
        >>> sampleset = sampler.sample(bqm)
        >>> dimod.testing.assert_response_energies(sampleset, bqm)

    """
    assert isinstance(sampleset, dimod.SampleSet), "expected sampleset to be a dimod SampleSet object"

    for sample, energy in sampleset.data(['sample', 'energy']):
        assert isinstance(sample, abc.Mapping), "'for sample in sampleset', each sample should be a Mapping"

        for v, value in sample.items():
            assert v in bqm.linear, 'sample contains a variable not in the given bqm'
            assert value in bqm.vartype.value, 'sample contains a value not of the correct type'

            for v in bqm.linear:
                assert v in sample, "bqm contains a variable not in sample"

        assert round(bqm.energy(sample) - energy, precision) == 0


def assert_bqm_almost_equal(actual, desired, places=7,
                            ignore_zero_interactions=False):
    """Test if two bqm have almost equal biases.

    Args:
        actual (:obj:`.BinaryQuadraticModel`)

        desired (:obj:`.BinaryQuadraticModel`)

        places (int, optional, default=7):
            Bias equality is computed as :code:`round(b0 - b1, places) == 0`

        ignore_zero_interactions (bool, optional, default=False):
            If true, interactions with 0 bias are ignored.

    """
    assert isinstance(actual, dimod.BinaryQuadraticModel), "not a binary quadratic model"
    assert isinstance(desired, dimod.BinaryQuadraticModel), "not a binary quadratic model"

    # vartype should match
    assert actual.vartype is desired.vartype, "unlike vartype"

    # variables should match
    variables_diff = set(actual).symmetric_difference(desired)
    if variables_diff:
        v = variables_diff.pop()
        msg = "{!r} is not a shared variable".format(v)
        raise AssertionError(msg)

    # offset
    if round(actual.offset - desired.offset, places):
        msg = 'offsets {} != {}'.format(desired.offset, actual.offset)
        raise AssertionError(msg)

    # linear biases - we already checked variables
    for v, bias in desired.linear.items():
        if round(bias - actual.linear[v], places):
            msg = 'linear bias associated with {!r} does not match, {!r} != {!r}'
            raise AssertionError(msg.format(v, bias, actual.linear[v]))

    default = 0 if ignore_zero_interactions else None

    for inter, bias in actual.quadratic.items():
        other_bias = desired.quadratic.get(inter, default)
        if other_bias is None:
            raise AssertionError('{!r} is not a shared interaction'.format(inter))
        if round(bias - other_bias, places):
            msg = 'quadratic bias associated with {!r} does not match, {!r} != {!r}'
            raise AssertionError(msg.format(inter, bias, other_bias))
    for inter, bias in desired.quadratic.items():
        other_bias = actual.quadratic.get(inter, default)
        if other_bias is None:
            raise AssertionError('{!r} is not a shared interaction'.format(inter))
        if round(bias - other_bias, places):
            msg = 'quadratic bias associated with {!r} does not match, {!r} != {!r}'
            raise AssertionError(msg.format(inter, bias, other_bias))
