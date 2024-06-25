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

import collections.abc as abc

import dimod


def assert_sampler_api(sampler):
    """Assert that an instantiated sampler exposes correct properties and methods.

    Args:
        sampler (:obj:`.Sampler`):
            User-made dimod sampler.

    Raises:
        AssertionError: If the given sampler does not match the sampler API.

    See also:
        :class:`.Sampler` for the abstract base class that defines the sampler API.

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
    """Assert that an instantiated composed sampler exposes correct composite
    properties and methods.

    Args:
        sampler (:obj:`.Composite`):
            User-made dimod composed sampler.

    Raises:
        AssertionError: If the given sampler does not match the composite API.

    See also:
        :class:`.Composite` for the abstract base class that defines the composite API.

        :obj:`~.assert_sampler_api` to assert that the composed sampler matches the sampler API.

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
    """Assert that an instantiated structured sampler exposes correct composite
    properties and methods.

    Args:
        sampler (:obj:`.Structured`):
            User-made dimod structured sampler.

    Raises:
        AssertionError: If the given sampler does not match the structured API.

    See also:
        :class:`.Structured` for the abstract base class that defines the structured API.

        :obj:`~.assert_sampler_api` to assert that the structured sampler matches the sampler API.

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
        response (:obj:`~dimod.SampleSet`):
            Response as returned by a dimod sampler.

        bqm (:obj:`.BinaryQuadraticModel`):
            Binary quadratic model (BQM) used to generate the samples.

        precision (int, optional, default=7):
            Equality of energy is tested by calculating the difference between
            the `response`'s sample energy and that returned by BQM's
            :meth:`~.BinaryQuadraticModel.energy`, rounding to the closest
            multiple of 10 to the power of minus `precision`.

    Raises:
        AssertionError: If any of the samples in the response do not match their
        associated energy.

    See also:
        :func:`.assert_sampleset_energies`

    """
    return assert_sampleset_energies(response, bqm, precision)


def assert_sampleset_energies(sampleset, bqm, precision=7):
    """Assert that each sample in the given sample set has the correct energy.

    Args:
        sampleset (:obj:`~dimod.SampleSet`):
            Sample set as returned by a dimod sampler.

        bqm (:obj:`.BinaryQuadraticModel`/:obj:`.BinaryPolynomial`):
            The binary quadratic model (BQM) or binary polynomial used to generate the samples.

        precision (int, optional, default=7):
            Equality of energy is tested by calculating the difference between
            the `response`'s sample energy and that returned by BQM's
            :meth:`~.BinaryQuadraticModel.energy`, rounding to the closest
            multiple of 10 to the power of minus `precision`.

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
            assert v in bqm.variables, 'sample contains a variable not in the given bqm'
            assert value in bqm.vartype.value, 'sample contains a value not of the correct type'

            for v in bqm.variables:
                assert v in sample, "bqm contains a variable not in sample"

        en = bqm.energy(sample)
        assert round(en - energy, precision) == 0, f"{en} is not almost equal to {energy}"


def assert_sampleset_energies_dqm(sampleset, dqm, precision=7):
    """Assert that each sample in the given sample set has the correct energy.

    Args:
        sampleset (:obj:`~dimod.SampleSet`):
            Sample set as returned by a dimod sampler.

        dqm (:obj:`.DiscreteQuadraticModel`):
            The discrete quadratic model (DQM) used to generate the samples.

        precision (int, optional, default=7):
            Equality of energy is tested by calculating the difference between
            the `response`'s sample energy and that returned by DQM's
            :meth:`~.DiscreteQuadraticModel.energy`, rounding to the closest
            multiple of 10 to the power of minus `precision`.

    Raises:
        AssertionError: If any of the samples in the sample set do not match
        their associated energy.

    """

    assert isinstance(sampleset, dimod.SampleSet), "expected sampleset to be a dimod SampleSet object"

    for sample, energy in sampleset.data(['sample', 'energy']):
        assert isinstance(sample, abc.Mapping), "'for sample in sampleset', each sample should be a Mapping"

        for v, value in sample.items():
            assert v in dqm.variables, 'sample contains a variable not in the given dqm'

            for v in dqm.variables:
                assert v in sample, "dqm contains a variable not in sample"

        assert round(dqm.energy(sample) - energy, precision) == 0
    
    
def assert_sampleset_energies_cqm(sampleset, cqm, precision=7):
    """Assert that each sample in the given sample set has the correct energy.

    Args:
        sampleset (:obj:`~dimod.SampleSet`):
            Sample set as returned by a dimod sampler.

        cqm (:obj:`.ConstrainedQuadraticModel`):
            The constrained quadratic model (CQM) used to generate the samples.

        precision (int, optional, default=7):
            Equality of energy is tested by calculating the difference between
            the `response`'s sample energy and that returned by CQM objective's
            :meth:`~.QuadraticModel.energy`, rounding to the closest
            multiple of 10 to the power of minus `precision`.

    Raises:
        AssertionError: If any of the samples in the sample set do not match
        their associated energy.

    """

    assert isinstance(sampleset, dimod.SampleSet), "expected sampleset to be a dimod SampleSet object"

    for sample, energy in sampleset.data(['sample', 'energy']):
        assert isinstance(sample, abc.Mapping), "'for sample in sampleset', each sample should be a Mapping"

        for v, value in sample.items():
            assert v in cqm.variables, 'sample contains a variable not in the given cqm'

            for v in cqm.variables:
                assert v in sample, "cqm contains a variable not in sample"

        assert round(cqm.objective.energy(sample) - energy, precision) == 0


def assert_bqm_almost_equal(actual, desired, places=7,
                            ignore_zero_interactions=False):
    """Test if two binary quadratic models have almost equal biases.

    Args:
        actual (:obj:`.BinaryQuadraticModel`):
            First binary quadratic model.

        desired (:obj:`.BinaryQuadraticModel`):
            Second binary quadratic model.

        places (int, optional, default=7):
            Bias equality is computed as :code:`round(b0 - b1, places) == 0`.

        ignore_zero_interactions (bool, optional, default=False):
            If true, interactions with 0 bias are ignored.

    """

    # vartype should match
    assert actual.vartype is desired.vartype, "unlike vartype"

    # variables should match
    variables_diff = set(actual.variables) ^ set(desired.variables)
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


def assert_consistent_bqm(bqm):
    """Test whether a BQM is self-consistent.

    This is useful when making new BQM subclasses. Asserts that all of the
    attributes are self-consistent.

    Args:
        bqm: A binary quadratic model.

    """
    # adjacency and linear are self-consistent
    for v in bqm.linear:
        assert v in bqm.adj
    for v in bqm.adj:
        assert v in bqm.linear

    # adjacency and quadratic are self-consistent
    for u, v in bqm.quadratic:
        assert v in bqm.linear
        assert v in bqm.adj
        assert u in bqm.adj[v]

        assert u in bqm.linear
        assert u in bqm.adj
        assert v in bqm.adj[u]

        assert bqm.adj[u][v] == bqm.quadratic[(u, v)]
        assert bqm.adj[u][v] == bqm.quadratic[(v, u)]
        assert bqm.adj[v][u] == bqm.adj[u][v]

    for u, v in bqm.quadratic:
        assert bqm.get_quadratic(u, v) == bqm.quadratic[(u, v)]
        assert bqm.get_quadratic(u, v) == bqm.quadratic[(v, u)]
        assert bqm.get_quadratic(v, u) == bqm.quadratic[(u, v)]
        assert bqm.get_quadratic(v, u) == bqm.quadratic[(v, u)]

    for u in bqm.adj:
        for v in bqm.adj[u]:
            assert (u, v) in bqm.quadratic
            assert (v, u) in bqm.quadratic

    assert len(bqm.quadratic) == bqm.num_interactions
    assert len(bqm.linear) == bqm.num_variables
    assert len(bqm.quadratic) == len(set(bqm.quadratic))
    assert len(bqm.variables) == len(bqm.linear)
    assert (bqm.num_variables, bqm.num_interactions) == bqm.shape
