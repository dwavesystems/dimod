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
This module contains asserts that can be used to test the correctness of dimod samplers,
composites and responses. This is useful for checking that a created sampler correctly fulfills
the dimod API.
"""
import warnings

from collections import Mapping, Sequence, Set


import dimod


def assert_response_correct_energy(response, bqm):
    """Maintained too keep backwards compatibility. Can be removed for any version greater than 0.7.x"""
    warnings.warn("assert_response_correct_energy is deprecated, please use assert_response_energies",
                  DeprecationWarning)
    assert_response_energies(response, bqm)


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
    assert isinstance(sampler.parameters, Mapping), msg

    msg = "instantiated sampler must have a 'properties' property, set to a Mapping"
    assert hasattr(sampler, 'properties'), msg
    assert not callable(sampler.properties), msg
    assert isinstance(sampler.properties, Mapping), msg


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
    assert isinstance(composed_sampler.children, Sequence), msg

    msg = "instantiated composed sampler must have a 'child' property, set to one of sampler.children"
    assert hasattr(composed_sampler, 'child'), msg
    assert isinstance(composed_sampler.child, dimod.Sampler), msg
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
    assert isinstance(sampler.adjacency, Mapping), msg
    for u, neighborhood in sampler.adjacency.items():
        assert isinstance(neighborhood, Set), msg
        for v in neighborhood:
            assert v in sampler.adjacency, msg
            assert u in sampler.adjacency[v], msg

    msg = "instantiated structured sampler must have a 'nodelist' property, set to a list"
    assert hasattr(sampler, 'nodelist'), msg
    assert isinstance(sampler.nodelist, Sequence), msg
    for v in sampler.nodelist:
        assert v in sampler.adjacency, msg

    msg = "instantiated structured sampler must have a 'edge' property, set to a list of 2-lists/tuples"
    assert hasattr(sampler, 'edgelist'), msg
    assert isinstance(sampler.edgelist, Sequence), msg
    for edge in sampler.edgelist:
        assert isinstance(edge, Sequence), msg
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
            Equality of energy is tested by calculating the difference between the `response`'s
            sample energy and that returned by `bqm`'s :meth:`~.BinaryQuadraticModel.energy`,
            rounding to the closest multiple of 10 to the power minus `precision`.

    Raises:
        AssertionError: If any of the samples in the response do not match their associated energy.

    Examples:

        >>> import dimod
        >>> import dimod.testing as dtest
        ...
        >>> sampler = dimod.ExactSolver()
        >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {(0, 1): -1})
        >>> response = sampler.sample(bqm)
        >>> dtest.assert_response_energies(response, bqm)

    """
    assert isinstance(response, dimod.SampleSet), "expected response to be a dimod SampleSet object"

    for sample, energy in response.data(['sample', 'energy']):
        assert isinstance(sample, Mapping), "'for sample in response', each sample should be a Mapping"

        for v, value in sample.items():
            assert v in bqm.linear, 'sample contains a variable not in the given bqm'
            assert value in bqm.vartype.value, 'sample contains a value not of the correct type'

            for v in bqm.linear:
                assert v in sample, "bqm contains a variable not in sample"

        assert round(bqm.energy(sample) - energy, precision) == 0
