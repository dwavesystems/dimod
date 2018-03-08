"""todo"""
from collections import Mapping

import dimod


# def assert_response_correct_energy(response, bqm):
#     """todo"""
#     for sample in response:
#         assert isinstance(sample, Mapping), "'for sample in response', each sample should be a Mapping"

#         for v, value in sample.items():
#             assert v in bqm.linear, 'sample contains a variable not in the given bqm'
#             assert value in bqm.vartype.value, 'sample contains a value not of the correct type'

#             for v in bqm.linear:
#                 assert v in sample, "bqm contains a variable not in sample"


def assert_sampler_api(sampler):
    """Assert that an instantiated sampler has the correct properties and methods exposed.
    """

    # abstract base class

    assert isinstance(sampler, dimod.Sampler), "must be a dimod Sampler."

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
    """todo"""

    assert isinstance(composed_sampler, dimod.Composite)

    # todo: check properties (including correctness of mixins)


def assert_structured_api(sampler):
    """todo"""

    assert isinstance(sampler, dimod.Structured), "must be a Structured sampler"

    # todo: check properties (including correctness of mixins)
