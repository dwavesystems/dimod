import unittest
import itertools

import dimod

#
# docstring tests
# test that the structured sampler as described in the docstring works appropriately. This will also
# be caught in doctests but here we can test it more generally
#


class TwoVariables(dimod.Sampler, dimod.Structured):
    def __init__(self):
        dimod.Sampler.__init__(self)
        dimod.Structured.__init__(self, [0, 1], [(0, 1)])

    @dimod.decorators.bqm_structured
    def sample(self, bqm):
        # All bqm's passed in will be a subgraph of the sampler's structure
        variable_list = list(bqm.linear)
        response = dimod.Response(bqm.vartype)
        for values in itertools.product(bqm.vartype.value, repeat=len(bqm)):
            sample = dict(zip(variable_list, values))
            energy = bqm.energy(sample)
            response.add_sample(sample, energy)
        return response


class TestTwoVariables(unittest.TestCase, dimod.test.SamplerAPITest):
    def setUp(self):
        self.sampler = TwoVariables()
        self.sampler_factory = TwoVariables

    def test_acceptable_bqms(self):
        sampler = TwoVariables()

        response = sampler.sample_ising({0: -1}, {})

        self.assertEqual(len(response), 2)
