import unittest
import itertools

import dimod

#
# docstring tests
# test that the structured sampler as described in the docstring works appropriately. This will also
# be caught in doctests but here we can test it more generally
#


class TwoVariablesSampler(dimod.Sampler, dimod.Structured):
    def __init__(self):
        dimod.Sampler.__init__(self)
        dimod.Structured.__init__(self, [0, 1], [(0, 1)])

    @dimod.decorators.bqm_structured
    def sample(self, bqm):
        # All bqm's passed in will be a subgraph of the sampler's structure
        variable_list = list(bqm.linear)
        samples = []
        energies = []
        for values in itertools.product(bqm.vartype.value, repeat=len(bqm)):
            sample = dict(zip(variable_list, values))
            samples.append(sample)
            energies.append(bqm.energy(sample))

        response = dimod.Response.from_dicts(samples, {'energy': energies}, vartype=bqm.vartype)

        return response


class TestTwoVariables(unittest.TestCase, dimod.test.SamplerAPITestCaseMixin):
    def setUp(self):
        self.sampler = TwoVariablesSampler()
        self.sampler_factory = TwoVariablesSampler

    def test_acceptable_bqms(self):
        sampler = TwoVariablesSampler()

        response = sampler.sample_ising({0: -1}, {})

        self.assertEqual(len(response), 2)

    def test_example(self):
        sampler = TwoVariablesSampler()

        self.assertEqual(sampler.structure.nodelist, [0, 1])
        self.assertEqual(sampler.structure.edgelist, [(0, 1)])
        self.assertEqual(sampler.structure.adjacency, {0: {1}, 1: {0}})
