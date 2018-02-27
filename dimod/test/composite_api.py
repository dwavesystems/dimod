from collections import Mapping

import dimod

__all__ = ['CompositeAPITestCaseMixin']


class CompositeAPITestCaseMixin(object):
    def test_instantiation(self):
        for sampler_factory in self.sampler_factories:
            sampler = self.composite_factory(sampler_factory())

            # check that is has all of the expected attritubes
            self.assertTrue(hasattr(sampler, 'sample'),
                            "sampler must have a 'sample' method")
            self.assertTrue(callable(sampler.sample),
                            "sampler must have a 'sample' method")

            self.assertTrue(hasattr(sampler, 'sample_ising'),
                            "sampler must have a 'sample_ising' method")
            self.assertTrue(callable(sampler.sample_ising),
                            "sampler must have a 'sample_ising' method")

            self.assertTrue(hasattr(sampler, 'sample_qubo'),
                            "sampler must have a 'sample_qubo' method")
            self.assertTrue(callable(sampler.sample_qubo),
                            "sampler must have a 'sample_qubo' method")

            # properties

            self.assertTrue(hasattr(sampler, 'parameters'),
                            "sampler must have a 'parameters' property")
            self.assertFalse(callable(sampler.parameters),
                             "sampler must have a 'parameters' property")
            self.assertIsInstance(sampler.parameters, dict)

            self.assertTrue(hasattr(sampler, 'properties'),
                            "sampler must have a 'properties' property")
            self.assertFalse(callable(sampler.properties),
                             "sampler must have a 'properties' property")
            self.assertIsInstance(sampler.properties, dict)


    # def test_sample_response_form(self):
    #     bqm = dimod.BinaryQuadraticModel({'b': 2}, {('a', 'b'): -1.}, 1.0, dimod.SPIN)

    #     for sampler_factory in self.sampler_factories:
    #         sampler = self.composite_factory(sampler_factory())

    #         try:
    #             response = sampler.sample(bqm)
    #         except dimod.exceptions.BinaryQuadraticModelValueError:
    #             return  # the sampler has responded that it cannot handle the give bqm

    #         self.assertIs(response.vartype, bqm.vartype, "response's vartype does not match the bqm's vartype")

    #         for sample in response:
    #             self.assertIsInstance(sample, Mapping, "'for sample in response', each sample should be a dict")
    #             for v, value in sample.items():
    #                 self.assertIn(v, bqm.linear, 'sample contains a variable not in the given bqm')
    #                 self.assertIn(value, bqm.vartype.value, 'sample contains a value not of the correct type')

    #                 for v in bqm.linear:
    #                     self.assertIn(v, sample)
