"""These are generic tests that can be applied to any sampler, this should
not be run as a unittest.

Example:
    >>> class TestMySampler(unittest.TestCase, TestSamplerAPI):
    ...     def __init__(self):
    ...         self.sampler = MySampler()

    This will run all of the tests in TestSamplerAPI

"""


class TestSamplerAPI:
    """Provides a series of generic API tests that all samplers should pass.
    """

    def test_single_node_qubo(self):
        """Create a QUBO that has only a single variable."""

        sampler = self.sampler

        Q = {(0, 0): 1}
        variables = [0]

        response = sampler.sample_qubo(Q)

        self.check_response_form(response, variables)

    def test_single_node_ising(self):
        sampler = self.sampler

        h = {0: -1}
        J = {}
        variables = [0]

        response = sampler.sample_ising(h, J)
        self.check_response_form(response, variables)

    def test_single_edge_qubo(self):
        sampler = self.sampler

        Q = {(0, 0): 0, (1, 1): 0, (0, 1): 1}
        variables = [0, 1]

        response = sampler.sample_qubo(Q)
        self.check_response_form(response, variables)

    def test_single_edge_ising(self):
        sampler = self.sampler

        h = {0: 0, 1: 1}
        J = {(0, 1): 1}
        variables = [0, 1]

        response = sampler.sample_ising(h, J)
        self.check_response_form(response, variables)

    def check_response_form(self, response, variables):
        for var in variables:
            for soln in response:
                self.assertIn(var, soln)
