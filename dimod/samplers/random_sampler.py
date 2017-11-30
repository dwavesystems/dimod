"""
A random sampler that can be used for unit testing and debugging.
"""
import random

from dimod import _PY2
from dimod.template_sampler import TemplateSampler
from dimod.decorators import ising, qubo, ising_index_labels
from dimod.responses import BinaryResponse

__all__ = ['RandomSampler']

if _PY2:
    range = xrange

    def iteritems(d):
        return d.iteritems()

else:
    def iteritems(d):
        return d.items()


class RandomSampler(TemplateSampler):
    """Gives random samples.

    Note that this sampler is intended for testing.

    """

    def __init__(self):
        TemplateSampler.__init__(self)

    @qubo(1)
    def sample_qubo(self, Q, num_samples=10):
        """Gives random samples.

        Args:
            Q (dict): Q dict of the QUBO biases. Should be a dict of the
                form {(u, v): bias, ...} where u, v are variables in the
                QUBO and bias is the quadratic bias associated with u and
                v. If u == v, then the bias is the linear bias associated
                with v.
            num_samples (int, optional): The number of random samples to
                take. Default 10.

        Returns:
            :obj:`BinaryResponse`

        Notes:
            For each variable in each sample, the value is chosen by a coin
            flip.

        """
        variables = set().union(*Q)
        samples = [{v: random.choice((0, 1)) for v in variables} for __ in range(num_samples)]
        response = BinaryResponse()
        response.add_samples_from(samples, Q=Q)
        return response
