"""An exact solver that calculates the energy of all possible samples.
"""
from dimod import _PY2
from dimod.template_sampler import TemplateSampler
from dimod.decorators import ising, ising_index_labels
from dimod.responses import SpinResponse
from dimod.utilities import ising_energy

__all__ = ['ExactSolver']

if _PY2:
    range = xrange

    def iteritems(d):
        return d.iteritems()

else:
    def iteritems(d):
        return d.items()


class ExactSolver(TemplateSampler):
    """A simple exact solver, intended for testing and debugging.

    Notes:
        This solver starts to become slow for problems with 18 or more
        variables.

    """

    def __init__(self):
        TemplateSampler.__init__(self)

    @ising(1, 2)
    @ising_index_labels(1, 2)
    def sample_ising(self, h, J):
        """Solves the Ising problem exactly.

        Args:
            h (dict/list): The linear terms in the Ising problem. If a
                dict, should be of the form {v: bias, ...} where v is
                a variable in the Ising problem, and bias is the linear
                bias associated with v. If a list, should be of the form
                [bias, ...] where the indices of the biases are the
                variables in the Ising problem.
            J (dict): A dictionary of the quadratic terms in the Ising
                problem. Should be of the form {(u, v): bias} where u,
                v are variables in the Ising problem and bias is the
                quadratic bias associated with u, v.

        Returns:
            :obj:`SpinResponse`

        Notes:
            Becomes slow for problems with 18 or more variables.

        """

        # it will be convenient to have J in a nested-dict form.
        adjJ = {v: {} for v in h}
        for (u, v), bias in iteritems(J):
            if v not in adjJ[u]:
                adjJ[u][v] = bias
            else:
                adjJ[u][v] += bias

            if u not in adjJ[v]:
                adjJ[v][u] = bias
            else:
                adjJ[v][u] += bias

        # initialize the response
        response = SpinResponse()

        # generate the first sample and add it to the response
        sample = {v: -1 for v in h}
        energy = ising_energy(h, J, sample)
        response.add_sample(sample.copy(), energy)

        # now we iterate, flipping one bit at a time until we have
        # traversed all samples. This is a Gray code.
        # https://en.wikipedia.org/wiki/Gray_code
        for i in range(1, 1 << len(h)):
            v = _ffs(i)

            # flip the bit in the sample
            sample[v] *= -1

            # get the energy difference
            quad_diff = sum(adjJ[v][u] * sample[u] for u in adjJ[v])

            # calculate the new energy as a difference from the old
            energy += 2 * sample[v] * (h[v] + quad_diff)

            response.add_sample(sample.copy(), energy)
        return response


def _ffs(x):
    """Gets the index of the least significant set bit of x."""
    return (x & -x).bit_length() - 1
