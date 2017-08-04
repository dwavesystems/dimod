import sys
import itertools

from dimod import DiscreteModelSampler, qubo_to_ising
from dimod.decorators import ising, qubo, ising_index_labels
from dimod import ising_energy, qubo_energy, SpinResponse, BinaryResponse

__all__ = ['ExactSolver']

if sys.version_info[0] == 2:
    range = xrange
    iteritems = lambda d: d.iteritems()
else:
    iteritems = lambda d: d.items()


class ExactSolver(DiscreteModelSampler):
    """The simplest possible brute-force solver.

    Note that this starts to become slow for problems with 18 or more
    variables. This solver is intended for testing, not for solving
    anything more than toy problems.

    """

    @qubo(1)
    def sample_qubo(self, Q):
        """TODO"""
        h, J, offset = qubo_to_ising(Q)
        spin_response = self.sample_ising(h, J)
        return spin_response.as_binary(offset)

    def sample_structured_qubo(self, Q):
        """TODO"""
        return self.sample_qubo(Q)

    def sample_structured_ising(self, h, J):
        """TODO"""
        return self.sample_ising(h, J)

    @ising(1, 2)
    @ising_index_labels(1, 2)
    def sample_ising(self, h, J):

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

        response = SpinResponse()
        sample = {v: -1 for v in h}
        energy = ising_energy(h, J, sample)
        response.add_sample(sample, energy)

        for i in range(1, 1 << len(h)):
            v = ffs(i)

            # flip the bit in the sample
            sample[v] *= -1

            linear_diff = 2 * sample[v] * h[v]
            quad_diff = 2 * sum(adjJ[v][u] * sample[u] * sample[v] for u in adjJ[v])

            energy += linear_diff + quad_diff

            response.add_sample(sample, energy)
        return response


def ffs(x):
    return (x & -x).bit_length() - 1
