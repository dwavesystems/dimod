import sys
import itertools
from multiprocessing import Pool

from dimod import DiscreteModelSampler
from dimod.decorators import ising, qubo
from dimod import ising_energy, qubo_energy, SpinResponse, BinaryResponse

__all__ = ['ExactSolver']

if sys.version_info[0] == 2:
    range = xrange


class ExactSolver(DiscreteModelSampler):

    @qubo(1)
    def sample_qubo(self, Q):
        """TODO"""
        variables = set().union(*Q)
        response = BinaryResponse()
        for ones in powerset(variables):
            sample = {v: v in ones and 1 or 0 for v in variables}
            energy = qubo_energy(Q, sample)
            response.add_sample(sample, energy)
        return response

    def sample_structured_qubo(self, Q):
        """TODO"""
        return self.sample_qubo(Q)

    @ising(1, 2)
    def sample_ising(self, h, J):
        """TODO"""
        response = SpinResponse()
        for ones in powerset(h):
            sample = {v: v in ones and 1 or -1 for v in h}
            energy = ising_energy(h, J, sample)
            response.add_sample(sample, energy)
        return response

    def sample_structured_ising(self, h, J):
        """TODO"""
        return self.sample_ising(h, J)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))
