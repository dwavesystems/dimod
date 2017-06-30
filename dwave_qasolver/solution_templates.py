import sys
import itertools
import bisect

from dwave_qasolver.decorators import solve_ising_api

# Python 2/3 compatibility
if sys.version_info[0] == 2:
    range = xrange
    zip = itertools.izip


class DiscreteModelResponse(object):
    def __init__(self):
        self._solutions = []
        self._energies = []
        self.data = {}

    def add_solution(self, solution, energy=float('nan')):
        idx = bisect.bisect(self._energies, energy)
        self._solutions.insert(idx, solution)
        self._energies.insert(idx, energy)

    def add_solutions_from(self, solutions, energies=None):
        if energies is None:
            energies = itertools.repeat(float('nan'))

        for soln, en in zip(solutions, energies):
            self.add_solution(soln, en)

    def __iter__(self):
        return iter(self._solutions)

    def __str__(self):
        return 'solutions: {}\nenergies:  {}'.format(self._solutions, self._energies)

    def solutions(self):
        return self._solutions

    def solutions_iter(self):
        return iter(self._solutions)

    def energies(self):
        return self._energies

    def energies_iter(self):
        return iter(self._energies)

    def items(self):
        return list(self.items_iter())

    def items_iter(self):
        for soln, en in zip(self.solutions_iter(), self.energies_iter()):
            yield soln, en

    def __getitem__(self, solution):
        try:
            idx = self_solutions.index(solution)
        except ValueError as e:
            raise KeyError(e.message)

        return self._energies[idx]

    def __len__(self):
        return self._solutions.__len__()

    def relabel_variables(self, mapping, copy=True):

        new_response = DiscreteModelResponse()

        for soln, en in self.items_iter():
            raise NotImplementedError
            new_response.add_solution(soln, en)

        new_response.data = self.data

        if copy:
            return new_response

        self = new_response
        return


class BinaryResponse(DiscreteModelResponse):
    def as_spins(self):
        spin_solutions = [{var: 2 * solution[var] - 1 for var in solution}
                          for solution in self.solutions]

        return SpinResponse(spin_solutions)

    def add_solution(self, solution, energy=float('nan'), Q=None):
        raise NotImplementedError

    def add_solutions_from(self, solutions, energies=None, Q=None):
        raise NotImplementedError


class SpinResponse(DiscreteModelResponse):
    def as_bool(self):
        bool_solutions = [{var: (solution[var] + 1) / 2 for var in solution}
                          for solution in self.solutions]

        return BooleanResponse(bool_solutions)

    def add_solution(self, solution, energy=float('nan'), h=None, J=None):
        if any(spin not in (-1, 1) for spin in solution.values()):
            raise ValueError("solution values must be spin (-1 or 1)")

        # the base case
        if h is None and J is None:
            DiscreteModelResponse.add_solution(self, solution, energy)
            return

        # if h, J are provided, we can use them to determine/check the energy
        if h is None:
            raise TypeError("input 'h' defined but not 'J'")
        if J is None:
            raise TypeError("input 'J' defined but not 'h'")

        @solve_ising_api()
        def check_hJ(__, h, J):
            pass

        check_hJ(None, h, J)

    def add_solutions_from(self, solutions, energies=None, h=None, J=None):
        raise NotImplementedError
