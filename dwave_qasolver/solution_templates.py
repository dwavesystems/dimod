import sys
import itertools
import bisect
import math

from dwave_qasolver.decorators import solve_ising_api, solve_qubo_api

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
            idx = self._solutions.index(solution)
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

    @solve_qubo_api(3)
    def add_solution(self, solution, energy=float('nan'), Q={}):
        """Adds a single solution to the response.

        Args:
            solution (dict): A single Discrete Model solution of the
            form {var: b, ...} where `var` is any hashable object and
            b is either 0 or 1.
            energy (float/int, optional): The energy indiced by each
            solution. Default is NaN.
            Q (dict): Dictionary of QUBO coefficients. Takes the form
            {(var0, var1): coeff, ...}. If not provided, the given
            energy is used. If energy is not provided, but Q is then
            the energy is calculated from Q. If both are provided then
            the energy is checked against Q.

        """

        # if no Q provided, just use the inherited method
        if Q is None:
            DiscreteModelResponse.add_solution(self, solution, energy)
            return

        # ok, so we have a Q to play with. So let's calculate the induced energy
        # from Q.
        calculated_energy = 0
        for v0, v1 in Q:
            calculated_energy += solution[v0] * solution[v1] * Q[(v0, v1)]

        # if both Q and energy were provided, let's check that they are equal
        if not math.isnan(energy) and energy != calculated_energy:
            raise ValueError("given energy ({}) and energy induced by Q ({}) do not agree"
                             .format(energy, calculated_energy))

        # finally add the solution
        DiscreteModelResponse.add_solution(self, solution, calculated_energy)

    def add_solutions_from(self, solutions, energies=None, Q=None):
        raise NotImplementedError


class SpinResponse(DiscreteModelResponse):
    def as_bool(self):
        bool_solutions = [{var: (solution[var] + 1) / 2 for var in solution}
                          for solution in self.solutions]

        return BooleanResponse(bool_solutions)

    @solve_ising_api(3, 4)
    def add_solution(self, solution, energy=float('nan'), h={}, J={}):
        if any(spin not in (-1, 1) for spin in solution.values()):
            raise ValueError("solution values must be spin (-1 or 1)")

        # the base case
        if not h and not J:
            DiscreteModelResponse.add_solution(self, solution, energy)
            return

        # if h, J are provided, we can use them to determine/check the energy
        if not h:
            raise TypeError("input 'h' defined but not 'J'")
        if not J:
            raise TypeError("input 'J' defined but not 'h'")

        # now calculate the energy
        energy = 0

        # first the linear biases
        for var in h:
            energy += solution[var] * h[var]

        # then the quadratic baises
        for var0, var1 in J:
            energy += solution[var0] * solution[var1] * J[(var0, var1)]

        # finally add the solution
        DiscreteModelResponse.add_solution(self, solution, energy)

    def add_solutions_from(self, solutions, energies=None, h=None, J=None):
        raise NotImplementedError
