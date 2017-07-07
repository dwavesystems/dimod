import sys
import itertools
import bisect
import math

from dwave_qasolver import solve_ising_api, solve_qubo_api
from dwave_qasolver import ising_energy, qubo_energy

# Python 2/3 compatibility
if sys.version_info[0] == 2:
    range = xrange
    zip = itertools.izip


class DiscreteModelResponse(object):
    def __init__(self):
        self._solutions = []
        self._energies = []
        self.data = {}

    def add_solution(self, solution, energy):
        """Loads a solution into the response.

        Args:
            solution (dict): TODO
            energy (float/int): TODO

        Notes:
            Solutions are stored in order of energy, lowest first.

        Raises:
            TypeError: If `solution` is not a dict.
            TypeError: If `energy` is not an int or float.

        Examples:
            >>> response = DiscreteModelResponse()
            >>> response.add_solution({0: 0}, 1)
            >>> response.add_solution({0: 1}, -1)
            >>> print(response.solutions())
            [{0: 1}, {0: 0}]
            >>> print(response.energies())
            [-1, 1]

        """

        if not isinstance(solution, dict):
            raise TypeError("expected 'solution' to be a dict")
        if not isinstance(energy, (float, int)):
            raise TypeError("expected 'energy' to be numeric")

        idx = bisect.bisect(self._energies, energy)
        self._solutions.insert(idx, solution)
        self._energies.insert(idx, energy)

    def add_solutions_from(self, solutions, energies):
        """Loads multiple solutions into response.

        Args:
            solutions: An iterable of solutions. Each solution TODO
            energies: An iterable of energies. Each TODO

        Notes:
            Solutions are stored in order of energy, lowest first.

        Raises:
            TypeError: If  and solution in `solutions` is not a dict.
            TypeError: If any energy in `energies` is not an int or float.

        """
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

    @solve_qubo_api(3)
    def add_solution(self, solution, energy=None, Q={}):
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

    @solve_qubo_api(3)
    def add_solutions_from(self, solutions, energies=None, Q={}):
        if Q is None and not energies:
            raise TypeError("Either 'energies' or 'Q' must be provided")

        if Q is not None:
            calculated_energies = (qubo_energy(Q, soln) for soln in solutions)

        raise NotImplementedError

    def as_spins(self, offset):
        raise NotImplementedError


class SpinResponse(DiscreteModelResponse):

    @solve_ising_api(3, 4)
    def add_solution(self, solution, energy=None, h={}, J={}):
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

    def as_binary(self, offset, copy=True):

        b_response = BinaryResponse()

        # create iterators over the stored data
        binary_solutions = ({v: (solution[v] + 1) / 2 for v in solution}
                            for solution in self.solutions_iter())
        binary_energies = (energy + offset for energy in self.energies_iter())

        b_response.add_solutions_from(binary_solutions, binary_energies)

        b_response = self.data

        if copy:
            return b_response
            return

        self = b_response
