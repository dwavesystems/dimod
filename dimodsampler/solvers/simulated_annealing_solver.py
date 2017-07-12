import sys
import random
import math
import itertools
from multiprocessing import Pool

from dimodsampler import DiscreteModelSolver
from dimodsampler.decorators import ising

__all__ = ['SimulatedAnnealingSolver']


if sys.version_info[0] == 2:
    range = xrange


class SimulatedAnnealingSolver(DiscreteModelSolver):

    @solve_ising_api()
    def solve_ising(self, h, J, samples=10, multiprocessing=False,
                    T_range=(10, .3), sweeps=1000):

        response = SpinResponse()

        if not multiprocessing or samples < 2:
            for __ in range(samples):
                sample, energy = solve_ising_simulated_annealing(h, J, T_range, sweeps)
                response.add_solution(sample, energy)

        else:
            p = Pool(10)
            args = itertools.repeat((h, J, T_range, sweeps), samples)
            for sample, energy in p.map(_solve_ising_sa, args):
                response.add_solution(sample, energy)

        return response

    def solve_structured_ising(self, h, J, **args):
        return self.solve_ising(h, J, **args)

    @solve_qubo_api()
    def solve_qubo(self, Q, **args):
        h, J, offset = qubo_to_ising(Q)
        spin_response = self.solve_ising(h, J, **args)
        return spin_response.as_binary(offset)

    def solve_structured_qubo(self, Q, **args):
        return self.solve_qubo(Q, **args)


def _solve_ising_sa(args):
    """Wrapper to allow solve_ising_simulated_annealing to be used
    with multiprocessing.
    """
    return solve_ising_simulated_annealing(*args)


@ising(0, 1)
def solve_ising_simulated_annealing(h, J, T_range=(10, .3), sweeps=1000,
                                    intermediate_solutions=True):
    """Tries to find the spins that minimize the given Ising problem.

    Args:
        h
        J
        T_range
        sweeps
        intermediate_solutions


    """

    if any(t <= 0 for t in T_range):
        raise ValueError('temperatures must be positive')

    # set up the adjacency matrix
    adj = {n: set() for n in h}
    for n0, n1 in J:
        adj[n0].add(n1)
        adj[n1].add(n0)

    t_init, t_final = T_range

    # ok, first up we want the inverse temperature schedule
    beta_init = 1. / t_init
    beta_final = 1. / t_final
    betas = [beta_init + i * (beta_final - beta_init) / (sweeps - 1) for i in range(sweeps)]

    # we also need a coloring of the graph. We just use a simply greedy heuristic
    __, colors = greedy_coloring(adj)

    # let's make our initial soln guess (randomly)
    # solution = {v: random.choice((-1, 1)) for v in h}
    solution = {0: -1, 1: 1, 2: -1, 3: -1}

    # # finally, before we get started, we want to track the best solution found over
    # # the full anneal
    # best_solution = solution
    # best_energy = ising_energy(h, J, solution)

    for swp in range(sweeps):

        # we want to know the gain in energy for flipping each of the spins in the solution
        # we can calculate all of the linear terms simultaniously
        energy_diff_h = {v: -2 * solution[v] * h[v] for v in h}

        # for each color, do updates
        for color in colors:

            nodes = colors[color]

            # we now want to know the energy change for flipping the spins within
            # the color class
            energy_diff_J = {}
            for v0 in nodes:
                ediff = 0
                for v1 in adj[v0]:
                    if (v0, v1) in J:
                        ediff += solution[v0] * solution[v1] * J[(v0, v1)]
                    if (v1, v0) in J:
                        ediff += solution[v0] * solution[v1] * J[(v1, v0)]

                energy_diff_J[v0] = -2. * ediff

            # now decide whether to flip spins in the solution according to the
            # following scheme:
            #   p ~ Uniform(0, 1)
            #   log(p) < -beta(swp) * (energy_diff)
            for v in nodes:
                logp = math.log(random.uniform(0, 1))
                if logp < -1. * betas[swp] * (energy_diff_h[v] + energy_diff_J[v]):
                    # flip the variable in the solution
                    solution[v] *= -1

    return solution, ising_energy(h, J, solution)


def greedy_coloring(adj):
    """TODO


    Returns:
        dict: the coloring {node: {color1, color2}}
        dict: the colors {color: [node, node]
    """

    # now let's start coloring
    coloring = {}
    colors = {}
    possible_colors = {n: set(range(len(adj))) for n in adj}
    while possible_colors:

        # get the n with the fewest possible colors
        n = min(possible_colors, key=lambda n: len(possible_colors[n]))

        # assign that node the lowest color it can still have
        color = min(possible_colors[n])
        coloring[n] = color
        if color not in colors:
            colors[color] = {n}
        else:
            colors[color].add(n)

        # also remove color from the possible colors for n's neighbors
        for neighbor in adj[n]:
            if neighbor in possible_colors and color in possible_colors[neighbor]:
                possible_colors[neighbor].remove(color)

        # finally remove n from nodes
        del possible_colors[n]

    return coloring, colors
