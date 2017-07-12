import sys
import random
import math
import itertools
from multiprocessing import Pool

from dimodsampler import DiscreteModelSampler
from dimodsampler.decorators import ising, qubo
from dimodsampler import ising_energy

__all__ = ['SimulatedAnnealingSolver']


if sys.version_info[0] == 2:
    range = xrange


class SimulatedAnnealingSolver(DiscreteModelSampler):

    @ising(1, 2)
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

    @qubo(1)
    def solve_qubo(self, Q, **args):
        h, J, offset = qubo_to_ising(Q)
        spin_response = self.solve_ising(h, J, **args)
        return spin_response.as_binary(offset)

    def solve_structured_qubo(self, Q, **args):
        return self.solve_qubo(Q, **args)


def _ising_simulated_annealing_multiprocessing(args):
    """TODO
    """
    raise NotImplementedError


def ising_simulated_annealing(h, J, beta_range=(.1, 3.33), sweeps=1000):
    """Tries to find the spins that minimize the given Ising problem.

    Args:
        h (dict): A dictionary of the linear biases in the Ising
        problem. Should be of the form {v: bias, ...} for each
        variable v in the Ising problem.
        J (dict): A dictionary of the quadratic biases in the Ising
        problem. Should be a dict of the form {(u, v): bias, ...}
        for each edge (u, v) in the Ising problem. If J[(u, v)] and
        J[(v, u)] exist then the biases are added.
        beta_range (tuple): A 2-tuple defining the beginning and end
        of the beta schedule (beta is the inverse temperature). The
        schedule is applied linearly in beta. Default is (.1, 3.33).
        sweeps (int): The number of sweeps or steps. Default is 1000.

    Returns:
        dict: A sample as a dictionary of spins.
        float: The energy of the returned sample.

    Raises:
        TypeError: If the values in `beta_range` are not numeric.
        TypeError: If `sweeps` is not an int.
        TypeError: If `beta_range` is not a tuple.
        ValueError: If the values in `beta_range` are not positive.
        ValueError: If `beta_range` is not a 2-tuple.
        ValueError: If `sweeps` is not positive.

    https://en.wikipedia.org/wiki/Simulated_annealing

    """

    # input checking, assume h and J are already checked
    if not isinstance(beta_range, (tuple, list)):
        raise TypeError("'beta_range' should be a tuple of length 2")
    if any(not isinstance(b, (int, float)) for b in beta_range):
        raise TypeError("values in 'beta_range' should be numeric")
    if any(b <= 0 for b in beta_range):
        raise ValueError("beta values in 'beta_range' should be positive")
    if len(beta_range) != 2:
        raise ValueError("'beta_range' should be a tuple of length 2")
    if not isinstance(sweeps, int):
        raise TypeError("'sweeps' should be a positive int")
    if sweeps <= 0:
        raise ValueError("'sweeps' should be a positive int")

    # We want the schedule to be linear in beta (inverse temperature)
    beta_init, beta_final = beta_range
    betas = [beta_init + i * (beta_final - beta_init) / (sweeps - 1.) for i in range(sweeps)]

    # set up the adjacency matrix. We can rely on every node in J already being in h
    adj = {n: set() for n in h}
    for n0, n1 in J:
        adj[n0].add(n1)
        adj[n1].add(n0)

    # we will use a vertex coloring the the graph and update the nodes by color. A quick
    # greedy coloring will be sufficient.
    __, colors = greedy_coloring(adj)

    # let's make our initial guess (randomly)
    spins = {v: random.choice((-1, 1)) for v in h}

    for swp in range(sweeps):

        # we want to know the gain in energy for flipping each of the spins
        # we can calculate all of the linear terms simultaniously
        energy_diff_h = {v: -2 * spins[v] * h[v] for v in h}

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
                        ediff += spins[v0] * spins[v1] * J[(v0, v1)]
                    if (v1, v0) in J:
                        ediff += spins[v0] * spins[v1] * J[(v1, v0)]

                energy_diff_J[v0] = -2. * ediff

            # now decide whether to flip spins according to the
            # following scheme:
            #   p ~ Uniform(0, 1)
            #   log(p) < -beta(swp) * (energy_diff)
            for v in nodes:
                logp = math.log(random.uniform(0, 1))
                if logp < -1. * betas[swp] * (energy_diff_h[v] + energy_diff_J[v]):
                    # flip the variable in the spins
                    spins[v] *= -1

    return spins, ising_energy(h, J, spins)


def greedy_coloring(adj):
    """Determines a vertex coloring.

    Args:
        adj (dict): The edge structure of the graph to be colored.
        `adj` should be of the form {node: neighbors, ...} where
        neighbors is a set.

    Returns:
        dict: the coloring {node: color, ...}
        dict: the colors {color: [node, ...], ...}

    Note:
        This is a greedy heuristic, the resulting coloring is not
        necessarily minimal.

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
