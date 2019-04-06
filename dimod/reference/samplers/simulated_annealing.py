# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# =============================================================================
"""
A reference implementation of a simulated annealing sampler.

:obj:`neal.sampler.SimulatedAnnealingSampler`
is a more performant implementation of simulated annealing you can use for
solving problems.
"""
import random
import math

from six import itervalues

from dimod.core.sampler import Sampler
from dimod.sampleset import SampleSet
from dimod.utilities import ising_energy
from dimod.vartypes import Vartype

__all__ = ['SimulatedAnnealingSampler']


class SimulatedAnnealingSampler(Sampler):
    """A simple simulated annealing sampler for testing and debugging code.

    Examples:
        This example solves a two-variable Ising model.

        >>> h = {'a': -0.5, 'b': 1.0}
        >>> J = {('a', 'b'): -1.5}
        >>> sampleset = dimod.SimulatedAnnealingSampler().sample_ising(h, J)

    """
    properties = None

    parameters = None
    """dict: Keyword arguments accepted by the sampling methods.

    Contents are exactly `{'beta_range': [], num_reads': [], 'num_sweeps': []}`
    """

    def __init__(self):
        self.parameters = {'num_reads': [],
                           'beta_range': [],
                           'num_sweeps': []}
        self.properties = {}

    def sample(self, bqm, beta_range=None, num_reads=10, num_sweeps=1000):
        """Sample from low-energy spin states using simulated annealing.

        Args:
            bqm (:obj:`.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            beta_range (tuple, optional): Beginning and end of the beta schedule
                (beta is the inverse temperature) as a 2-tuple. The schedule is applied
                linearly in beta. Default is chosen based on the total bias associated
                with each node.

            num_reads (int, optional, default=10):
                Number of reads. Each sample is the result of a single run of
                the simulated annealing algorithm.

            num_sweeps (int, optional, default=1000):
                Number of sweeps or steps.

        Returns:
            :obj:`.SampleSet`

        Note:
            This is a reference implementation, not optimized for speed
            and therefore not an appropriate sampler for benchmarking.

        """

        # input checking
        # h, J are handled by the @ising decorator
        # beta_range, sweeps are handled by ising_simulated_annealing
        if not isinstance(num_reads, int):
            raise TypeError("'samples' should be a positive integer")
        if num_reads < 1:
            raise ValueError("'samples' should be a positive integer")

        h, J, offset = bqm.to_ising()

        # run the simulated annealing algorithm
        samples = []
        energies = []
        for __ in range(num_reads):
            sample, energy = ising_simulated_annealing(h, J, beta_range, num_sweeps)
            samples.append(sample)
            energies.append(energy)

        response = SampleSet.from_samples(samples, Vartype.SPIN, energies)
        response.change_vartype(bqm.vartype, offset, inplace=True)
        return response


def ising_simulated_annealing(h, J, beta_range=None, num_sweeps=1000):
    """Tries to find the spins that minimize the given Ising problem.

    Args:
        h (dict): A dictionary of the linear biases in the Ising
            problem. Should be of the form {v: bias, ...} for each
            variable v in the Ising problem.
        J (dict): A dictionary of the quadratic biases in the Ising
            problem. Should be a dict of the form {(u, v): bias, ...}
            for each edge (u, v) in the Ising problem. If J[(u, v)] and
            J[(v, u)] exist then the biases are added.
        beta_range (tuple, optional): A 2-tuple defining the
            beginning and end of the beta schedule (beta is the
            inverse temperature). The schedule is applied linearly
            in beta. Default is chosen based on the total bias associated
            with each node.
        num_sweeps (int, optional): The number of sweeps or steps.
            Default is 1000.

    Returns:
        dict: A sample as a dictionary of spins.
        float: The energy of the returned sample.

    Raises:
        TypeError: If the values in `beta_range` are not numeric.
        TypeError: If `num_sweeps` is not an int.
        TypeError: If `beta_range` is not a tuple.
        ValueError: If the values in `beta_range` are not positive.
        ValueError: If `beta_range` is not a 2-tuple.
        ValueError: If `num_sweeps` is not positive.

    https://en.wikipedia.org/wiki/Simulated_annealing

    """

    if beta_range is None:
        beta_init = .1

        sigmas = {v: abs(h[v]) for v in h}
        for u, v in J:
            sigmas[u] += abs(J[(u, v)])
            sigmas[v] += abs(J[(u, v)])

        if sigmas:
            beta_final = 2. * max(itervalues(sigmas))
        else:
            beta_final = 0.0

    else:
        if not isinstance(beta_range, (tuple, list)):
            raise TypeError("'beta_range' should be a tuple of length 2")
        if any(not isinstance(b, (int, float)) for b in beta_range):
            raise TypeError("values in 'beta_range' should be numeric")
        if any(b <= 0 for b in beta_range):
            raise ValueError("beta values in 'beta_range' should be positive")
        if len(beta_range) != 2:
            raise ValueError("'beta_range' should be a tuple of length 2")
        beta_init, beta_final = beta_range
    if not isinstance(num_sweeps, int):
        raise TypeError("'sweeps' should be a positive int")
    if num_sweeps <= 0:
        raise ValueError("'sweeps' should be a positive int")

    # We want the schedule to be linear in beta (inverse temperature)
    betas = [beta_init + i * (beta_final - beta_init) / (num_sweeps - 1.)
             for i in range(num_sweeps)]

    # set up the adjacency matrix. We can rely on every node in J already being in h
    adj = {n: set() for n in h}
    for n0, n1 in J:
        adj[n0].add(n1)
        adj[n1].add(n0)

    # we will use a vertex coloring for the graph and update the nodes by color. A quick
    # greedy coloring will be sufficient.
    __, colors = greedy_coloring(adj)

    # let's make our initial guess (randomly)
    spins = {v: random.choice((-1, 1)) for v in h}

    # there are exactly as many betas as sweeps
    for beta in betas:

        # we want to know the gain in energy for flipping each of the spins.
        # We can calculate all of the linear terms simultaneously
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
            #   log(p) < -beta * (energy_diff)
            for v in nodes:
                logp = math.log(random.uniform(0, 1))
                if logp < -1. * beta * (energy_diff_h[v] + energy_diff_J[v]):
                    # flip the variable in the spins
                    spins[v] *= -1

    return spins, ising_energy(spins, h, J)


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
        This is a greedy heuristic: the resulting coloring is not
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
