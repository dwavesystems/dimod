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

import warnings

from typing import Callable, Collection, List, Mapping, Optional

import numpy as np

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.decorators import graph_argument
from dimod.typing import GraphLike, Variable
from dimod.vartypes import SPIN

__all__ = ['frustrated_loop']


Predicate = Callable[[List[Variable]], bool]


@graph_argument('graph')
def frustrated_loop(graph: GraphLike,
                    num_cycles: int,
                    R: float = float('inf'),
                    cycle_predicates: Collection[Predicate] = tuple(),
                    max_failed_cycles: int = 100,
                    plant_solution: bool = True,
                    planted_solution: Optional[Mapping[Variable, int]] = None,
                    seed: Optional[int] = None,
                    ) -> BinaryQuadraticModel:
    """Generate a frustrated-loop problem.

    A generic frustrated-loop (FL) problem is a sum of Hamiltonians, each 
    generated from a single loop, as follows:

    1. Generate a loop by random walking on the specified graph, ``graph``.
    2. If the cycle meets provided predicates, continue; if not, return to  step 1.
    3. Choose one edge of the loop uniformly at random to be anti-ferromagnetic 
       (AFM, coupling value of `1`) and all others ferromagnetic 
       (`-1`). Or if :code:`plant_solution is False`, sample uniformly couplers
       subject to the constraint that there are an odd number of AFM couplers.
    4. Add the loop's interactions to the FL problem.
       If at any time the absolute value of an interaction in the FL problem,
       accumulated on an edge over good loops, exceeds a given maximum, ``R``,
       remove that edge from consideration in the loop generation procedure.

    This is a generic generator of FL problems that encompasses both the original
    FL problem definition [#HJARTL]_ and the limited FL problem
    definition [#KLH]_.

    Args:
        graph:
            The graph to build the frustrated loops on. Either an integer, `n`,
            interpreted as a complete graph of size `n`, a nodes/edges pair, a
            list of edges, or a NetworkX graph.

        num_cyles:
            Desired number of frustrated cycles.

        R:
            Maximum absolute interaction weight an edge can accumulate from good
            cycles.

        cycle_predicates:
            An iterable of functions, which should accept a cycle and return a bool.

        max_failed_cycles:
            Maximum number of failures to find a cycle before terminating.

        plant_solution: 
            Select frustrated loops with only 1 AFM coupler all 1 (and all -1) 
            solutions are among the ground states. 

        planted_solution:
            A dictionary assigning variables to spin states.  When provided, and
            :code:`planted_solution=True` spin states are relabeled so that the
            variable assignment becomes a planted ground state in place of the 
            all 1 solution.  
            This option is deprecated; use of spin-reversal transforms is 
            recommended for this purpose.

        seed: Random seed.

    .. [#HJARTL] Hen, I., J. Job, T. Albash, T.F. Rønnow, M. Troyer, D. Lidar. Probing for quantum
        speedup in spin glass problems with planted solutions. https://arxiv.org/abs/1502.01663v2

    .. [#KLH] King, A.D., T. Lanting, R. Harris. Performance of a quantum annealer on range-limited
        constraint satisfaction problems. https://arxiv.org/abs/1502.02098

    """
    nodes, edges = graph
    if num_cycles <= 0:
        raise ValueError("num_cycles should be a positive integer")
    if R <= 0:
        raise ValueError("R should be a positive integer")
    if max_failed_cycles <= 0:
        raise ValueError("max_failed_cycles should be a positive integer")
    
    r = np.random.RandomState(seed)

    adj = {v: set() for v in nodes}
    for u, v in edges:
        if u in adj:
            adj[u].add(v)
        else:
            adj[u] = {v}
        if v in adj:
            adj[v].add(u)
        else:
            adj[v] = {u}
    bqm = BinaryQuadraticModel({v: 0.0 for v in nodes}, {edge: 0.0 for edge in edges}, 0.0, SPIN)

    failed_cycles = 0
    good_cycles = 0
    while good_cycles < num_cycles and failed_cycles < max_failed_cycles:

        cycle = _random_cycle(adj, r)

        # if the cycle failed or it is otherwise invalid, mark as failed and continue
        if cycle is None or not all(pred(cycle) for pred in cycle_predicates):
            failed_cycles += 1
            continue
        good_cycles += 1

        if plant_solution:
            # randomly select from all frustrated loops with one AFM edge
            cycle_J = {(cycle[i - 1], cycle[i]) : -1 for i in range(len(cycle))}
            idx = r.randint(len(cycle))
            cycle_J[(cycle[idx - 1], cycle[idx])] = 1
        else:
            # randomly select from all frustrated loops (odd number of AFM edges)
            cycle_J = {(cycle[i], cycle[i+1]) : -1 for i in range(len(cycle)-1)}
            cycle_J[(cycle[-1], cycle[0])] = (1 - 2*(len(cycle_J) & 1))*np.prod(list(cycle_J.values()))
            
        # update the bqm
        bqm.add_interactions_from(cycle_J)
        for u, v in cycle_J:
            if abs(bqm.adj[u][v]) >= R:
                adj[u].remove(v)
                adj[v].remove(u)

    if good_cycles < num_cycles:
        raise RuntimeError(f"number of found cycles, {good_cycles}, is below " 
                           f"requested {num_cycles}")

    if planted_solution is not None:
        warnings.warn("planted_solution is deprecated; use spin-reversal transforms: "
                      "(1) bqm.flip_variable() for superficial reorientation of "
                      "spin labels to match desired planting or "
                      "(2) SpinReversalComposite for presentation of a randomized "
                      "planted solution to some solver.",
                      DeprecationWarning, stacklevel=3)
        # A spin-reversal transform for a BQM with zero linear biases
        for e in bqm.quadratic:
            bqm.set_quadratic(e[0], e[1],
                              bqm.quadratic[e]*planted_solution[e[0]]*planted_solution[e[1]])

    return bqm


def _random_cycle(adj, random_state):
    """Find a cycle using a random graph walk."""

    # step through idx values in adj to pick a random one, random.choice does not work on dicts
    n = random_state.randint(len(adj))
    for idx, v in enumerate(adj):
        if idx == n:
            break
    start = v

    walk = [start]
    visited = {start: 0}

    while True:
        if len(walk) > 1:
            # as long as we don't step back one we won't have any repeated edges
            previous = walk[-2]
            neighbors = [u for u in adj[walk[-1]] if u != previous]
        else:
            neighbors = list(adj[walk[-1]])

        if not neighbors:
            # we've walked into a dead end
            return None

        # in some cases, neighbors will be interpreted as a 2d array by np.random.choice
        # so let's ensure it's always a 1d
        arr = np.empty(len(neighbors), dtype=object)
        arr[:] = neighbors

        # get a random neighbor
        u = random_state.choice(arr)
        if u in visited:
            # if we've seen this neighbour, then we have a cycle starting from it
            return walk[visited[u]:]
        else:
            # add to walk and keep moving
            walk.append(u)
            visited[u] = len(visited)
