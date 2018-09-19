import networkx as nx
import collections
import itertools
from ezdw.iterator import pairwise, is_unique
from ezdw import chimera
import numpy as np
import networkx


def generate_frustrated_loop_problem(edges, num_cycles, random, min_cycle_length=8, max_failed_cycles=100):
    """
    Generate a frustrated loop problem as per Hen et al. (https://arxiv.org/abs/1502.01663v2).

    :param edges: edges on which the problem should be generated.
    :param num_cycles: number of cycles/loops to generate.
    :param random: the random state used for sampling.
    :param min_cycle_length: the minimum acceptable length of a cycle.
    :param max_failed_cycles: max number of failed/bad cycles we can encounter during generation before failing.
    :return: a frustrated loop problem.
    """

    def cycle_length_predicate(cycle): return len(cycle) >= min_cycle_length

    return generate_generic_frustrated_loop_problem(
        edges=edges,
        num_cycles=num_cycles,
        random=random,
        cycle_predicates=(cycle_length_predicate,),
        max_failed_cycles=max_failed_cycles,
    )


def generate_limited_frustrated_loop_problem(edges, num_cycles, R, random, dims=None, max_failed_cycles=100):
    """
    Generate a native limited frustrated loop problem as per King et al. (https://arxiv.org/abs/1502.02098).

    :param edges: Chimera edges on which the problem should be generated.
    :param num_cycles: number of cycles/loops to generate.
    :param R: precision limit.
    :param random: the random state used for sampling.
    :param dims: Chimera dimensions tuple (see arguments to `isingproblems.native.chimera.get_chimera_dimensions`). If
    `None`, smallest square MxMx4 Chimera graph is inferred from edges.
    :param max_failed_cycles: max number of failed/bad cycles we can encounter during generation before failing.
    :return: a limited frustrated loop problem.
    """
    if dims is None:
        n = max(i for e in edges for i in e)+1
        dims = (chimera.get_min_square_chimera_dimensions(n),)

    def cycle_in_chimera_tile_predicate(cycle): return is_unique(chimera.to_chimera_index(c, *dims)[:2] for c in cycle)

    return generate_generic_frustrated_loop_problem(
        edges=edges,
        num_cycles=num_cycles,
        R=R,
        random=random,
        cycle_predicates=(cycle_in_chimera_tile_predicate, ),
        max_failed_cycles=max_failed_cycles,
    )


def generate_generic_frustrated_loop_problem(edges, num_cycles, random, R=np.inf, cycle_predicates=tuple(), max_failed_cycles=100):
    """
    A (generic) frustrated loop (FL) problem is a sum of Hamiltonians, each generated from a single "good" loop.
        1. Generate a loop by random walking on the support graph.
        2. If the cycle is "good" (according to provided predicates), continue, else go to 1.
        3. Choose one edge of the loop to be anti-ferromagnetic; all other edges are ferromagnetic.
        4. Add the loop's coupler values to the FL problem.
    If at any time the magnitude of a coupler in the FL problem exceeds a given precision `R`, remove that coupler
    from consideration in the loop generation procedure.

    This is a generic generator of FL problems that encompasses both the original FL problem definition from Hen et al.
    (https://arxiv.org/abs/1502.01663v2) and the limited FL problem definition from King et al.
    (https://arxiv.org/abs/1502.02098).

    :param edges: edges on which the problem should be generated.
    :param num_cycles: number of good cycles/loops to generate.
    :param random: the random state used for sampling.
    :param R: precision limit.
    :param cycle_predicates: list of predicates taking a cycle iterable and returning True iff the cycle is acceptable.
    :param max_failed_cycles: max number of failed/bad cycles we can encounter during generation before failing.
    :return: a generic frustrated loop problem.
    """
    assert num_cycles > 0
    assert R > 0
    assert max_failed_cycles > 0

    G = nx.Graph(edges)
    J = collections.defaultdict(int)

    failed_cycles = 0
    good_cycles = 0
    while good_cycles < num_cycles and failed_cycles < max_failed_cycles:
        # Get a cycle
        cycle = _get_random_walk_cycle(G, random)

        if cycle is None or not all(pred(cycle) for pred in cycle_predicates):
            # If its a failed cycle, increment failed cycle count.
            failed_cycles += 1
        else:
            # If its a good cycle, modify J with it.
            good_cycles += 1
            # Update edge counts, removing edges with two large of a count.
            # Chose random anti-ferromagnetic edge.
            index = random.randint(len(cycle))
            for i, (u, v) in enumerate(itertools.islice(pairwise(itertools.cycle(cycle)), len(cycle))):
                u, v = (u, v) if u <= v else (v, u)
                J[u, v] += (1 if i == index else -1)
                if abs(J[u, v]) == R:
                    G.remove_edge(u, v)

    assert good_cycles >= num_cycles, 'Found %d of the necessary %d good cycles (failed %d times).' % (good_cycles, num_cycles, failed_cycles)

    J = dict(J)

    return J


def _get_random_walk_cycle(G, random):
    """
    :param G: a graph.
    :param random: the random state used for sampling.
    :return: a cycle in G of length at least 3 that was obtained by random walking through the graph; `None` if no cycle
     could be found.
    """

    start = list(G)[random.randint(G.number_of_nodes())]
    walk = [start]
    visited = {start}

    while True:
        degree = G.degree(walk[-1])
        neighbors = G.neighbors(walk[-1])
        if len(walk) > 1:  # Avoid going back in one step.
            degree -= 1
            neighbors = itertools.ifilter(lambda u: u != walk[-2], neighbors)
        if degree == 0:
            return None
        index = random.randint(degree)
        neighbor = next(itertools.islice(neighbors, index, index+1))
        if neighbor in visited:
            return walk[walk.index(neighbor):]
        else:
            walk.append(neighbor)
            visited.add(neighbor)


def _generate_lattice_fcl(L, alpha, random, connected=True, periodic=False, max_failed_cycles=1000):
    lattice = networkx.grid_2d_graph(L, L, periodic=periodic)
    for _ in itertools.count():
        try:
            J = generate_generic_frustrated_loop_problem(
                edges=lattice.edges(),
                random=random,
                num_cycles=L*L*alpha,
                R=1,
                max_failed_cycles=max_failed_cycles,
            )
        except:
            continue
        H = networkx.Graph()
        for p, q in J:
            H.add_edge(p, q)
        if not connected or networkx.connected.is_connected(H):
            return J
    raise ValueError('Could not generate a valid problem.')


def generate_DCL(L, alpha, ell, R, random, connected=True, periodic=False):
    J_lat = _generate_lattice_fcl(L, alpha, random, connected=connected, periodic=periodic)
    J_chim= {}
    dims = (L, L, 4)
    edges = list(chimera.edge_iter(*dims))
    if periodic:
        periodic_horizontal_edges = [((i, L - 1, 1, k), (i, 0, 1, k)) for i in xrange(L) for k in xrange(4)]
        periodic_vertical_edges = [((L - 1, j, 0, k), (0, j, 0, k)) for j in xrange(L) for k in xrange(4)]
        edges.extend(periodic_horizontal_edges)
        edges.extend(periodic_vertical_edges)

    for e in edges:
        v = None
        if chimera.is_intracell_edge(*e):
            v = -1./ell
        else:
            (i1, j1, u1, k1), (i2, j2, u2, k2) = e
            p = (i1, j1)
            q = (i2, j2)
            if (p, q) in J_lat or (q, p) in J_lat:
                v = J_lat[p, q] if (p, q) in J_lat else J_lat[q, p]
                v *= 1. / R
        if v is not None:
            cindex1, cindex2 = e
            lindex1 = chimera.ctol(*(cindex1 + dims))
            lindex2 = chimera.ctol(*(cindex2 + dims))
            J_chim[lindex1, lindex2] = v
    h = [0] * chimera.num_nodes(*dims)
    return h, J_chim
