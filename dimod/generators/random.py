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

import collections.abc as abc

import numpy as np

from dimod.bqm import AdjVectorBQM
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.decorators import graph_argument

__all__ = ['gnm_random_bqm', 'gnp_random_bqm', 'uniform', 'ran_r', 'randint']


def gnm_random_bqm(variables, num_interactions, vartype, *,
                   cls=AdjVectorBQM,
                   random_state=None,
                   bias_generator=None):
    """Generate a random BQM with a fixed number of variables and interactions.

    Args:
        variables (int/sequence):
            The variables. If `variables` is an int, the variables are labelled
            as `[0, variables)`.

        num_interactions (int): The number of interactions.

        vartype (:class:`.Vartype`/str/set):
            Variable type for the BQM. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        cls (type, optional):
            Binary quadratic model class to build from. Default is
            :class:`.AdjVectorBQM`.

        random_state (:class:`numpy.random.RandomState`/int, optional):
            A random seed or a random state generator. Used for generating
            the structure of the BQM and, if `bias_generator` is not given, for
            the bias generation.

        bias_generator (callable, optional):
            Bias generating function.
            Should accept a single argument `n` and return an
            :class:`~numpy.ndarray` of biases of length `n`.
            May be called multiple times.
            If not provided, :meth:numpy.random.RandomState.uniform is used by
            default.

    Returns:
        A binary quadratic model of type `cls`.

    """

    if isinstance(variables, abc.Sequence):
        labels = variables
        num_variables = len(labels)
    else:
        labels = range(variables)
        num_variables = variables

    if num_variables < 0:
        raise ValueError('num_variables must not be negative')
    if num_interactions < 0:
        raise ValueError('num_interactions must not be negative')

    # upper bound to complete graph
    num_interactions = min(num_variables*(num_variables-1)//2,
                           num_interactions)

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    if bias_generator is None:
        def bias_generator(n):
            return random_state.uniform(size=n)

    bqm = cls(vartype=vartype)

    for vi, bias in enumerate(bias_generator(num_variables)):
        bqm.set_linear(labels[vi], bias)

    qbias = bias_generator(num_interactions)

    if num_interactions:
        ui = 0
        vi = 1
        k = 0
        for t in range(num_interactions):
            # this randint one-at-a-time actually dominates the runtime, there
            # is some stuff we can do to imporve performance but it gets into
            # cython territory quickly so I think this is fine for now
            if random_state.randint(num_interactions - t) < num_interactions - k:
                bqm.set_quadratic(labels[ui], labels[vi], qbias[k])
                k += 1
                if k == num_interactions:
                    break
            vi += 1
            if vi == num_variables:  # go to next row of adjacency matrix
                ui += 1
                vi = ui + 1

    bqm.offset, = bias_generator(1)

    return bqm


def gnp_random_bqm(n, p, vartype,
                   cls=AdjVectorBQM,
                   random_state=None, bias_generator=None):
    """Generate a BQM structured as an Erdős-Rényi graph.

    Args:
        n (int/sequence):
            The variables. If `n` is an int, the variables are labelled as
            `[0, n)`.

        p (float): The probability for interaction creation.

        vartype (:class:`.Vartype`/str/set):
            Variable type for the BQM. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        cls (type, optional):
            Binary quadratic model class to build from. Default is
            :class:`.AdjVectorBQM`.

        random_state (:class:`numpy.random.RandomState`/int, optional):
            A random seed or a random state generator. Used for generating
            the structure of the BQM and, if `bias_generator` is not given, for
            the bias generation.

        bias_generator (callable, optional):
            Bias generating function.
            Should accept a single argument `n` and return an
            :class:`~numpy.ndarray` of biases of length `n`.
            May be called multiple times.
            If not provided, :meth:numpy.random.RandomState.uniform is used by
            default.

    Returns:
        A binary quadratic model of type `cls`.

    Notes:
        This algorithm runs in O(n^2) time and space.

    """
    if isinstance(n, abc.Sequence):
        labels = n
        n = len(labels)
    else:
        labels = None

    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)

    neighborhoods = []
    num_interactions = 0
    for v in range(n):
        # determine what variables are connected
        exists = random_state.uniform(size=(n - v - 1)) < p
        neighbors = np.arange(v+1, n)[exists]

        neighborhoods.append(neighbors)
        num_interactions += len(neighbors)

    # construct irow, icol
    irow = np.empty(num_interactions, dtype=np.int)
    icol = np.empty(num_interactions, dtype=np.int)

    q = 0
    for v, neighbors in enumerate(neighborhoods):
        irow[q:q+len(neighbors)] = v
        icol[q:q+len(neighbors)] = neighbors
        q += len(neighbors)

    # calculate the biases
    if bias_generator is None:
        def bias_generator(n):
            return random_state.uniform(size=n)

    ldata = bias_generator(n)
    qdata = bias_generator(num_interactions)
    offset, = bias_generator(1)

    return cls.from_numpy_vectors(ldata, (irow, icol, qdata), offset, vartype,
                                  variable_order=labels)


@graph_argument('graph')
def uniform(graph, vartype, low=0.0, high=1.0, cls=BinaryQuadraticModel,
            seed=None):
    """Generate a binary quadratic model with random biases and offset.

    Biases and offset are drawn from a uniform distribution range (low, high).

    Args:
        graph (int/tuple[nodes, edges]/list[edge]/:obj:`~networkx.Graph`):
            The graph to build the binary quadratic model (BQM) on. Either an
            integer n, interpreted as a complete graph of size n, a nodes/edges
            pair, a list of edges or a NetworkX graph.

        vartype (:class:`.Vartype`/str/set):
            Variable type for the BQM. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        low (float, optional, default=0.0):
            The low end of the range for the random biases.

        high (float, optional, default=1.0):
            The high end of the range for the random biases.

        cls (type, optional):
            Binary quadratic model class to build from. Default is
            :class:`.BinaryQuadraticModel`.

        seed (int, optional, default=None):
            Random seed.

    Returns:
        :obj:`.BinaryQuadraticModel`

    """
    if seed is None:
        seed = np.random.randint(2**32, dtype=np.uint32)
    r = np.random.RandomState(seed)

    variables, edges = graph

    index = {v: idx for idx, v in enumerate(variables)}

    if edges:
        irow, icol = zip(*((index[u], index[v]) for u, v in edges))
    else:
        irow = icol = tuple()

    ldata = r.uniform(low, high, size=len(variables))
    qdata = r.uniform(low, high, size=len(irow))
    offset = r.uniform(low, high)

    return cls.from_numpy_vectors(ldata, (irow, icol, qdata), offset, vartype,
                                  variable_order=variables)


@graph_argument('graph')
def randint(graph, vartype, low=0, high=1, cls=BinaryQuadraticModel,
            seed=None):
    """Generate a binary quadratic model with random biases and offset.

    Biases and offset are integer-valued in range [low, high] inclusive.

    Args:
        graph (int/tuple[nodes, edges]/list[edge]/:obj:`~networkx.Graph`):
            The graph to build the binary quadratic model (BQM) on. Either an
            integer n, interpreted as a complete graph of size n, a nodes/edges
            pair, a list of edges or a NetworkX graph.

        vartype (:class:`.Vartype`/str/set):
            Variable type for the BQM. Accepted input values:

            * :class:`.Vartype.SPIN`, ``'SPIN'``, ``{-1, 1}``
            * :class:`.Vartype.BINARY`, ``'BINARY'``, ``{0, 1}``

        low (float, optional, default=0):
            The low end of the range for the random biases.

        high (float, optional, default=1):
            The high end of the range for the random biases.

        cls (type, optional):
            Binary quadratic model class to build from. Default is
            :class:`.BinaryQuadraticModel`.

        seed (int, optional, default=None):
            Random seed.

    Returns:
        :obj:`.BinaryQuadraticModel`

    """
    if seed is None:
        seed = np.random.randint(2**32, dtype=np.uint32)
    r = np.random.RandomState(seed)

    variables, edges = graph

    index = {v: idx for idx, v in enumerate(variables)}

    if edges:
        irow, icol = zip(*((index[u], index[v]) for u, v in edges))
    else:
        irow = icol = tuple()

    # high+1 for inclusive range
    ldata = r.randint(low, high+1, size=len(variables))
    qdata = r.randint(low, high+1, size=len(irow))
    offset = r.randint(low, high+1)

    return cls.from_numpy_vectors(ldata, (irow, icol, qdata), offset, vartype,
                                  variable_order=variables)


@graph_argument('graph')
def ran_r(r, graph, cls=BinaryQuadraticModel, seed=None):
    """Generate an Ising model for a RANr problem.

    In RANr problems all linear biases are zero and quadratic values are uniformly
    selected integers between -r to r, excluding zero. This class of problems is relevant
    for binary quadratic models (BQM) with spin variables (Ising models).

    This generator of RANr problems follows the definition in [Kin2015]_\ .

    Args:
        r (int):
            Order of the RANr problem.

        graph (int/tuple[nodes, edges]/list[edge]/:obj:`~networkx.Graph`):
            The graph to build the bqm on. Either an integer n,
            interpreted as a complete graph of size n, a nodes/edges pair, a
            list of edges or a NetworkX graph.

        cls (type, optional):
            Binary quadratic model class to build from. Default is
            :class:`.BinaryQuadraticModel`.

        seed (int, optional, default=None):
            Random seed.

    Returns:
        :obj:`.BinaryQuadraticModel`.

    Examples:

    >>> import networkx as nx
    >>> K_7 = nx.complete_graph(7)
    >>> bqm = dimod.generators.random.ran_r(1, K_7)
    >>> max(bqm.quadratic.values()) == -min(bqm.quadratic.values())
    True

    .. [Kin2015] James King, Sheir Yarkoni, Mayssam M. Nevisi, Jeremy P. Hilton,
        Catherine C. McGeoch. Benchmarking a quantum annealing processor with the
        time-to-target metric. https://arxiv.org/abs/1508.05087

    """
    if not isinstance(r, int):
        raise TypeError("r should be a positive integer")
    if r < 1:
        raise ValueError("r should be a positive integer")

    if seed is None:
        seed = np.random.randint(2**32, dtype=np.uint32)
    rnd = np.random.RandomState(seed)

    variables, edges = graph

    index = {v: idx for idx, v in enumerate(variables)}

    if edges:
        irow, icol = zip(*((index[u], index[v]) for u, v in edges))
    else:
        irow = icol = tuple()

    ldata = np.zeros(len(variables))

    rvals = np.empty(2*r)
    rvals[0:r] = range(-r, 0)
    rvals[r:] = range(1, r+1)
    qdata = rnd.choice(rvals, size=len(irow))

    offset = 0

    return cls.from_numpy_vectors(ldata, (irow, icol, qdata), offset, vartype='SPIN',
                                  variable_order=variables)


@graph_argument('graph')
def doped(p, graph, cls=BinaryQuadraticModel, seed=None, fm=True):
    """Generate a BQM for a doped ferromagnetic (FM) or antiferromagnetic (AFM) problem.

    In a doped FM problem, `p`, the doping parameter, determines the probability of
    couplers set to AFM (flipped to 1). The remaining couplers remain FM (-1). In a doped
    AFM problem, the opposite is true.

    Args:
        p (float):
            Doping parameter [0,1] determines the probability of couplers flipped.

        graph (int/tuple[nodes, edges]/list[edge]/:obj:`~networkx.Graph`):
            The graph to build the bqm on. Either an integer n,
            interpreted as a complete graph of size n, a nodes/edges pair, a
            list of edges or a NetworkX graph.

        cls (type, optional):
            Binary quadratic model class to build from. Default is
            :class:`.BinaryQuadraticModel`.

        seed (int, optional, default=None):
            Random seed.

        fm (bool):
            If True, the default undoped graph is FM. If False, it is AFM.

    Returns:
        :obj:`.BinaryQuadraticModel`

    """

    if seed is None:
        seed = np.random.randint(2**32, dtype=np.uint32)
    rnd = np.random.RandomState(seed)

    if p>1 or p<0:
        raise ValueError('Doping must be in the range [0,1]')

    variables, edges = graph

    if not fm:
        p = 1 - p

    bqm = cls({},{},0, 'SPIN')

    for u, v in edges:
        bqm.set_linear(u, 0)
        bqm.set_linear(v, 0)

        J = rnd.choice([1,-1], p=[p, 1 - p])
        bqm.add_interaction(u, v, J)

    return bqm
