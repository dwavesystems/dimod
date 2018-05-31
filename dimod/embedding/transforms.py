from __future__ import division

import itertools

from six import iteritems, itervalues

from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.embedding.chain_breaks import majority_vote
from dimod.embedding.utils import chain_to_quadratic
from dimod.response import Response
from dimod.vartypes import Vartype

__all__ = ['embed_bqm', 'embed_ising', 'embed_qubo', 'iter_unembed', 'unembed_response']


def embed_bqm(source_bqm, embedding, target_adjacency, chain_strength=1.0, embed_singleton_variables=True):
    """Embed a binary quadratic model onto a target graph.

    Args:
        source_bqm (:obj:`.BinaryQuadraticModel`):
            Binary quadratic model to embed.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: {t, ...}, ...},
            where s is a source-model variable and t is a target-model variable.

        target_adjacency (dict/:class:`networkx.Graph`):
            Adjacency of the target graph as a dict of form {t: Nt, ...},
            where t is a variable in the target graph and Nt is its set of neighbours.

        chain_strength (float, optional):
            Magnitude of the quadratic bias (in SPIN-space) applied between variables to create chains. Note
            that the energy penalty of chain breaks is 2 * `chain_strength`.

        embed_singleton_variables (bool, True):
            Attempt to allocate singleton variables in the source binary quadratic model to unused
            nodes in the target graph.

    Returns:
        :obj:`.BinaryQuadraticModel`: Target binary quadratic model.

    Examples:
        This example embeds a fully connected :math:`K_3` graph onto a square target graph.
        Embedding is accomplished by an edge contraction operation on the target graph:
        target-nodes 2 and 3 are chained to represent source-node c.

        >>> import dimod
        >>> import networkx as nx
        >>> # Binary quadratic model for a triangular source graph
        >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1})
        >>> # Target graph is a graph
        >>> target = nx.cycle_graph(4)
        >>> # Embedding from source to target graphs
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> # Embed the BQM
        >>> target_bqm = dimod.embed_bqm(bqm, embedding, target)
        >>> target_bqm.quadratic[(0, 1)] == bqm.quadratic[('a', 'b')]
        True
        >>> target_bqm.quadratic   # doctest: +SKIP
        {(0, 1): 1.0, (0, 3): 1.0, (1, 2): 1.0, (2, 3): -1.0}

        This example embeds a fully connected :math:`K_3` graph onto the target graph
        of a dimod reference structured sampler, `StructureComposite`, using the dimod reference
        `ExactSolver` sampler with a square graph specified. Target-nodes 2 and 3
        are chained to represent source-node c.

        >>> import dimod
        >>> # Binary quadratic model for a triangular source graph
        >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1})
        >>> # Structured dimod sampler with a structure defined by a square graph
        >>> sampler = dimod.StructureComposite(dimod.ExactSolver(), [0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (0, 3)])
        >>> # Embedding from source to target graph
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> # Embed the BQM
        >>> target_bqm = dimod.embed_bqm(bqm, embedding, sampler.adjacency)
        >>> # Sample
        >>> response = sampler.sample(target_bqm)
        >>> response.samples_matrix   # doctest: +SKIP
        matrix([[-1, -1, -1, -1],
                [ 1, -1, -1, -1],
                [ 1,  1, -1, -1],
                [-1,  1, -1, -1],
                [-1,  1,  1, -1],
        >>> # Snipped above response for brevity

    """
    if embed_singleton_variables:
        unused = set(target_adjacency)
        unused.difference_update(*embedding.values())
        if unused:
            # if there are unused variables then we may be alterning the embedding so we need to
            # make a copy, we'll be adding new key/value pairs so we only need a shallow copy
            embedding = embedding.copy()
    else:
        unused = False

    # create a new empty binary quadratic model with the same class as source_bqm
    target_bqm = source_bqm.empty(source_bqm.vartype)

    # add the offset
    target_bqm.add_offset(source_bqm.offset)

    # start with the linear biases, spreading the source bias equally over the target variables in
    # the chain
    for v, bias in iteritems(source_bqm.linear):

        if v in embedding:
            chain = embedding[v]
        elif embed_singleton_variables and unused:
            # if a variable is in the source_bqm but is not mentioned in embedding, then we try to map
            # it to a source node not already mentioned in embedding
            chain = embedding[v] = {unused.pop()}
        else:
            raise ValueError('no embedding provided for source variable {}'.format(v))

        if any(u not in target_adjacency for u in chain):
            raise ValueError('chain variable {} not in target_adjacency'.format(v))

        b = bias / len(chain)

        target_bqm.add_variables_from({u: b for u in chain})

    # next up the quadratic biases, spread the quadratic biases evenly over the available
    # interactions
    for (u, v), bias in iteritems(source_bqm.quadratic):
        available_interactions = {(s, t) for s in embedding[u] for t in embedding[v] if s in target_adjacency[t]}

        if not available_interactions:
            raise ValueError("no edges in target graph between source variables {}, {}".format(u, v))

        b = bias / len(available_interactions)

        target_bqm.add_interactions_from((u, v, b) for u, v in available_interactions)

    for chain in itervalues(embedding):

        # in the case where the chain has length 1, there are no chain quadratic biases, but we
        # none-the-less want the chain variables to appear in the target_bqm
        if len(chain) == 1:
            v, = chain
            target_bqm.add_variable(v, 0.0)
            continue

        quadratic_chain_biases = chain_to_quadratic(chain, target_adjacency, chain_strength)
        target_bqm.add_interactions_from(quadratic_chain_biases, vartype=Vartype.SPIN)  # these are spin

        # add the energy for satisfied chains to the offset
        energy_diff = -sum(itervalues(quadratic_chain_biases))
        target_bqm.add_offset(energy_diff)

    return target_bqm


def embed_ising(souce_h, source_J, embedding, target_adjacency, chain_strength=1.0, embed_singleton_variables=True):
    """Embed an Ising problem onto a target graph.

    Args:
        source_h (dict[variable, bias]/list[bias]):
            Linear biases of the Ising problem. If a list, the list's indices are used as
            variable labels.

        source_J (dict[(variable, variable), bias]):
            Quadratic biases of the Ising problem.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: {t, ...}, ...},
            where s is a source-model variable and t is a target-model variable.

        target_adjacency (dict/:class:`networkx.Graph`):
            Adjacency of the target graph as a dict of form {t: Nt, ...},
            where t is a target-graph variable and Nt is its set of neighbours.

        chain_strength (float, optional):
            Magnitude of the quadratic bias (in SPIN-space) applied between variables to form a chain. Note
            that the energy penalty of chain breaks is 2 * `chain_strength`.

        embed_singleton_variables (bool, True):
            Attempt to allocate singleton variables in the source binary quadratic model to unused
            nodes in the target graph.

    Returns:
        tuple: A 2-tuple:

            dict[variable, bias]: Linear biases of the target Ising problem.

            dict[(variable, variable), bias]: Quadratic biases of the target Ising problem.

    Examples:
        This example embeds a fully connected :math:`K_3` graph onto a square target graph.
        Embedding is accomplished by an edge contraction operation on the target graph: target-nodes
        2 and 3 are chained to represent source-node c.

        >>> import dimod
        >>> import networkx as nx
        >>> # Ising problem for a triangular source graph
        >>> h = {}
        >>> J = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}
        >>> # Target graph is a square graph
        >>> target = nx.cycle_graph(4)
        >>> # Embedding from source to target graph
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> # Embed the Ising problem
        >>> target_h, target_J = dimod.embed_ising(h, J, embedding, target)
        >>> target_J[(0, 1)] == J[('a', 'b')]
        True
        >>> target_J        # doctest: +SKIP
        {(0, 1): 1.0, (0, 3): 1.0, (1, 2): 1.0, (2, 3): -1.0}

        This example embeds a fully connected :math:`K_3` graph onto the target graph
        of a dimod reference structured sampler, `StructureComposite`, using the dimod reference
        `ExactSolver` sampler with a square graph specified. Target-nodes 2 and 3 are chained to
        represent source-node c.

        >>> import dimod
        >>> # Ising problem for a triangular source graph
        >>> h = {}
        >>> J = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}
        >>> # Structured dimod sampler with a structure defined by a square graph
        >>> sampler = dimod.StructureComposite(dimod.ExactSolver(), [0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (0, 3)])
        >>> # Embedding from source to target graph
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> # Embed the Ising problem
        >>> target_h, target_J = dimod.embed_ising(h, J, embedding, sampler.adjacency)
        >>> # Sample
        >>> response = sampler.sample_ising(target_h, target_J)
        >>> for sample in response.samples(n=3, sorted_by='energy'):   # doctest: +SKIP
        ...     print(sample)
        ...
        {0: 1, 1: -1, 2: -1, 3: -1}
        {0: 1, 1: 1, 2: -1, 3: -1}
        {0: -1, 1: 1, 2: -1, 3: -1}

    """
    source_bqm = BinaryQuadraticModel.from_ising(souce_h, source_J)
    target_bqm = embed_bqm(source_bqm, embedding, target_adjacency, chain_strength=chain_strength,
                           embed_singleton_variables=embed_singleton_variables)
    target_h, target_J, __ = target_bqm.to_ising()
    return target_h, target_J


def embed_qubo(source_Q, embedding, target_adjacency, chain_strength=1.0, embed_singleton_variables=True):
    """Embed a QUBO onto a target graph.

    Args:
        source_Q (dict[(variable, variable), bias]):
            Coefficients of a quadratic unconstrained binary optimization (QUBO) model.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: {t, ...}, ...},
            where s is a source-model variable and t is a target-model variable.

        target_adjacency (dict/:class:`networkx.Graph`):
            Adjacency of the target graph as a dict of form {t: Nt, ...},
            where t is a target-graph variable and Nt is its set of neighbours.

        chain_strength (float, optional):
            Magnitude of the quadratic bias (in SPIN-space) applied between variables to form a chain. Note
            that the energy penalty of chain breaks is 2 * `chain_strength`.

        embed_singleton_variables (bool, True):
            Attempt to allocate singleton variables in the source binary quadratic model to unused
            nodes in the target graph.

    Returns:
        dict[(variable, variable), bias]: Quadratic biases of the target QUBO.

    Examples:
        This example embeds a square source graph onto fully connected :math:`K_5` graph.
        Embedding is accomplished by an edge deletion operation on the target graph: target-node
        0 is not used.

        >>> import dimod
        >>> import networkx as nx
        >>> # QUBO problem for a square graph
        >>> Q = {(1, 1): -4.0, (1, 2): 4.0, (2, 2): -4.0, (2, 3): 4.0,
        ...      (3, 3): -4.0, (3, 4): 4.0, (4, 1): 4.0, (4, 4): -4.0}
        >>> # Target graph is a fully connected k5 graph
        >>> K_5 = nx.complete_graph(5)
        >>> 0 in K_5
        True
        >>> # Embedding from source to target graph
        >>> embedding = {1: {4}, 2: {3}, 3: {1}, 4: {2}}
        >>> # Embed the QUBO
        >>> target_Q = dimod.embed_qubo(Q, embedding, K_5)
        >>> (0, 0) in target_Q
        False
        >>> target_Q     # doctest: +SKIP
        {(1, 1): -4.0,
         (1, 2): 4.0,
         (2, 2): -4.0,
         (2, 4): 4.0,
         (3, 1): 4.0,
         (3, 3): -4.0,
         (4, 3): 4.0,
         (4, 4): -4.0}

        This example embeds a square graph onto the target graph of a dimod reference structured
        sampler, `StructureComposite`, using the dimod reference `ExactSolver` sampler with a
        fully connected :math:`K_5` graph specified.

        >>> import dimod
        >>> import networkx as nx
        >>> # QUBO problem for a square graph
        >>> Q = {(1, 1): -4.0, (1, 2): 4.0, (2, 2): -4.0, (2, 3): 4.0,
        ...      (3, 3): -4.0, (3, 4): 4.0, (4, 1): 4.0, (4, 4): -4.0}
        >>> # Structured dimod sampler with a structure defined by a K5 graph
        >>> sampler = dimod.StructureComposite(dimod.ExactSolver(), list(K_5.nodes), list(K_5.edges))
        >>> sampler.adjacency      # doctest: +SKIP
        {0: {1, 2, 3, 4},
         1: {0, 2, 3, 4},
         2: {0, 1, 3, 4},
         3: {0, 1, 2, 4},
         4: {0, 1, 2, 3}}
        >>> # Embedding from source to target graph
        >>> embedding = {0: {4}, 1: {3}, 2: {1}, 3: {2}}
        >>> # Embed the QUBO
        >>> target_Q = dimod.embed_qubo(Q, embedding, sampler.adjacency)
        >>> # Sample
        >>> response = sampler.sample_qubo(target_Q)
        >>> for datum in response.data():   # doctest: +SKIP
        ...     print(datum)
        ...
        Sample(sample={1: 0, 2: 1, 3: 1, 4: 0}, energy=-8.0)
        Sample(sample={1: 1, 2: 0, 3: 0, 4: 1}, energy=-8.0)
        Sample(sample={1: 1, 2: 0, 3: 0, 4: 0}, energy=-4.0)
        Sample(sample={1: 1, 2: 1, 3: 0, 4: 0}, energy=-4.0)
        Sample(sample={1: 0, 2: 1, 3: 0, 4: 0}, energy=-4.0)
        Sample(sample={1: 1, 2: 1, 3: 1, 4: 0}, energy=-4.0)
        >>> # Snipped above response for brevity

    """
    source_bqm = BinaryQuadraticModel.from_qubo(source_Q)
    target_bqm = embed_bqm(source_bqm, embedding, target_adjacency, chain_strength=chain_strength,
                           embed_singleton_variables=embed_singleton_variables)
    target_Q, __ = target_bqm.to_qubo()
    return target_Q


def iter_unembed(target_samples, embedding, chain_break_method=None):
    """Yield unembedded samples.

    :func:`.iter_unembed` is an iterator (to reduce memory footprint) used by :func:`.unembed_response`
    for unembedding; you may use it directly, for example, to increase performance for responses with
    large numbers of samples of which you need only a small portion, say those with lowest energy.

    Args:
        target_samples (iterable[mapping[variable, value]]):
            Iterable of samples. Each sample is a dict of form {t: val, ...},
            where t is a variable in the target graph and val its associated value.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: {t, ...}, ...},
            where s is a source-model variable and t is a target-model variable.

        chain_break_method (function, optional, default=:func:`.majority_vote`):
            Method used to resolve chain breaks.

    Yields:
        dict[variable, value]: An unembedded sample as a dict of form {s: val, ...},
        where s is a variable in the source graph and val its associated value.

    Examples:
        This example demonstrates the use of :func:`.iter_unembed` to derive samples
        for a triangular source graph from synthetic samples of a square target graph.

        >>> import dimod
        >>> # Synthetic samples from a square-structured target problem
        >>> samples = [{0: +1, 1: -1, 2: +1, 3: +1},
        ...            {0: -1, 1: -1, 2: -1, 3: -1},
        ...            {0: +1, 1: +1, 2: +1, 3: +1}]
        >>> # Embedding from source to target graphs
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> for source_sample in dimod.iter_unembed(samples, embedding):  # doctest: +SKIP
        ...     print(source_sample)
        ...
        {'a': 1, 'b': -1, 'c': 1}
        {'a': -1, 'b': -1, 'c': -1}
        {'a': 1, 'b': 1, 'c': 1}

        This example uses :func:`.iter_unembed` to unembed samples of a binary quadratic model
        for a triangular source graph that is embedded in a square-structured graph with dimod
        reference structured sampler, `StructureComposite`, using the dimod reference `ExactSolver`
        sampler. Note that this flow is used by dwave-system_'s :obj:`~dwave.system.composites.EmbeddingComposite`.

        >>> import dimod
        >>> # Binary quadratic model for a triangular source graph
        >>> bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1})
        >>> # Structured dimod sampler with a structure defined by a square graph
        >>> sampler = dimod.StructureComposite(dimod.ExactSolver(), [0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (0, 3)])
        >>> # Embedding from source to target graph
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> # Embed the BQM
        >>> target_bqm = dimod.embed_bqm(bqm, embedding, sampler.adjacency)
        >>> # Sample (the response is in the form of a target structure)
        >>> target_response = sampler.sample(target_bqm)
        >>> for datum in target_response.data():    # doctest: +SKIP
        ...     print(datum)
        ...
        Sample(sample={0: 1, 1: -1, 2: -1, 3: -1}, energy=-1.0)
        Sample(sample={0: 1, 1: 1, 2: -1, 3: -1}, energy=-1.0)
        Sample(sample={0: -1, 1: 1, 2: -1, 3: -1}, energy=-1.0)
        Sample(sample={0: 1, 1: -1, 2: 1, 3: -1}, energy=-1.0)
        >>> # Snipped above response for brevity
        >>> # Unembed
        >>> source_response = dimod.Response.from_dicts(dimod.iter_unembed(target_response,
        ...                          embedding),
        ...                          {'energy': target_response.data_vectors['energy']})
        >>> for datum in source_response.data():    # doctest: +SKIP
        ...     print(datum)
        ...
        Sample(sample={'a': 1, 'c': -1, 'b': 1}, energy=-1.0)
        Sample(sample={'a': -1, 'c': -1, 'b': 1}, energy=-1.0)
        Sample(sample={'a': 1, 'c': 1, 'b': -1}, energy=-1.0)
        Sample(sample={'a': -1, 'c': 1, 'b': 1}, energy=-1.0)
        >>> # Snipped above response for brevity

    .. _dwave-system: https://github.com/dwavesystems/dwave-system

    """
    if chain_break_method is None:
        chain_break_method = majority_vote

    for target_sample in target_samples:
        for source_sample in chain_break_method(target_sample, embedding):
            yield source_sample


def unembed_response(target_response, embedding, source_bqm, chain_break_method=None):
    """Unembed the response.

    Construct a response for the source binary quadratic model (BQM) by unembedding the given
    response from the target BQM.

    Args:
        target_response (:obj:`.Response`):
            Response from the target BQM.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: {t, ...}, ...},
            where s is a source-model variable and t is a target-model variable.

        source_bqm (:obj:`.BinaryQuadraticModel`):
            Source binary quadratic model.

        chain_break_method (function, optional, default=:func:`.majority_vote`):
            Method used to resolve chain breaks.

    Returns:
        :obj:`.Response`:
            Response for the source binary quadratic model.

    Examples:
        This example embeds a Boolean AND gate,
        :math:`x_3 \Leftrightarrow x_1 \wedge x_2`, in a square-structured
        graph and samples with dimod reference structured sampler,
        `StructureComposite`, using the dimod reference `ExactSolver` sampler.
        The gate is represented as penalty model
        :math:`x_1 x_2 - 2(x_1+x_2)x_3 +3x_3`, which is submitted to the
        sampler as QUBO  problem
        :math:`E(a_i, b_{i,j}; x_i) = 3x_3 + x_1x_2 - 2x_1x_3 - 2x_2x_3`.
        This QUBO represents a fully connected :math:`K_3` graph.
        Samples are unembedded by :func:`.unembed_response` and show that
        only valid states of the AND gate have zero energy (e.g., only input
        :math:`x_1 x_2=1,1` results in :math:`z=1`), while invalid states have
        higher energy.

        >>> import dimod
        >>> import networkx as nx
        >>> # Binary quadratic model for the AND gate
        >>> Q = {('x1', 'x2'): 1, ('x1', 'z'): -2, ('x2', 'z'): -2, ('z', 'z'): 3}
        >>> bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        >>> # Embed the BQM in a structured dimod sampler defined by a square graph
        >>> target_graph = nx.cycle_graph(4)
        >>> sampler = dimod.StructureComposite(dimod.ExactSolver(),
        ...                          list(target_graph.nodes), list(target_graph.edges))
        >>> embedding = {'x1': {0}, 'x2': {1}, 'z': {2, 3}}
        >>> target_Q = dimod.embed_qubo(Q, embedding, sampler.adjacency)
        >>> # Sample on the target graph
        >>> target_response = sampler.sample_qubo(target_Q)
        >>> # Unembed samples back to the problem graph
        >>> source_response = dimod.unembed_response(target_response, embedding, bqm)
        >>> # Verify correct representation of the AND gate (first automatically then manually)
        >>> for datum in source_response.data():
        ...     if (datum.sample['x1'] and datum.sample['x2']) == datum.sample['z']:
        ...         if datum.energy > 0:
        ...            print('Valid AND has high energy')
        ...
        ...     else:
        ...         if datum.energy == 0:
        ...             print('invalid AND has low energy')
        ...
        >>> for datum in source_response.data():     # doctest: +SKIP
        ...     print(datum)
        ...
        Sample(sample={'x2': 0, 'x1': 0, 'z': 0}, energy=0.0)
        Sample(sample={'x2': 0, 'x1': 1, 'z': 0}, energy=0.0)
        Sample(sample={'x2': 1, 'x1': 0, 'z': 0}, energy=0.0)
        Sample(sample={'x2': 1, 'x1': 1, 'z': 1}, energy=0.0)
        Sample(sample={'x2': 1, 'x1': 0, 'z': 0}, energy=0.0)
        Sample(sample={'x2': 0, 'x1': 1, 'z': 0}, energy=0.0)
        Sample(sample={'x2': 0, 'x1': 1, 'z': 0}, energy=0.0)
        Sample(sample={'x2': 0, 'x1': 0, 'z': 0}, energy=0.0)
        Sample(sample={'x2': 1, 'x1': 0, 'z': 0}, energy=0.0)
        Sample(sample={'x2': 0, 'x1': 0, 'z': 0}, energy=0.0)
        Sample(sample={'x2': 1, 'x1': 1, 'z': 0}, energy=1.0)
        Sample(sample={'x2': 0, 'x1': 1, 'z': 1}, energy=1.0)
        Sample(sample={'x2': 1, 'x1': 0, 'z': 1}, energy=1.0)
        Sample(sample={'x2': 1, 'x1': 1, 'z': 0}, energy=1.0)
        Sample(sample={'x2': 1, 'x1': 1, 'z': 0}, energy=1.0)
        Sample(sample={'x2': 0, 'x1': 0, 'z': 1}, energy=3.0)

    """
    if any(v not in embedding for v in source_bqm):
        raise ValueError("given bqm does not match the embedding")

    energies = []

    def _samples():
        # populate energies as the samples are resolved one-at-a-time
        for sample in iter_unembed(target_response, embedding, chain_break_method):
            energies.append(source_bqm.energy(sample))
            yield sample

    # overwrite energy with the new values
    data_vectors = target_response.data_vectors.copy()
    data_vectors['energy'] = energies

    # NB: this works because response.from_dict does not resolve the energies until AFTER the samples
    return target_response.from_dicts(_samples(), data_vectors,
                                      vartype=target_response.vartype,
                                      info=target_response.info)
