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
            The binary quadratic model to be embedded.

        embedding (dict):
            Mapping from the source graph to the target graph.
            Should be of the form {v: {ls, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.

        target_adjacency (dict/:class:`networkx.Graph`):
            Adjacency dict of the target
            graph. Should be a dict of the form {s: Ns, ...} where s is a variable
            in the target graph and Ns is the set of neighbours of s.

        chain_strength (float, optional):
            Magnitude quadratic bias (in SPIN-space) that should be used to create chains. Note
            that the energy penalty of chain breaks will be 2 * chain_strength.

        embed_singleton_variables (bool, True):
            Attempt to allocate singleton variables in the source binary quadratic model to unused
            nodes in the target graph.

    Returns:
        :obj:`.BinaryQuadraticModel`: The target binary quadratic model.

    Examples:

        .. code-block:: python

            import networkx as nx

            # Get a binary quadratic model, source graph in this case is a triangle
            bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1})

            # Target graph is a square to a square graph
            target = nx.cycle_graph(4)

            # Make an embedding from the source graph to target graph
            embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

            # embed the bqm
            target_bqm = dimod.embed_bqm(bqm, embedding, target)

        Embedding to a dimod Sampler

        .. code-block:: python

            # Get a binary quadratic model, source graph in this case is a triangle
            bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1})

            # get a structured dimod sampler with a structure defined by a square graph
            sampler = dimod.StructureComposite(dimod.ExactSolver(), [0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (0, 3)])

            # Make an embedding from the source graph to target graph
            embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

            # embed the bqm
            target_bqm = dimod.embed_bqm(bqm, embedding, sampler.adjacency)

            # sample
            response = sampler.sample(target_bqm)


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

    # go ahead and add the offset
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
            Mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.

        target_adjacency (dict/:class:`networkx.Graph`):
            Adjacency dict of the target
            graph. Should be a dict of the form {s: Ns, ...} where s is a variable
            in the target graph and Ns is the set of neighbours of s.

        chain_strength (float, optional):
            Magnitude quadratic bias (in SPIN-space) that should be used to create chains. Note
            that the energy penalty of chain breaks will be 2 * chain_strength.

        embed_singleton_variables (bool, True):
            Attempt to allocate singleton variables in the source binary quadratic model to unused
            nodes in the target graph.

    Returns:
        tuple: A 2-tuple:

            dict[variable, bias]: Linear biases of the target Ising problem.

            dict[(variable, variable), bias]: Quadratic biases of the target Ising problem.

    Examples:

        .. code-block:: python

            import networkx as nx

            # Get an Ising problem, source graph in this case is a triangle
            h = {}
            J = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}

            # Target graph is a square to a square graph
            target = nx.cycle_graph(4)

            # Make an embedding from the source graph to target graph
            embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

            # embed the Ising problem
            target_h, target_J = dimod.embed_ising(h, J, embedding, target)

        Embedding to a dimod Sampler

        .. code-block:: python

            # Get an Ising problem, source graph in this case is a triangle
            h = {}
            J = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}

            # get a structured dimod sampler with a structure defined by a square graph
            sampler = dimod.StructureComposite(dimod.ExactSolver(), [0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (0, 3)])

            # Make an embedding from the source graph to target graph
            embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

            # embed the Ising problem
            target_h, target_J = dimod.embed_ising(h, J, embedding, sampler.adjacency)

            # sample
            response = sampler.sample_ising(target_h, target_J)

    """
    source_bqm = BinaryQuadraticModel.from_ising(souce_h, source_J)
    target_bqm = embed_bqm(source_bqm, embedding, target_adjacency, chain_strength=chain_strength,
                           embed_singleton_variables=embed_singleton_variables)
    target_h, target_J, __ = target_bqm.to_ising()
    return target_h, target_J


def embed_qubo(source_Q, embedding, target_adjacency, chain_strength=1.0, embed_singleton_variables=True):
    """Embed a quadratic unconstrained binary optimization (QUBO) onto a target graph.

    Args:
        source_Q (dict[(variable, variable), bias]):
            Coefficients of a quadratic unconstrained binary optimization (QUBO) model.

        embedding (dict):
            Mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.

        target_adjacency (dict/:class:`networkx.Graph`):
            Adjacency dict of the target
            graph. Should be a dict of the form {s: Ns, ...} where s is a variable
            in the target graph and Ns is the set of neighbours of s.

        chain_strength (float, optional):
            Magnitude quadratic bias (in SPIN-space) that should be used to create chains. Note
            that the energy penalty of chain breaks will be 2 * chain_strength.

        embed_singleton_variables (bool, True):
            Attempt to allocate singleton variables in the source binary quadratic model to unused
            nodes in the target graph.

    Returns:
        dict[(variable, variable), bias]: Quadratic biases of the target QUBO.

    Examples:

        .. code-block:: python

            import networkx as nx

            # Get a QUBO, source graph in this case is a triangle
            Q = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}

            # Target graph is a square to a square graph
            target = nx.cycle_graph(4)

            # Make an embedding from the source graph to target graph
            embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

            # embed the QUBO
            target_Q = dimod.embed_qubo(Q, embedding, target)

        Embedding to a dimod Sampler

        .. code-block:: python

            # Get a QUBO, source graph in this case is a triangle
            Q = {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1}

            # get a structured dimod sampler with a structure defined by a square graph
            sampler = dimod.StructureComposite(dimod.ExactSolver(), [0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (0, 3)])

            # Make an embedding from the source graph to target graph
            embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

            # embed the QUBO
            target_Q = dimod.embed_qubo(Q, embedding, sampler.adjacency)

            # sample
            response = sampler.sample_qubo(target_Q)

    """
    source_bqm = BinaryQuadraticModel.from_qubo(source_Q)
    target_bqm = embed_bqm(source_bqm, embedding, target_adjacency, chain_strength=chain_strength,
                           embed_singleton_variables=embed_singleton_variables)
    target_Q, __ = target_bqm.to_qubo()
    return target_Q


def iter_unembed(target_samples, embedding, chain_break_method=None):
    """Yield unembedded samples.

    Args:
        target_samples (iterable[mapping[variable, value]]):
            Iterable of samples. Each sample of the form {v: val, ...} where v is a variable in the
            target graph and val is the associated value.

        embedding (dict):
            Mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.

        chain_break_method (function, optional, default=:func:`.majority_vote`):
            The method used to resolve chain breaks.

    Yields:
        dict[variable, value]: An unembedded sample of the form {v: val, ...} where v is a variable
        in the source graph and val is the associated value.

    Notes:
        This is implemented as an iterator to keep down memory footprint. It is intended to be used
        with :meth:`.Response.from_dicts`.

    Examples:

        >>> # Get a collection of samples from the target problem (say a square)
        >>> samples = [{0: +1, 1: -1, 2: +1, 3: +1},
        ...            {0: -1, 1: -1, 2: -1, 3: -1},
        ...            {0: +1, 1: +1, 2: +1, 3: +1}]
        >>> # Make an embedding from the source graph to target graph
        >>> embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}
        >>> for source_sample in dimod.iter_unembed(samples, embedding):  # doctest: +SKIP
        ...     print(source_sample)
        {'a': 1, 'b': -1, 'c': 1}
        {'a': -1, 'b': -1, 'c': -1}
        {'a': 1, 'b': 1, 'c': 1}

        An example using both embedding an unembedding. Note that this is the flow used by
        dwave-system_'s :obj:`~dwave.system.composites.EmbeddingComposite`.

        .. code-block:: python

            # Get a binary quadratic model, source graph in this case is a triangle
            bqm = dimod.BinaryQuadraticModel.from_ising({}, {('a', 'b'): 1, ('b', 'c'): 1, ('a', 'c'): 1})

            # get a structured dimod sampler with a structure defined by a square graph
            sampler = dimod.StructureComposite(dimod.ExactSolver(), [0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (0, 3)])

            # Make an embedding from the source graph to target graph
            embedding = {'a': {0}, 'b': {1}, 'c': {2, 3}}

            # embed the bqm
            target_bqm = dimod.embed_bqm(bqm, embedding, sampler.adjacency)

            # get a response in the form of a target structure
            target_response = sampler.sample(target_bqm)

            # now unembed
            source_response = dimod.Response.from_dicts(dimod.iter_unembed(target_response, embedding),
                                                        {'energy': target_response.data_vectors['energy']})

    .. _dwave-system: https://github.com/dwavesystems/dwave-system

    """
    if chain_break_method is None:
        chain_break_method = majority_vote

    for target_sample in target_samples:
        for source_sample in chain_break_method(target_sample, embedding):
            yield source_sample


def unembed_response(target_response, embedding, source_bqm, chain_break_method=None):
    """Unembed the response to construct a response for the source bqm.

    Args:
        target_response (:obj:`.Response`):
            A response from the target bqm

        embedding (dict):
            Mapping from the source graph to the target graph.
            Should be of the form {v: {s, ...}, ...} where v is a variable in the
            source model and s is a variable in the target model.

        source_bqm (:obj:`.BinaryQuadraticModel`):
            The source binary quadratic model.

        chain_break_method (function, optional, default=:func:`.majority_vote`):
            The method used to resolve chain breaks.

    Returns:
        :obj:`.Response`

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
